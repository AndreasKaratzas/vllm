#!/usr/bin/env python3
"""
Test if cached KV values are EXACTLY the same when read back.

This test instruments the actual vLLM KV cache to capture what values
are written and read, then compares them between runs.
"""

import asyncio
import logging
import sys
import torch

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Global storage for captured KV values
captured_kv = {
    'run1_write': {},  # KV written in run 1
    'run2_read': {},   # KV read from cache in run 2
    'run2_write': {},  # KV written in run 2 (new tokens)
}

def patch_kv_cache_write():
    """Patch the KV cache write function to capture written values."""
    from vllm.v1.attention.backends import triton_attn

    original_update = triton_attn.TritonAttentionBackend.do_kv_cache_update

    def patched_update(self, key, value, kv_cache, attn_metadata, layer):
        # Call original
        result = original_update(self, key, value, kv_cache, attn_metadata, layer)

        # Capture written KV
        layer_idx = getattr(layer, 'layer_idx', '?')
        run_id = getattr(attn_metadata, '_debug_run_id', 'unknown')

        if run_id == 'run1':
            captured_kv['run1_write'][layer_idx] = {
                'key': key.clone().detach().cpu(),
                'value': value.clone().detach().cpu(),
            }
            logger.info(f"[CAPTURE] Run 1 wrote layer {layer_idx}: "
                       f"key={key.shape}, mean={key.float().mean():.6f}")
        elif run_id == 'run2':
            captured_kv['run2_write'][layer_idx] = {
                'key': key.clone().detach().cpu(),
                'value': value.clone().detach().cpu(),
            }
            logger.info(f"[CAPTURE] Run 2 wrote layer {layer_idx}: "
                       f"key={key.shape}, mean={key.float().mean():.6f}")

        return result

    triton_attn.TritonAttentionBackend.do_kv_cache_update = patched_update


def patch_kv_cache_read():
    """Patch the attention forward to capture KV values read from cache."""
    from vllm.v1.attention.backends import triton_attn

    original_forward = triton_attn.TritonAttentionBackend.forward

    def patched_forward(self, layer, query, key, value, kv_cache, attn_metadata,
                       output=None, output_scale=None, output_block_scale=None):
        # Capture KV cache state BEFORE attention
        layer_idx = getattr(layer, 'layer_idx', '?')
        run_id = getattr(attn_metadata, '_debug_run_id', 'unknown')

        if run_id == 'run2' and kv_cache is not None:
            key_cache, value_cache = kv_cache.unbind(1)
            # Capture the cached portion (first 16 tokens = 1 block of 16)
            # Assuming block_size=16
            captured_kv['run2_read'][layer_idx] = {
                'key_cache': key_cache.clone().detach().cpu(),
                'value_cache': value_cache.clone().detach().cpu(),
            }
            logger.info(f"[CAPTURE] Run 2 reading layer {layer_idx} cache: "
                       f"shape={key_cache.shape}")

        # Call original
        return original_forward(self, layer, query, key, value, kv_cache,
                              attn_metadata, output, output_scale, output_block_scale)

    triton_attn.TritonAttentionBackend.forward = patched_forward


async def test_cache_values():
    """Test if cached KV values are identical between runs."""

    # Apply patches
    logger.info("Applying KV cache capture patches...")
    patch_kv_cache_write()
    patch_kv_cache_read()

    # Import after patching
    from vllm.v1.engine.async_llm import AsyncLLM

    # Initialize engine
    logger.info("\n" + "="*80)
    logger.info("Initializing vLLM engine with BF16...")
    logger.info("="*80)

    engine = AsyncLLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=512,
        enable_prefix_caching=True,
    )

    prompt = "What is the capital of France? Answer:"

    # Run 1: Cache miss
    logger.info("\n" + "="*80)
    logger.info("RUN 1 - Cache Miss (writes KV to cache)")
    logger.info("="*80)

    # Hacky way to mark this as run1 - inject into request metadata
    # (This is a test hack, not production code)
    import vllm.v1.engine.core
    original_step = vllm.v1.engine.core.EngineCore.step

    def patched_step_run1(self, scheduler_output):
        # Mark all metadata as run1
        if hasattr(scheduler_output, 'scheduled_new_reqs'):
            for req in scheduler_output.scheduled_new_reqs:
                if hasattr(req, 'attn_metadata') and req.attn_metadata:
                    req.attn_metadata._debug_run_id = 'run1'
        return original_step(self, scheduler_output)

    vllm.v1.engine.core.EngineCore.step = patched_step_run1

    result1 = await engine.generate(prompt, sampling_params={"max_tokens": 5, "temperature": 0})
    logger.info(f"Run 1 output: {result1.outputs[0].text}")

    # Wait a bit for cache to stabilize
    await asyncio.sleep(0.5)

    # Run 2: Cache hit
    logger.info("\n" + "="*80)
    logger.info("RUN 2 - Cache Hit (reads KV from cache)")
    logger.info("="*80)

    def patched_step_run2(self, scheduler_output):
        # Mark all metadata as run2
        if hasattr(scheduler_output, 'scheduled_new_reqs'):
            for req in scheduler_output.scheduled_new_reqs:
                if hasattr(req, 'attn_metadata') and req.attn_metadata:
                    req.attn_metadata._debug_run_id = 'run2'
        return original_step(self, scheduler_output)

    vllm.v1.engine.core.EngineCore.step = patched_step_run2

    result2 = await engine.generate(prompt, sampling_params={"max_tokens": 5, "temperature": 0})
    logger.info(f"Run 2 output: {result2.outputs[0].text}")

    # Restore original
    vllm.v1.engine.core.EngineCore.step = original_step

    # Compare outputs
    logger.info("\n" + "="*80)
    logger.info("COMPARISON")
    logger.info("="*80)

    if result1.outputs[0].text != result2.outputs[0].text:
        logger.error("✗ OUTPUTS DIFFER (bug confirmed)")
        logger.error(f"  Run 1: '{result1.outputs[0].text}'")
        logger.error(f"  Run 2: '{result2.outputs[0].text}'")
    else:
        logger.info("✓ Outputs match")

    # Compare cached KV values
    logger.info("\n" + "="*80)
    logger.info("KV CACHE VALUE COMPARISON")
    logger.info("="*80)

    if not captured_kv['run1_write']:
        logger.error("✗ Failed to capture Run 1 KV writes")
        return False

    if not captured_kv['run2_read']:
        logger.error("✗ Failed to capture Run 2 KV reads")
        return False

    # Compare layer 0 as example
    layer_idx = 0
    if layer_idx not in captured_kv['run1_write']:
        logger.error(f"✗ Layer {layer_idx} not found in Run 1 writes")
        return False

    run1_key = captured_kv['run1_write'][layer_idx]['key']
    run1_value = captured_kv['run1_write'][layer_idx]['value']

    logger.info(f"\nRun 1 wrote (layer {layer_idx}):")
    logger.info(f"  Key shape: {run1_key.shape}")
    logger.info(f"  Key mean: {run1_key.float().mean():.8f}")
    logger.info(f"  Key std: {run1_key.float().std():.8f}")
    logger.info(f"  Key dtype: {run1_key.dtype}")

    if layer_idx in captured_kv['run2_read']:
        # Extract the cached portion from run 2's cache
        run2_key_cache = captured_kv['run2_read'][layer_idx]['key_cache']
        run2_value_cache = captured_kv['run2_read'][layer_idx]['value_cache']

        logger.info(f"\nRun 2 read from cache (layer {layer_idx}):")
        logger.info(f"  Cache shape: {run2_key_cache.shape}")

        # TODO: Extract the specific cached tokens from the block structure
        # This requires understanding the KV cache layout (blocks, etc.)
        logger.warning("  Cache extraction not yet implemented - need to decode block structure")

    logger.info("\n" + "="*80)
    logger.info("VERDICT")
    logger.info("="*80)
    logger.info("Test inconclusive - need to properly extract cached values from block structure")
    logger.info("Next step: Add logging directly in unified_attention kernel to see KV values")

    return True


if __name__ == "__main__":
    try:
        asyncio.run(test_cache_values())
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
