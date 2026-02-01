#!/usr/bin/env python3
"""
Debug script to identify the exact point of numerical divergence
between cache-miss and cache-hit attention paths on gfx950.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess
import sys

MODEL_NAME = "Qwen/Qwen3-0.6B"

def check_gpu_arch():
    """Verify we're running on gfx950."""
    arch = torch.cuda.get_device_properties("cuda").gcnArchName
    print(f"GPU Architecture: {arch}")
    if "gfx950" not in arch:
        print(f"⚠️  WARNING: Not running on gfx950! This bug is specific to MI355X.")
    return arch

def extract_attention_activations(model, input_ids, use_cache=True):
    """
    Run forward pass and extract attention intermediate values.
    Returns dict with logits, attention outputs, and KV cache.
    """
    with torch.no_grad():
        outputs = model(
            input_ids,
            use_cache=use_cache,
            output_attentions=True,
            return_dict=True,
        )

    return {
        'logits': outputs.logits,
        'attentions': outputs.attentions,  # Tuple of attention weights per layer
        'past_key_values': outputs.past_key_values if use_cache else None,
    }

def compare_attention_outputs():
    """
    Compare attention outputs between cache and no-cache paths.
    """
    print("\n" + "="*80)
    print("COMPARING HUGGINGFACE ATTENTION OUTPUTS")
    print("="*80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How many countries are in the EU?"},
    ]

    token_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False
    )
    input_ids = torch.tensor([token_ids], device=model.device)

    print(f"\nInput shape: {input_ids.shape}")
    print(f"Input tokens: {len(token_ids)}")

    # Run with cache
    print("\n--- WITH CACHE ---")
    outputs_cache = extract_attention_activations(model, input_ids, use_cache=True)

    # Run without cache
    print("--- WITHOUT CACHE ---")
    outputs_nocache = extract_attention_activations(model, input_ids, use_cache=False)

    # Compare logits
    print("\n" + "="*80)
    print("LOGITS COMPARISON")
    print("="*80)

    logits_cache = outputs_cache['logits'][0, -1, :]  # Last token logits
    logits_nocache = outputs_nocache['logits'][0, -1, :]

    diff = (logits_cache - logits_nocache).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max logit difference:  {max_diff:.6e}")
    print(f"Mean logit difference: {mean_diff:.6e}")

    # Compare top-5 predictions
    top5_cache = torch.topk(logits_cache, k=5)
    top5_nocache = torch.topk(logits_nocache, k=5)

    print(f"\nTop-5 tokens (WITH cache):")
    for i, (idx, val) in enumerate(zip(top5_cache.indices, top5_cache.values)):
        token = tokenizer.decode([idx.item()])
        print(f"  {i+1}. {repr(token):20s} logit={val.item():+.6f}")

    print(f"\nTop-5 tokens (WITHOUT cache):")
    for i, (idx, val) in enumerate(zip(top5_nocache.indices, top5_nocache.values)):
        token = tokenizer.decode([idx.item()])
        print(f"  {i+1}. {repr(token):20s} logit={val.item():+.6f}")

    # Compare attention outputs layer by layer
    print("\n" + "="*80)
    print("ATTENTION OUTPUTS (LAYER-BY-LAYER)")
    print("="*80)

    num_layers = len(outputs_cache['attentions'])
    print(f"Number of layers: {num_layers}")

    for layer_idx in range(num_layers):
        attn_cache = outputs_cache['attentions'][layer_idx]
        attn_nocache = outputs_nocache['attentions'][layer_idx]

        diff = (attn_cache - attn_nocache).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"\nLayer {layer_idx:2d}:  max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return max_diff < 1e-4  # Tolerance for bfloat16

def diagnose_triton_precision():
    """
    Check Triton's default precision settings.
    """
    print("\n" + "="*80)
    print("TRITON PRECISION DIAGNOSTICS")
    print("="*80)

    try:
        from vllm.triton_utils import triton
        print(f"Triton version: {triton.__version__}")

        # Check if input_precision affects results
        print("\nChecking available Triton dot operation modes:")
        print("  - 'ieee': Standard IEEE floating-point")
        print("  - 'tf32': TensorFloat-32 (if available)")
        print("  - None: Default (platform-specific)")

    except Exception as e:
        print(f"Error checking Triton: {e}")

def main():
    print("="*80)
    print("GFX950 PREFIX CACHING BUG DIAGNOSTIC TOOL")
    print("="*80)

    # Check environment
    arch = check_gpu_arch()

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"Device 0: {torch.cuda.get_device_name(0)}")

    # Run diagnostics
    diagnose_triton_precision()

    # Compare HF attention
    print("\n" + "="*80)
    print("Running HuggingFace attention comparison...")
    print("="*80)

    try:
        matches = compare_attention_outputs()
        if matches:
            print("\n✓ HuggingFace attention outputs match (within tolerance)")
        else:
            print("\n✗ HuggingFace attention outputs DIFFER significantly")
    except Exception as e:
        print(f"\n✗ Error during comparison: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Run the full test suite:
   pytest -s -v tests/entrypoints/openai/test_prefix_cache_debug.py

2. Check vLLM kernel precision settings in:
   - vllm/v1/attention/ops/prefix_prefill.py (line 664)
   - vllm/v1/attention/ops/chunked_prefill_paged_decode.py (kernel_paged_attention_2d)

3. Try forcing consistent precision:
   Add input_precision parameter to kernel_paged_attention_2d

4. Enable debug logging:
   export VLLM_LOGGING_LEVEL=DEBUG
    """)

if __name__ == "__main__":
    main()
