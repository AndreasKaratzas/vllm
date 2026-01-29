#!/usr/bin/env python3
"""
Test if KV cache write/read roundtrip preserves values for bf16 vs fp16
"""

import torch
from vllm.platforms import current_platform

def test_cache_roundtrip(dtype):
    """Test if writing and reading from cache preserves values"""
    print(f"\n{'='*80}")
    print(f"Testing {dtype} cache roundtrip")
    print('='*80)

    # Create fake K/V tensors
    num_tokens = 10
    num_heads = 4
    head_size = 128
    block_size = 16
    num_blocks = 2

    # Input tensors
    key = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device='cuda')
    value = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device='cuda')

    # Cache tensors (using standard layout: [num_blocks, block_size, num_heads, head_size])
    key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=dtype, device='cuda')
    value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=dtype, device='cuda')

    # Slot mapping: map each token to a cache slot
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device='cuda')

    # Write to cache using Triton kernel
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash

    triton_reshape_and_cache_flash(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        kv_cache_dtype="auto",  # No FP8, just preserve dtype
        k_scale=torch.tensor(1.0, dtype=torch.float32, device='cuda'),
        v_scale=torch.tensor(1.0, dtype=torch.float32, device='cuda'),
    )

    # Read back from cache
    key_read = torch.zeros_like(key)
    value_read = torch.zeros_like(value)

    for i in range(num_tokens):
        slot_idx = slot_mapping[i].item()
        block_idx = slot_idx // block_size
        block_offset = slot_idx % block_size

        key_read[i] = key_cache[block_idx, block_offset]
        value_read[i] = value_cache[block_idx, block_offset]

    # Compare
    key_matches = torch.allclose(key, key_read, rtol=0, atol=0)
    value_matches = torch.allclose(value, value_read, rtol=0, atol=0)

    key_max_diff = (key - key_read).abs().max().item()
    value_max_diff = (value - value_read).abs().max().item()

    print(f"Key matches:   {key_matches} (max diff: {key_max_diff})")
    print(f"Value matches: {value_matches} (max diff: {value_max_diff})")

    if key_matches and value_matches:
        print(f"✓ {dtype} cache roundtrip is EXACT")
        return True
    else:
        print(f"✗ {dtype} cache roundtrip has ERRORS")
        if key_max_diff > 0:
            print(f"  Key diff: {key_max_diff}")
        if value_max_diff > 0:
            print(f"  Value diff: {value_max_diff}")
        return False

if __name__ == "__main__":
    print(f"\nGPU: {torch.cuda.get_device_properties('cuda').gcnArchName}")
    print(f"Platform: ROCm={current_platform.is_rocm()}")

    # Test both dtypes
    fp16_ok = test_cache_roundtrip(torch.float16)
    bf16_ok = test_cache_roundtrip(torch.bfloat16)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"FP16:  {'✓ PASS' if fp16_ok else '✗ FAIL'}")
    print(f"BF16:  {'✓ PASS' if bf16_ok else '✗ FAIL'}")

    if fp16_ok and not bf16_ok:
        print("\n⚠️  Cache roundtrip works for FP16 but NOT for BF16!")
        print("   This could be the root cause of the prefix caching bug.")
    elif fp16_ok and bf16_ok:
        print("\n✓ Cache roundtrip works for both dtypes")
        print("  The bug must be elsewhere (model computation, attention, etc.)")
