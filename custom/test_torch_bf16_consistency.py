#!/usr/bin/env python3
"""
Test if PyTorch operations on gfx950 are consistent with bfloat16
This tests if the underlying issue is in PyTorch/ROCm itself
"""

import torch
from vllm.platforms import current_platform

def test_matmul_consistency(dtype, num_iterations=5):
    """Test if repeated matmuls give consistent results"""
    print(f"\n{'='*80}")
    print(f"Testing {dtype} matmul consistency ({num_iterations} iterations)")
    print('='*80)

    # Create test tensors (similar sizes to attention)
    batch_size = 1
    seq_len = 128
    hidden_size = 256

    # Input (this stays fixed)
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device='cuda')

    # Weight matrix (this stays fixed)
    weight = torch.randn(hidden_size, hidden_size, dtype=dtype, device='cuda')

    results = []
    for i in range(num_iterations):
        # Do the same computation each time
        with torch.no_grad():
            y = torch.matmul(x, weight)

        results.append(y.clone())

        if i == 0:
            print(f"Iteration {i+1}: baseline")
        else:
            diff = (results[i] - results[0]).abs().max().item()
            matches = torch.allclose(results[i], results[0], rtol=0, atol=0)
            print(f"Iteration {i+1}: max diff from baseline = {diff:.2e}, exact={matches}")

    # Check if all results match
    all_match = all(torch.allclose(r, results[0], rtol=0, atol=0) for r in results)

    if all_match:
        print(f"✓ {dtype} matmul is DETERMINISTIC")
        return True
    else:
        print(f"✗ {dtype} matmul is NON-DETERMINISTIC")
        max_diff = max((r - results[0]).abs().max().item() for r in results[1:])
        print(f"  Max difference: {max_diff:.2e}")
        return False

def test_attention_consistency(dtype, num_iterations=5):
    """Test if attention computation is consistent"""
    print(f"\n{'='*80}")
    print(f"Testing {dtype} attention consistency ({num_iterations} iterations)")
    print('='*80)

    # Attention parameters
    batch_size = 1
    seq_len = 128
    num_heads = 4
    head_dim = 64

    # Create Q, K, V (fixed)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device='cuda')

    scale = 1.0 / (head_dim ** 0.5)

    results = []
    for i in range(num_iterations):
        with torch.no_grad():
            # Compute attention: softmax(Q @ K^T / sqrt(d)) @ V
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = torch.nn.functional.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, v)

        results.append(output.clone())

        if i == 0:
            print(f"Iteration {i+1}: baseline")
        else:
            diff = (results[i] - results[0]).abs().max().item()
            matches = torch.allclose(results[i], results[0], rtol=0, atol=0)
            print(f"Iteration {i+1}: max diff from baseline = {diff:.2e}, exact={matches}")

    all_match = all(torch.allclose(r, results[0], rtol=0, atol=0) for r in results)

    if all_match:
        print(f"✓ {dtype} attention is DETERMINISTIC")
        return True
    else:
        print(f"✗ {dtype} attention is NON-DETERMINISTIC")
        max_diff = max((r - results[0]).abs().max().item() for r in results[1:])
        print(f"  Max difference: {max_diff:.2e}")
        return False

if __name__ == "__main__":
    print(f"\nGPU: {torch.cuda.get_device_properties('cuda').gcnArchName}")
    print(f"Platform: ROCm={current_platform.is_rocm()}")

    # Test FP16
    print("\n" + "#"*80)
    print("# FLOAT16 TESTS")
    print("#"*80)
    fp16_matmul = test_matmul_consistency(torch.float16)
    fp16_attn = test_attention_consistency(torch.float16)

    # Test BF16
    print("\n" + "#"*80)
    print("# BFLOAT16 TESTS")
    print("#"*80)
    bf16_matmul = test_matmul_consistency(torch.bfloat16)
    bf16_attn = test_attention_consistency(torch.bfloat16)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"FP16  matmul:    {'✓ PASS' if fp16_matmul else '✗ FAIL'}")
    print(f"FP16  attention: {'✓ PASS' if fp16_attn else '✗ FAIL'}")
    print(f"BF16  matmul:    {'✓ PASS' if bf16_matmul else '✗ FAIL'}")
    print(f"BF16  attention: {'✓ PASS' if bf16_attn else '✗ FAIL'}")

    if fp16_matmul and fp16_attn and not (bf16_matmul and bf16_attn):
        print("\n⚠️  PyTorch operations are NON-DETERMINISTIC with BF16 on this GPU!")
        print("   This is likely a ROCm/gfx950 issue, not a vLLM bug.")
        print("   Workaround: Use FP16 instead of BF16 on MI355X.")
    elif bf16_matmul and bf16_attn:
        print("\n✓ PyTorch BF16 operations are deterministic")
        print("  The vLLM bug must be in a different layer (KV cache, model, etc.)")
