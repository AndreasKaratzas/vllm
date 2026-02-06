#!/usr/bin/env python3
"""Analyze if tiny scale differences cause quantization mismatches at FP8 boundaries."""

import torch
from tests.kernels.quant_utils import native_per_token_group_quant_fp8
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

torch.set_default_device("cuda")
torch.manual_seed(0)

M, K = 83, 7168
block_k = 128
dtype = torch.bfloat16

a = torch.randn((M, K), dtype=dtype) / 10

# Get quantizations
a_q_native, a_s_native = native_per_token_group_quant_fp8(a, block_k)
a_q_cuda, a_s_cuda = per_token_group_quant_fp8(a, block_k)

# Find mismatches
mismatch = (a_q_native.float() != a_q_cuda.float())
mismatch_indices = torch.nonzero(mismatch, as_tuple=True)

print("="*80)
print("Analyzing quantization boundaries")
print("="*80)

if len(mismatch_indices[0]) > 0:
    print(f"\nFound {len(mismatch_indices[0])} mismatches")

    # Analyze first 5 mismatches in detail
    for idx in range(min(5, len(mismatch_indices[0]))):
        i, j = mismatch_indices[0][idx].item(), mismatch_indices[1][idx].item()
        group_idx = j // block_k

        orig_val = a[i, j]
        scale_native = a_s_native[i, group_idx]
        scale_cuda = a_s_cuda[i, group_idx]

        q_native = a_q_native[i, j].float()
        q_cuda = a_q_cuda[i, j].float()

        # Compute what the float division gives
        div_native_bf16 = (orig_val / scale_native).item()
        div_native_f32 = (orig_val.to(torch.float32) / scale_native).item()
        div_cuda_f32 = (orig_val.to(torch.float32) / scale_cuda).item()

        print(f"\nMismatch {idx}: [{i}, {j}], group={group_idx}")
        print(f"  Original value: {orig_val.item():.8f}")
        print(f"  Scales: native={scale_native.item():.12e}, cuda={scale_cuda.item():.12e}")
        print(f"  Scale diff: {abs(scale_native.item() - scale_cuda.item()):.12e}")
        print(f"  Division (bf16): {div_native_bf16:.6f}")
        print(f"  Division (f32 with native scale): {div_native_f32:.6f}")
        print(f"  Division (f32 with cuda scale): {div_cuda_f32:.6f}")
        print(f"  Quantized: native={q_native.item():.1f}, cuda={q_cuda.item():.1f}")

        # Test if this is a boundary case
        # Check what FP8 values are near the division result
        test_vals = torch.tensor([
            div_cuda_f32 - 8, div_cuda_f32 - 4, div_cuda_f32 - 2,
            div_cuda_f32,
            div_cuda_f32 + 2, div_cuda_f32 + 4, div_cuda_f32 + 8
        ], dtype=torch.float32, device='cuda')

        fp8_vals = test_vals.to(torch.float8_e4m3fn).float()
        print(f"  Nearby FP8 values: {fp8_vals.tolist()}")
        print(f"  -> {div_cuda_f32:.2f} is between FP8 values {fp8_vals[3].item():.1f} and next")

# Now let's check if the scale computation itself is the issue
print("\n" + "="*80)
print("Checking scale computation differences")
print("="*80)

# Manually compute scales both ways
finfo = torch.finfo(torch.float8_e4m3fn)
fp8_max = finfo.max

x_ = a.reshape(a.numel() // block_k, block_k)

# Native way
amax_native = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-10).to(torch.float32)
scales_native_manual = amax_native / fp8_max

# Compare with actual scales from native
scales_native_manual_flat = scales_native_manual.flatten()
scales_native_flat = a_s_native.flatten()

scale_diff = (scales_native_manual_flat - scales_native_flat).abs()
print(f"\nNative manual vs actual:")
print(f"  Max diff: {scale_diff.max():.12e}")
print(f"  Mean diff: {scale_diff.mean():.12e}")
print(f"  Non-zero diffs: {(scale_diff > 0).sum()} / {len(scale_diff)}")

# Compare native vs CUDA scales
scale_diff_nc = (a_s_native - a_s_cuda).abs()
print(f"\nNative vs CUDA scales:")
print(f"  Max diff: {scale_diff_nc.max():.12e}")
print(f"  Mean diff: {scale_diff_nc.mean():.12e}")
print(f"  Non-zero diffs: {(scale_diff_nc > 1e-15).sum()} / {scale_diff_nc.numel()}")

# Check if UE8M0 is being used
from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used
print(f"\nis_deep_gemm_e8m0_used(): {is_deep_gemm_e8m0_used()}")
