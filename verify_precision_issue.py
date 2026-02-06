#!/usr/bin/env python3
"""Verify that intermediate precision causes the quantization mismatch."""

import torch
from tests.kernels.quant_utils import native_per_token_group_quant_fp8
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

torch.set_default_device("cuda")
torch.manual_seed(0)

# Use exact test parameters
M, K = 83, 7168
block_k = 128
dtype = torch.bfloat16

# Create test data
a = torch.randn((M, K), dtype=dtype) / 10

print("="*80)
print("Testing intermediate precision hypothesis")
print("="*80)

# Native quantization (reference) - division in bfloat16
a_q_native, a_s_native = native_per_token_group_quant_fp8(a, block_k)

# CUDA kernel - division in float32
a_q_cuda, a_s_cuda = per_token_group_quant_fp8(a, block_k)

# Fixed native version - force float32 intermediate
finfo = torch.finfo(torch.float8_e4m3fn)
fp8_min, fp8_max = finfo.min, finfo.max

x_ = a.reshape(a.numel() // block_k, block_k)
amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-10).to(torch.float32)
x_s = amax / fp8_max

# Key fix: convert to float32 BEFORE division
x_float32 = x_.to(torch.float32)
x_q_fixed = (x_float32 / x_s).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
x_q_fixed = x_q_fixed.reshape(a.shape)
x_s_fixed = x_s.reshape(a.shape[:-1] + (a.shape[-1] // block_k,))

print(f"\n1. Original native (bfloat16 intermediate):")
print(f"   Sample quantized values: {a_q_native[0, :5].float()}")

print(f"\n2. CUDA kernel (float32 intermediate):")
print(f"   Sample quantized values: {a_q_cuda[0, :5].float()}")

print(f"\n3. Fixed native (float32 intermediate):")
print(f"   Sample quantized values: {x_q_fixed[0, :5].float()}")

# Compare
native_vs_cuda = (a_q_native.float() != a_q_cuda.float()).sum()
fixed_vs_cuda = (x_q_fixed.float() != a_q_cuda.float()).sum()

print(f"\n" + "="*80)
print(f"Results:")
print(f"="*80)
print(f"Native vs CUDA mismatches: {native_vs_cuda} / {a.numel()}")
print(f"Fixed vs CUDA mismatches: {fixed_vs_cuda} / {a.numel()}")

if fixed_vs_cuda == 0:
    print(f"\n✓ SUCCESS: Fixed version matches CUDA kernel exactly!")
    print(f"✓ Root cause confirmed: intermediate precision in division")
else:
    print(f"\n✗ Still have mismatches, need further investigation")

# Show example of precision loss
print(f"\n" + "="*80)
print(f"Example of precision loss in bfloat16 division:")
print(f"="*80)
test_val = a[0, 100]
test_scale = a_s_native[0, 100 // block_k]

# Bfloat16 division
result_bf16 = (test_val / test_scale).item()

# Float32 division
result_f32 = (test_val.to(torch.float32) / test_scale).item()

print(f"Value: {test_val.item():.8f}")
print(f"Scale: {test_scale.item():.8e}")
print(f"Bfloat16 division result: {result_bf16:.8f}")
print(f"Float32 division result:  {result_f32:.8f}")
print(f"Difference: {abs(result_bf16 - result_f32):.8f}")
