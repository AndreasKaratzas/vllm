#!/usr/bin/env python3
"""Test FP8 rounding behavior between PyTorch .to() and CUDA kernel."""

import torch
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)

torch.set_default_device("cuda")

# Create a test case where values fall between FP8 representable values
# FP8 e4m3 has discrete representable values, so some float values will need rounding

# Test specific values that we saw differing
test_values = torch.tensor([
    -0.015442,  # This gave native=-22, cuda=-24 (with scale=6.713868e-04)
    0.030884,   # This gave native=44, cuda=48
    -0.067383,  # This gave native=-88, cuda=-96 (with scale=7.324219e-04)
], dtype=torch.bfloat16).unsqueeze(0)  # Shape: [1, 3]

# Pad to make it divisible by 128
padding_size = 128 - (test_values.shape[1] % 128)
test_values = torch.cat([test_values, torch.zeros(1, padding_size, dtype=torch.bfloat16)], dim=1)

print("Testing FP8 quantization rounding:")
print(f"Input shape: {test_values.shape}")
print(f"First 3 values: {test_values[0, :3]}")

# Use the CUDA kernel
q_cuda, s_cuda = per_token_group_quant_fp8(test_values, group_size=128)

print(f"\nCUDA kernel:")
print(f"  Scale: {s_cuda[0, 0]:.6e}")
print(f"  Quantized values: {q_cuda[0, :3].float()}")

# Now test what PyTorch .to() does
finfo = torch.finfo(torch.float8_e4m3fn)
fp8_min, fp8_max = finfo.min, finfo.max

# Manually quantize like the reference
x_flat = test_values.reshape(-1, 128)
amax = x_flat.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-10).to(torch.float32)
scale = amax / fp8_max

print(f"\nManual PyTorch (reference style):")
print(f"  Scale: {scale[0, 0]:.6e}")

# Method 1: Direct .to() like the reference
q_direct = (x_flat / scale).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
print(f"  Quantized (direct .to()): {q_direct[0, :3].float()}")

# Method 2: Convert to float first, then round, then convert
q_float = (x_flat / scale).clamp(min=fp8_min, max=fp8_max)
print(f"  Before .to() (as float32): {q_float[0, :3]}")

# Check PyTorch's conversion behavior
test_float_values = torch.tensor([-23.0, -23.5, -23.9999], dtype=torch.float32, device='cuda')
test_fp8 = test_float_values.to(torch.float8_e4m3fn)
print(f"\nPyTorch rounding test:")
for i, (f, fp8) in enumerate(zip(test_float_values, test_fp8)):
    print(f"  {f.item():.4f} -> {fp8.float().item():.1f}")

# Test CUDA kernel with same input
print(f"\nComparing with CUDA kernel:")
print(f"  Scale difference: {abs(s_cuda[0, 0] - scale[0, 0]):.6e}")
print(f"  Quant diff: {(q_cuda[0, :3].float() - q_direct[0, :3].float()).abs()}")

# Test intermediate values
intermediate = test_values[0, :3] / scale[0, 0]
print(f"\nIntermediate values (before FP8 conversion):")
for i, val in enumerate(intermediate):
    native_fp8 = q_direct[0, i].float().item()
    cuda_fp8 = q_cuda[0, i].float().item()
    print(f"  [{i}]: {val.item():.6f} -> native={native_fp8:.1f}, cuda={cuda_fp8:.1f}, diff={abs(native_fp8-cuda_fp8):.1f}")

# Check what the actual float32 values are that are being converted
print(f"\nActual float32 values being quantized (first 3):")
vals_to_quant = (test_values[0, :3].to(torch.float32) / s_cuda[0, 0])
for i, v in enumerate(vals_to_quant):
    print(f"  [{i}]: {v.item():.10f}")
    # Find nearest FP8 values
    fp8_test = torch.tensor([v.item() - 4, v.item() - 2, v.item(), v.item() + 2, v.item() + 4], dtype=torch.float32, device='cuda')
    fp8_rounded = fp8_test.to(torch.float8_e4m3fn).float()
    print(f"       Nearby FP8 values: {fp8_rounded}")
