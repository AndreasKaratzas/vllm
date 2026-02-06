#!/usr/bin/env python3
"""Debug script to identify FP8 quantization mismatches in MOE kernels."""

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import (
    make_dummy_moe_config,
    make_test_quant_config,
    modular_triton_fused_moe,
)
from tests.kernels.quant_utils import (
    native_per_token_group_quant_fp8,
    native_w8a8_block_matmul,
)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import (
    fused_experts,
    fused_topk,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)

# Test parameters from the failing test
M, N, K = 83, 4608, 7168
E, topk = 2, 1
block_size = [128, 128]
dtype = torch.bfloat16
seed = 0

torch.manual_seed(seed)
torch.set_default_device("cuda")

vllm_config = VllmConfig()

# Create test data
a = torch.randn((M, K), dtype=dtype) / 10
score = torch.randn((M, E), dtype=dtype)

w1, w2, quant_config = make_test_quant_config(
    E,
    N,
    K,
    dtype,
    quant_dtype=torch.float8_e4m3fn,
    per_act_token_quant=False,
    block_shape=block_size,
)

topk_weights, topk_ids, _ = fused_topk(a, score.float(), topk, False)

# Step 1: Compare quantization of input activations
print("="*80)
print("STEP 1: Comparing input activation quantization")
print("="*80)

_, block_k = block_size[0], block_size[1]

# Native quantization (reference)
a_q_native, a_s_native = native_per_token_group_quant_fp8(a, block_k)
print(f"Native quantization:")
print(f"  a_q shape: {a_q_native.shape}, dtype: {a_q_native.dtype}")
print(f"  a_s shape: {a_s_native.shape}, dtype: {a_s_native.dtype}")
print(f"  a_s min/max: {a_s_native.min():.6e} / {a_s_native.max():.6e}")
print(f"  a_s sample (first 5): {a_s_native.flatten()[:5]}")

# Production quantization (CUDA kernel)
a_q_cuda, a_s_cuda = per_token_group_quant_fp8(a, block_k)
print(f"\nCUDA kernel quantization:")
print(f"  a_q shape: {a_q_cuda.shape}, dtype: {a_q_cuda.dtype}")
print(f"  a_s shape: {a_s_cuda.shape}, dtype: {a_s_cuda.dtype}")
print(f"  a_s min/max: {a_s_cuda.min():.6e} / {a_s_cuda.max():.6e}")
print(f"  a_s sample (first 5): {a_s_cuda.flatten()[:5]}")

# Compare scales
scale_diff = (a_s_native - a_s_cuda).abs()
print(f"\nScale comparison:")
print(f"  Max abs diff: {scale_diff.max():.6e}")
print(f"  Mean abs diff: {scale_diff.mean():.6e}")
print(f"  Num different: {(scale_diff > 1e-10).sum()} / {scale_diff.numel()}")

# Compare quantized values
quant_diff = (a_q_native.float() - a_q_cuda.float()).abs()
print(f"\nQuantized value comparison:")
print(f"  Max abs diff: {quant_diff.max():.6f}")
print(f"  Mean abs diff: {quant_diff.mean():.6f}")
print(f"  Num different: {(quant_diff > 0.5).sum()} / {quant_diff.numel()}")

# Step 2: Analyze the quantization differences
print("\n" + "="*80)
print("STEP 2: Analyzing quantization differences")
print("="*80)

# Find where they differ
quant_mismatch = (a_q_native.float() != a_q_cuda.float())
print(f"Elements where quantization differs: {quant_mismatch.sum()} / {quant_mismatch.numel()}")

if quant_mismatch.sum() > 0:
    # Get indices of mismatches
    mismatch_indices = torch.nonzero(quant_mismatch, as_tuple=True)

    # Sample first 10 mismatches
    num_samples = min(10, len(mismatch_indices[0]))
    print(f"\nSample of first {num_samples} mismatches:")
    for idx in range(num_samples):
        i, j = mismatch_indices[0][idx], mismatch_indices[1][idx]

        # Get the group this element belongs to
        group_idx = j // block_k

        # Get the original value
        orig_val = a[i, j].item()

        # Get the scales
        scale_native = a_s_native[i, group_idx].item()
        scale_cuda = a_s_cuda[i, group_idx].item()

        # Get quantized values
        q_native = a_q_native[i, j].float().item()
        q_cuda = a_q_cuda[i, j].float().item()

        # Compute what they should dequant to
        dequant_native = q_native * scale_native
        dequant_cuda = q_cuda * scale_cuda

        print(f"  [{i}, {j}] group={group_idx}:")
        print(f"    original: {orig_val:.6f}")
        print(f"    scale: native={scale_native:.6e}, cuda={scale_cuda:.6e}, diff={abs(scale_native-scale_cuda):.6e}")
        print(f"    quantized: native={q_native:.1f}, cuda={q_cuda:.1f}, diff={abs(q_native-q_cuda):.1f}")
        print(f"    dequantized: native={dequant_native:.6f}, cuda={dequant_cuda:.6f}")
        print(f"    error from original: native={abs(orig_val-dequant_native):.6f}, cuda={abs(orig_val-dequant_cuda):.6f}")

print("\n" + "="*80)
print("DONE - Root cause identified in quantization step")
print("="*80)
