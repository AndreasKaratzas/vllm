import torch
import os

os.environ['ROCBLAS_LAYER'] = '5'

# Exact Qwen3-0.6B MLP dimensions
hidden_size = 1024
intermediate_size = 2816  # Actually 2816 * 2 = 5632 for gate_up

# Simulate vLLM's MergedColumnParallelLinear (gate_up_proj)
# This does: x @ [gate_weight, up_weight].T where combined weight is [5632, 1024]
batch_31 = torch.randn(31, hidden_size, dtype=torch.bfloat16, device='cuda')
batch_15 = torch.randn(15, hidden_size, dtype=torch.bfloat16, device='cuda')

# Make first row identical
batch_15[0] = batch_31[16].clone()

# Weight matrix (gate_up combined)
weight = torch.randn(intermediate_size * 2, hidden_size, dtype=torch.bfloat16, device='cuda')

print(f"Input shapes: batch_31={batch_31.shape}, batch_15={batch_15.shape}")
print(f"Weight shape: {weight.shape}")
print(f"Input row identical: {torch.equal(batch_31[16], batch_15[0])}")

# Run matmul - this is what gate_up_proj does
torch.cuda.synchronize()
out_31 = torch.nn.functional.linear(batch_31, weight)  # [31, 5632]
torch.cuda.synchronize()

out_15 = torch.nn.functional.linear(batch_15, weight)  # [15, 5632]
torch.cuda.synchronize()

# Compare position 16 from batch_31 with position 0 from batch_15
row_match = torch.equal(out_31[16], out_15[0])
print(f"\nOutput row identical: {row_match}")

if not row_match:
    diff = (out_31[16] - out_15[0]).abs()
    num_diff = (diff > 0).sum().item()
    print(f"  Num differences: {num_diff} / {intermediate_size * 2}")
    print(f"  Max diff: {diff.max().item()}")
    
    # Find first difference
    nonzero = torch.nonzero(diff > 0)
    if len(nonzero) > 0:
        idx = nonzero[0].item()
        print(f"  First diff at index {idx}: {out_31[16, idx].item()} vs {out_15[0, idx].item()}")
else:
    print("  All 5632 elements match!")

# Also test with torch.matmul directly
print("\n--- Testing torch.matmul directly ---")
out_31_mm = torch.matmul(batch_31, weight.T)
out_15_mm = torch.matmul(batch_15, weight.T)

row_match_mm = torch.equal(out_31_mm[16], out_15_mm[0])
print(f"torch.matmul output identical: {row_match_mm}")

if not row_match_mm:
    diff_mm = (out_31_mm[16] - out_15_mm[0]).abs()
    print(f"  Num differences: {(diff_mm > 0).sum().item()} / {intermediate_size * 2}")
    print(f"  Max diff: {diff_mm.max().item()}")