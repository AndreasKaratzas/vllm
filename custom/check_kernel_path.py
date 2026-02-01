#!/usr/bin/env python3
"""
Diagnostic script to check which attention kernel path is being used.
"""

import torch
from vllm.platforms.rocm import use_rocm_custom_paged_attention

# Simulate typical test conditions
test_configs = [
    {
        "name": "Default (fp16, head_size=128, block_size=16)",
        "qtype": torch.float16,
        "head_size": 128,
        "block_size": 16,
        "gqa_ratio": 1,
        "max_seq_len": 512,
        "sliding_window": 0,
        "kv_cache_dtype": "auto",
        "alibi_slopes": None,
        "sinks": None,
    },
    {
        "name": "With block_size=32",
        "qtype": torch.float16,
        "head_size": 128,
        "block_size": 32,
        "gqa_ratio": 1,
        "max_seq_len": 512,
        "sliding_window": 0,
        "kv_cache_dtype": "auto",
        "alibi_slopes": None,
        "sinks": None,
    },
    {
        "name": "With block_size=544 (non-power-of-2)",
        "qtype": torch.float16,
        "head_size": 128,
        "block_size": 544,
        "gqa_ratio": 1,
        "max_seq_len": 512,
        "sliding_window": 0,
        "kv_cache_dtype": "auto",
        "alibi_slopes": None,
        "sinks": None,
    },
]

print("="*80)
print("KERNEL PATH DIAGNOSTIC")
print("="*80)
print()

# Check GPU architecture
GPU_ARCH = torch.cuda.get_device_properties("cuda").gcnArchName
print(f"GPU Architecture: {GPU_ARCH}")
print()

import os
import vllm.envs as envs

print("Environment:")
print(f"  VLLM_ROCM_CUSTOM_PAGED_ATTN env: {os.getenv('VLLM_ROCM_CUSTOM_PAGED_ATTN', 'not set')}")
print(f"  VLLM_ROCM_CUSTOM_PAGED_ATTN value: {envs.VLLM_ROCM_CUSTOM_PAGED_ATTN}")
print()

print("="*80)
print("TESTING DIFFERENT CONFIGURATIONS")
print("="*80)

for config in test_configs:
    print(f"\n{config['name']}")
    print("-" * 60)

    use_custom = use_rocm_custom_paged_attention(
        config["qtype"],
        config["head_size"],
        config["block_size"],
        config["gqa_ratio"],
        config["max_seq_len"],
        config["sliding_window"],
        config["kv_cache_dtype"],
        config["alibi_slopes"],
        config["sinks"],
    )

    # Check if power of 2
    block_size = config["block_size"]
    is_pow2 = block_size > 0 and (block_size & (block_size - 1) == 0)

    # Final decision
    if use_custom and is_pow2:
        kernel = "Native HIP C++ (ops.paged_attention_rocm)"
        uses_fix = "✗ NO - Our IN_PRECISION fix doesn't apply"
    else:
        kernel = "Triton (kernel_paged_attention_2d)"
        uses_fix = "✓ YES - Our IN_PRECISION fix applies"

    print(f"  Block size: {block_size} ({'power-of-2' if is_pow2 else 'non-power-of-2'})")
    print(f"  use_custom: {use_custom}")
    print(f"  Kernel used: {kernel}")
    print(f"  Fix applies: {uses_fix}")

print()
print("="*80)
print("RECOMMENDATION")
print("="*80)

if envs.VLLM_ROCM_CUSTOM_PAGED_ATTN:
    print("""
⚠️  Native HIP kernel may be used, bypassing our IN_PRECISION fix!

To force Triton kernel (where fix is applied), use ONE of these methods:

METHOD 1: Disable custom paged attention (Recommended)
  export VLLM_ROCM_CUSTOM_PAGED_ATTN=0

  Then run your test/server normally.

METHOD 2: Use non-power-of-2 block size
  vllm serve MODEL --block-size 544

  This forces Triton path even with custom paged attention enabled.

METHOD 3: Run test with forced Triton:
  bash /app/test_with_triton_kernel.sh
""")
else:
    print("""
✓ Triton kernel will be used (custom paged attention is disabled)

Your IN_PRECISION fix should apply correctly!
""")

print("="*80)
