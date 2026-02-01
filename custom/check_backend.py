#!/usr/bin/env python3
"""
Check which attention backend is being selected
"""

import torch
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950, on_gfx942

print("="*80)
print("CHECKING ATTENTION BACKEND SELECTION")
print("="*80)
print()

# Check platform
arch = torch.cuda.get_device_properties('cuda').gcnArchName
print(f"GPU Architecture: {arch}")
print(f"Is ROCm: {current_platform.is_rocm()}")
print(f"Is gfx950 (MI355X): {on_gfx950()}")
print(f"Is gfx942 (MI325X): {on_gfx942()}")
print()

# Check which backend would be selected
print("="*80)
print("BACKEND SELECTION LOGIC")
print("="*80)
print()

# The backend selection happens in vllm/v1/attention/selector.py
# or vllm/platforms/rocm.py

from vllm import envs

print(f"VLLM_ROCM_CUSTOM_PAGED_ATTN: {envs.VLLM_ROCM_CUSTOM_PAGED_ATTN}")
print()

print("For gfx950 with enforce-eager:")
print("  Likely backend: ROCM_AITER_FA or TRITON_ATTN")
print()

print("To force specific backend, set:")
print("  VLLM_ATTENTION_BACKEND=TRITON_ATTN")
print("  VLLM_ATTENTION_BACKEND=ROCM_AITER_FA")
print("  VLLM_ATTENTION_BACKEND=ROCM_AITER_UNIFIED_ATTN")
print()

print("="*80)
print("RECOMMENDATION")
print("="*80)
print("""
We need to check and potentially fix ALL Triton attention kernels:
  1. chunked_prefill_paged_decode.py (✓ already fixed)
  2. prefix_prefill.py (✓ already fixed)
  3. triton_decode_attention.py (needs check)
  4. triton_prefill_attention.py (needs check)
  5. triton_unified_attention.py (needs check)

Let's check which one is actually being used in your tests.
""")
