#!/bin/bash
#
# Comprehensive verification of the bfloat16 fix on MI355X (gfx950)
#

echo "================================================================================"
echo "VERIFYING BFLOAT16 FIX FOR MI355X PREFIX CACHING BUG"
echo "================================================================================"
echo ""

# Check GPU architecture
python3 -c "
import torch
from vllm.platforms.rocm import on_gfx950
arch = torch.cuda.get_device_properties('cuda').gcnArchName
print(f'GPU Architecture: {arch}')
print(f'Is gfx950 (MI355X): {on_gfx950()}')
print()
if not on_gfx950():
    print('⚠️  WARNING: Not running on gfx950. This fix is specific to MI355X.')
    print()
"

echo "================================================================================"
echo "TEST 1: BFLOAT16 (where bug occurs)"
echo "================================================================================"
cd /app
python test_bfloat16_fix.py

EXIT_CODE_BF16=$?

echo ""
echo "================================================================================"
echo "TEST 2: FLOAT16 (should always work)"
echo "================================================================================"
python debug_step1_cache_hit.py

EXIT_CODE_FP16=$?

echo ""
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"

if [ $EXIT_CODE_BF16 -eq 0 ]; then
    echo "✓ BFLOAT16 test passed"
else
    echo "✗ BFLOAT16 test failed (exit code: $EXIT_CODE_BF16)"
fi

if [ $EXIT_CODE_FP16 -eq 0 ]; then
    echo "✓ FLOAT16 test passed"
else
    echo "✗ FLOAT16 test failed (exit code: $EXIT_CODE_FP16)"
fi

echo ""
echo "================================================================================"
echo "WHAT THE FIX DOES"
echo "================================================================================"
echo "
The fix forces IEEE precision mode for bfloat16 on gfx950 (MI355X) in BOTH:
  1. prefix_prefill.py (cache-miss path)
  2. chunked_prefill_paged_decode.py (cache-hit path)

This ensures numerical consistency between the two code paths.

Files modified:
  - /app/vllm/vllm/v1/attention/ops/chunked_prefill_paged_decode.py
  - /app/vllm/vllm/v1/attention/ops/prefix_prefill.py

Changes:
  - Added bfloat16 dtype detection
  - Added gfx950 platform detection
  - Force IN_PRECISION='ieee' for bfloat16 on gfx950
"

echo "================================================================================"
