#!/bin/bash
#
# Comprehensive test suite for MI355X bfloat16 prefix caching bug
#

echo "================================================================================"
echo "COMPREHENSIVE TEST SUITE FOR MI355X BFLOAT16 BUG"
echo "================================================================================"
echo ""

# Check environment
python3 -c "
import torch
from vllm.platforms.rocm import on_gfx950
arch = torch.cuda.get_device_properties('cuda').gcnArchName
print(f'GPU: {arch}')
print(f'Is gfx950: {on_gfx950()}')
print()
"

echo "================================================================================"
echo "TEST 1: Debug logging (check if IN_PRECISION is being set)"
echo "================================================================================"
cd /app
python test_with_debug.py 2>&1 | tee test1_debug.log

echo ""
echo "Checking for debug messages..."
if grep -q "🔍 GFX950 BF16 DETECTED" test1_debug.log; then
    echo "✓ IN_PRECISION logic IS being triggered"
else
    echo "✗ IN_PRECISION logic NOT being triggered (this is the problem!)"
fi

echo ""
echo "================================================================================"
echo "TEST 2: BFloat16 test (the main bug test)"
echo "================================================================================"
python test_bfloat16_fix.py

echo ""
echo "================================================================================"
echo "TEST 3: Float16 test (should always work)"
echo "================================================================================"
python debug_step1_cache_hit.py

echo ""
echo "================================================================================"
echo "ANALYSIS"
echo "================================================================================"
echo ""
echo "Check the test1_debug.log file for '🔍 GFX950 BF16 DETECTED' messages."
echo ""
echo "If you see these messages:"
echo "  - IN_PRECISION is being set correctly"
echo "  - But bug still exists"
echo "  → Problem is elsewhere (not in precision mode)"
echo ""
echo "If you DON'T see these messages:"
echo "  - IN_PRECISION logic not being triggered"
echo "  → Need to check why condition isn't matching"
echo ""
echo "Possible next steps:"
echo "  1. Check if bfloat16 detection is working: q_dtype_is_bf16"
echo "  2. Check if gfx950 detection is working: IS_GFX950"
echo "  3. Check if different attention backend is being used"
echo "  4. Check if issue is in model layers, not attention"
echo ""
