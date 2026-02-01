#!/bin/bash
#
# Test the ACTUAL FIX for BF16 prefix caching bug
#

echo "================================================================================"
echo "MI355X BF16 PREFIX CACHING - TESTING ACTUAL FIX"
echo "================================================================================"
echo ""
echo "THE BUG:"
echo "  Line 384 in triton_unified_attention.py converted P (FP32) to BF16"
echo "  before dot product: acc += tl.dot(P.to(V.dtype), V, ...)"
echo ""
echo "  When tiles are processed in different order (cache hit vs miss),"
echo "  BF16 accumulation causes different rounding → different outputs"
echo ""
echo "THE FIX:"
echo "  When IN_PRECISION is set (BF16 on gfx950), keep P in FP32:"
echo "  acc += tl.dot(P, V.to(tl.float32), input_precision=IN_PRECISION)"
echo ""
echo "  This ensures deterministic accumulation regardless of tile order."
echo ""
echo "--------------------------------------------------------------------------------"
echo ""

python3 test_simple_compare.py 2>&1 | tee fix_test.log

echo ""
echo "================================================================================"
echo "RESULTS"
echo "================================================================================"
echo ""

# Check for divergence
if grep -q "DIVERGENCE DETECTED" fix_test.log; then
    echo "✗ BUG STILL PRESENT"
    echo ""
    grep -A 5 "DIVERGENCE DETECTED" fix_test.log
    echo ""
    echo "The fix may not be complete. Check fix_test.log for details."
else
    echo "✓ BUG FIXED! All runs produce IDENTICAL outputs!"
    echo ""
    grep "All runs produced same tokens" fix_test.log
fi

echo ""
echo "--------------------------------------------------------------------------------"
echo ""

# Show prefix cache is still active
echo "Prefix cache status:"
grep "\[PREFIX_CACHE\]" fix_test.log | head -3

echo ""
if grep -q "num_computed_tokens=16" fix_test.log; then
    echo "✓ Prefix caching is ACTIVE (16 tokens reused from cache)"
else
    echo "⚠ Prefix caching may not be working"
fi

echo ""
echo "Full log: fix_test.log"
echo ""
