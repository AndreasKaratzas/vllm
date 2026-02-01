#!/bin/bash
#
# Test the BF16 prefix caching workaround
#

echo "================================================================================"
echo "Testing BF16 Prefix Caching Workaround on MI355X"
echo "================================================================================"
echo ""
echo "The workaround should:"
echo "  1. Detect BF16 + gfx950 combination"
echo "  2. Disable prefix caching automatically"
echo "  3. Log a warning message"
echo "  4. Produce consistent outputs (no divergence)"
echo ""
echo "--------------------------------------------------------------------------------"
echo ""

python3 test_simple_compare.py 2>&1 | tee workaround_test.log

echo ""
echo "================================================================================"
echo "ANALYSIS"
echo "================================================================================"
echo ""

# Check if workaround was triggered
if grep -q "Disabling prefix caching for BFloat16 on AMD MI355X" workaround_test.log; then
    echo "✓ Workaround ACTIVE - Prefix caching disabled for BF16"
    echo ""
    grep "Disabling prefix caching" workaround_test.log
else
    echo "✗ Workaround NOT triggered"
fi

echo ""
echo "--------------------------------------------------------------------------------"
echo ""

# Check for divergence
if grep -q "DIVERGENCE DETECTED" workaround_test.log; then
    echo "✗ BUG STILL PRESENT - Outputs differ"
    echo ""
    grep -A 5 "DIVERGENCE DETECTED" workaround_test.log
else
    echo "✓ NO DIVERGENCE - All runs produced same output!"
fi

echo ""
echo "--------------------------------------------------------------------------------"
echo ""

# Check if prefix cache was actually disabled
echo "Prefix cache statistics:"
grep "\[PREFIX_CACHE\]" workaround_test.log | head -3

echo ""
echo "If num_computed_tokens=0 for ALL runs:"
echo "  → Prefix caching is disabled (expected with workaround)"
echo ""
echo "If num_computed_tokens=16 for runs 2-3:"
echo "  → Prefix caching is still active (workaround failed)"
echo ""
echo "Full log: workaround_test.log"
echo ""
