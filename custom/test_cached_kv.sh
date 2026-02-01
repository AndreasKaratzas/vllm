#!/bin/bash
#
# Test to compare cached KV values between runs
#

echo "================================================================================"
echo "MI355X BF16 PREFIX CACHING - CACHED KV VALUE COMPARISON"
echo "================================================================================"
echo ""
echo "Running test and capturing [CACHED_BLOCK] logs..."
echo "This shows the actual cached KV block values (not entire cache pool)"
echo ""

python3 test_simple_compare.py 2>&1 | tee cached_kv_test.log

echo ""
echo "================================================================================"
echo "ANALYSIS: Comparing cached KV values between runs"
echo "================================================================================"
echo ""

# Extract layer 0 cached block values
echo "Layer 0 - First cached block KV statistics:"
echo ""

echo "Run 1 (fresh, max_seqlen_q=31):"
grep "\[CACHED_BLOCK\].*layer=0.*max_seqlen_q=31" cached_kv_test.log | head -1

echo ""
echo "Run 2 (cached, max_seqlen_q=15):"
grep "\[CACHED_BLOCK\].*layer=0.*max_seqlen_q=15" cached_kv_test.log | head -1

echo ""
echo "Run 3 (cached, max_seqlen_q=15):"
grep "\[CACHED_BLOCK\].*layer=0.*max_seqlen_q=15" cached_kv_test.log | tail -1

echo ""
echo "--------------------------------------------------------------------------------"
echo ""

# Show divergence
if grep -q "DIVERGENCE DETECTED" cached_kv_test.log; then
    echo "✗ BUG CONFIRMED"
    grep -A 5 "DIVERGENCE DETECTED" cached_kv_test.log
else
    echo "✓ NO DIVERGENCE"
fi

echo ""
echo "================================================================================"
echo "VERDICT"
echo "================================================================================"
echo ""
echo "Compare the k_mean values above:"
echo ""
echo "If Run 2 and Run 3 have IDENTICAL k_mean for block_idx:"
echo "  → The SAME cached block is being read (good!)"
echo ""
echo "If Run 1's k_mean (when written) matches Run 2/3's k_mean (when read):"
echo "  → Cached KV values are preserved correctly"
echo "  → Bug is in the attention computation itself"
echo ""
echo "If k_mean DIFFERS between Run 1 write and Run 2/3 read:"
echo "  → Cache storage/retrieval bug (values changed!)"
echo ""
echo "Full log: cached_kv_test.log"
echo ""
