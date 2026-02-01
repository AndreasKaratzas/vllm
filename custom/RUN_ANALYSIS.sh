#!/bin/bash
#
# Run test with enhanced KV cache logging
#

echo "================================================================================"
echo "MI355X BFLOAT16 PREFIX CACHING - KV CACHE ANALYSIS"
echo "================================================================================"
echo ""
echo "Added logging to show KV cache statistics in unified_attention kernel:"
echo "  [UNIFIED_ATTN_2D/3D] - Shows K/V cache mean values and shapes"
echo ""
echo "This will help us determine if the cached KV values are identical"
echo "between Run 1 (fresh) and Runs 2-3 (cached)."
echo ""
echo "--------------------------------------------------------------------------------"
echo ""

python3 test_simple_compare.py 2>&1 | tee kv_analysis.log

echo ""
echo "================================================================================"
echo "ANALYSIS - Compare KV cache values between runs"
echo "================================================================================"
echo ""

# Show first occurrence of UNIFIED_ATTN for each run
echo "Run 1 (fresh computation, 31 tokens):"
grep "\[UNIFIED_ATTN" kv_analysis.log | head -3

echo ""
echo "Run 2 (cached, 15 fresh + 16 cached tokens):"
grep "\[UNIFIED_ATTN" kv_analysis.log | grep -A 3 "Run 2" | head -6

echo ""
echo "--------------------------------------------------------------------------------"
echo "KEY QUESTION: Does k_mean differ between Run 1 and Runs 2-3?"
echo ""
echo "If k_mean is IDENTICAL:"
echo "  → KV cache storage is working correctly"
echo "  → Bug is in the attention kernel computation itself"
echo ""
echo "If k_mean DIFFERS:"
echo "  → KV cache has a storage/retrieval bug"
echo "  → Cached values are not being preserved correctly"
echo ""
echo "Full output in: kv_analysis.log"
echo ""
