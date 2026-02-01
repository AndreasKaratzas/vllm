#!/bin/bash
#
# Quick debugging script - runs simple test with logging enabled
#

echo "================================================================================"
echo "QUICK DEBUG: MI355X BFLOAT16 PREFIX CACHING BUG"
echo "================================================================================"
echo ""
echo "This will:"
echo "  1. Run 3 identical requests with BF16"
echo "  2. Compare outputs"
echo "  3. Show debug logs from vLLM internals"
echo ""
echo "Look for these debug markers in the output:"
echo "  [KV_WRITE]  - When KV cache is written (cache miss)"
echo "  [ATTN_FWD]  - Attention forward pass (shows prefill vs decode)"
echo "  [ATTN_OUT]  - Attention output statistics"
echo ""
echo "--------------------------------------------------------------------------------"
echo ""

# Run simple comparison test
python3 test_simple_compare.py 2>&1 | tee debug_output.log

echo ""
echo "================================================================================"
echo "ANALYSIS"
echo "================================================================================"
echo ""

if grep -q "DIVERGENCE DETECTED" debug_output.log; then
    echo "✗ BUG CONFIRMED - Outputs differ between runs"
    echo ""

    # Extract key information
    echo "Key observations:"
    echo ""

    # Count KV_WRITE occurrences
    kv_writes=$(grep -c "\[KV_WRITE\]" debug_output.log || echo "0")
    echo "  - Number of KV cache writes: $kv_writes"

    # Count ATTN_FWD with prefill vs decode
    prefill_count=$(grep "\[ATTN_FWD\]" debug_output.log | grep -c "prefill=True" || echo "0")
    decode_count=$(grep "\[ATTN_FWD\]" debug_output.log | grep -c "decode=True" || echo "0")
    echo "  - Prefill operations: $prefill_count"
    echo "  - Decode operations: $decode_count"

    echo ""
    echo "Detailed attention logs:"
    grep "\[ATTN_FWD\]\|\[ATTN_OUT\]" debug_output.log | head -20

    echo ""
    echo "Next steps:"
    echo "  1. Check if attention outputs differ between runs"
    echo "  2. Compare [ATTN_OUT] mean/std values"
    echo "  3. If they differ, the bug is in attention or model layers"

elif grep -q "All runs are CONSISTENT" debug_output.log; then
    echo "✓ NO BUG - All runs are consistent!"
    echo "The fixes may have resolved the issue."
else
    echo "⚠️  Test result unclear - check debug_output.log"
fi

echo ""
echo "Full output saved to: debug_output.log"
echo "To see just the debug markers:"
echo "  grep '\\[KV_WRITE\\]\\|\\[ATTN' debug_output.log"
echo ""
