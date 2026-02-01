#!/bin/bash
#
# Run this to see the actual bug with proper logging
#

echo "================================================================================"
echo "MI355X BFLOAT16 PREFIX CACHING BUG - WITH CORRECT LOGGING"
echo "================================================================================"
echo ""
echo "I've added logging to the CORRECT files:"
echo "  1. /app/vllm/vllm/v1/core/kv_cache_manager.py - prefix cache block selection"
echo "  2. /app/vllm/vllm/v1/attention/backends/triton_attn.py - attention forward & KV write"
echo ""
echo "Look for these markers:"
echo "  [PREFIX_CACHE] - Shows which cached blocks are reused"
echo "  [KV_WRITE]     - Shows when KV is written to cache (cache miss)"
echo "  [TRITON_ATTN_FWD] - Shows attention forward pass info"
echo "  [TRITON_ATTN_OUT] - Shows attention output statistics"
echo ""
echo "--------------------------------------------------------------------------------"
echo ""

python3 test_simple_compare.py 2>&1 | tee debug_triton.log

echo ""
echo "================================================================================"
echo "ANALYSIS"
echo "================================================================================"
echo ""

if grep -q "DIVERGENCE DETECTED" debug_triton.log; then
    echo "✗ BUG CONFIRMED"
    echo ""

    # Show prefix cache info
    echo "Prefix cache hits:"
    grep "\[PREFIX_CACHE\]" debug_triton.log | head -10

    echo ""
    echo "Attention outputs (compare mean/std between runs):"
    grep "\[TRITON_ATTN_OUT\]" debug_triton.log | grep "layer=0" | head -10

    echo ""
    echo "Full analysis:"
    echo "  1. Check if num_computed_tokens differs between runs"
    echo "  2. Compare output_mean values for layer 0"
    echo "  3. If they differ, that's where the bug is!"

else
    echo "✓ BUG FIXED!"
fi

echo ""
echo "Full output in: debug_triton.log"
echo ""
echo "To see just the debug markers:"
echo "  grep '\\[PREFIX_CACHE\\]\\|\\[TRITON_ATTN' debug_triton.log"
echo ""
