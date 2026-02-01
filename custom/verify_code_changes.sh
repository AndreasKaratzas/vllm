#!/bin/bash
#
# Quick verification that code changes were applied correctly
#

echo "================================================================================"
echo "VERIFYING CODE CHANGES"
echo "================================================================================"
echo ""

FILE1="/app/vllm/vllm/v1/attention/ops/chunked_prefill_paged_decode.py"
FILE2="/app/vllm/vllm/v1/attention/ops/prefix_prefill.py"

echo "[1/6] Checking for bfloat16 detection in chunked_prefill_paged_decode.py..."
if grep -q "q_dtype_is_bf16 = query.dtype is torch.bfloat16" "$FILE1"; then
    echo "✓ PASS: bfloat16 detection added"
else
    echo "✗ FAIL: bfloat16 detection missing"
    exit 1
fi

echo "[2/6] Checking for gfx950 detection in chunked_prefill_paged_decode.py..."
if grep -q "from vllm.platforms.rocm import on_gfx950" "$FILE1" && \
   grep -q "IS_GFX950 = current_platform.is_rocm() and on_gfx950()" "$FILE1"; then
    echo "✓ PASS: gfx950 detection added"
else
    echo "✗ FAIL: gfx950 detection missing"
    exit 1
fi

echo "[3/6] Checking for IEEE precision for bfloat16+gfx950 in chunked_prefill_paged_decode.py..."
if grep -q "elif IS_GFX950 and q_dtype_is_bf16:" "$FILE1"; then
    echo "✓ PASS: IEEE precision for bfloat16+gfx950 added"
else
    echo "✗ FAIL: IEEE precision logic missing"
    exit 1
fi

echo "[4/6] Checking for bfloat16 detection in prefix_prefill.py..."
if grep -q "q_dtype_is_bf16 = q.dtype is torch.bfloat16" "$FILE2"; then
    echo "✓ PASS: bfloat16 detection added"
else
    echo "✗ FAIL: bfloat16 detection missing"
    exit 1
fi

echo "[5/6] Checking for gfx950 detection in prefix_prefill.py..."
if grep -q "from vllm.platforms.rocm import on_gfx950" "$FILE2" && \
   grep -q "IS_GFX950 = current_platform.is_rocm() and on_gfx950()" "$FILE2"; then
    echo "✓ PASS: gfx950 detection added"
else
    echo "✗ FAIL: gfx950 detection missing"
    exit 1
fi

echo "[6/6] Checking for IEEE precision for bfloat16+gfx950 in prefix_prefill.py..."
if grep -q "elif IS_GFX950 and q_dtype_is_bf16:" "$FILE2"; then
    echo "✓ PASS: IEEE precision for bfloat16+gfx950 added"
else
    echo "✗ FAIL: IEEE precision logic missing"
    exit 1
fi

echo ""
echo "================================================================================"
echo "✓ ALL CODE CHANGES VERIFIED!"
echo "================================================================================"
echo ""
echo "Summary of changes:"
echo "  1. Added bfloat16 dtype detection (q_dtype_is_bf16)"
echo "  2. Added gfx950 platform detection (IS_GFX950)"
echo "  3. Force IN_PRECISION='ieee' for bfloat16 on gfx950"
echo "  4. Applied to BOTH cache-miss and cache-hit paths"
echo ""
echo "Files modified:"
echo "  - $FILE1"
echo "  - $FILE2"
echo ""
echo "Next step: Run verification tests"
echo "  bash /app/verify_bfloat16_fix.sh"
echo ""
