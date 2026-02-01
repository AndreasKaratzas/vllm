#!/bin/bash
# Quick verification script to check if the fix was applied correctly

echo "======================================================================"
echo "VERIFYING PREFIX CACHING FIX FOR MI355X (gfx950)"
echo "======================================================================"
echo ""

FILE="/app/vllm/vllm/v1/attention/ops/chunked_prefill_paged_decode.py"

echo "Checking if all required changes are present..."
echo ""

# Check 1: IN_PRECISION parameter in kernel signature
echo "[1/5] Checking kernel signature for IN_PRECISION parameter..."
if grep -q "IN_PRECISION: tl.constexpr,  # precision mode for tl.dot operations" "$FILE"; then
    echo "✓ PASS: IN_PRECISION parameter added to kernel signature"
else
    echo "✗ FAIL: IN_PRECISION parameter missing from kernel signature"
    exit 1
fi

# Check 2: First tl.dot() call updated
echo "[2/5] Checking first tl.dot() call (QK computation)..."
if grep -q "qk = scale \* tl.dot(Q, K, input_precision=IN_PRECISION)" "$FILE"; then
    echo "✓ PASS: First tl.dot() call uses input_precision=IN_PRECISION"
else
    echo "✗ FAIL: First tl.dot() call not updated"
    exit 1
fi

# Check 3: Second tl.dot() call updated
echo "[3/5] Checking second tl.dot() call (accumulation)..."
if grep -q "acc += tl.dot(p.to(V.dtype), V, input_precision=IN_PRECISION)" "$FILE"; then
    echo "✓ PASS: Second tl.dot() call uses input_precision=IN_PRECISION"
else
    echo "✗ FAIL: Second tl.dot() call not updated"
    exit 1
fi

# Check 4: IN_PRECISION computation in function
echo "[4/5] Checking IN_PRECISION computation..."
if grep -q "IN_PRECISION = \"ieee\" if IS_TURING and q_dtype_is_f32 else None" "$FILE"; then
    echo "✓ PASS: IN_PRECISION computation added"
else
    echo "✗ FAIL: IN_PRECISION computation missing"
    exit 1
fi

# Check 5: IN_PRECISION passed to kernel
echo "[5/5] Checking kernel invocation..."
if grep -q "IN_PRECISION=IN_PRECISION," "$FILE"; then
    echo "✓ PASS: IN_PRECISION passed to kernel"
else
    echo "✗ FAIL: IN_PRECISION not passed to kernel"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✓ ALL CHECKS PASSED - FIX APPLIED SUCCESSFULLY!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "1. Run the test suite:"
echo "   cd /app/vllm"
echo "   pytest -s -v tests/entrypoints/openai/test_prefix_cache_debug.py::test_side_by_side"
echo ""
echo "2. Expected result after fix:"
echo "   --- VLLM WITH PREFIX CACHING ---"
echo "     Deterministic: ✓ All 5 runs identical"
echo ""
echo "3. All runs should produce identical output like:"
echo "   vLLM+PC Run 1: 'The European Union (EU) consists of **2'"
echo "   vLLM+PC Run 2: 'The European Union (EU) consists of **2'"
echo "   vLLM+PC Run 3: 'The European Union (EU) consists of **2'"
echo "   vLLM+PC Run 4: 'The European Union (EU) consists of **2'"
echo "   vLLM+PC Run 5: 'The European Union (EU) consists of **2'"
echo ""
