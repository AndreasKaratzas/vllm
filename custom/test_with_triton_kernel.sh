#!/bin/bash
#
# Test vLLM with FORCED Triton kernel (not native HIP kernel)
# This ensures our IN_PRECISION fix is actually used
#

echo "================================================================================"
echo "TESTING WITH TRITON KERNEL (FORCING TRITON PATH)"
echo "================================================================================"
echo ""
echo "Why: The native HIP C++ kernel (ops.paged_attention_rocm) may bypass our fix."
echo "     By setting VLLM_ROCM_CUSTOM_PAGED_ATTN=0, we force the Triton kernel"
echo "     where our IN_PRECISION parameter is applied."
echo ""
echo "================================================================================"
echo ""

cd /app/vllm

# Disable custom ROCm paged attention to force Triton path
export VLLM_ROCM_CUSTOM_PAGED_ATTN=0
export VLLM_LOGGING_LEVEL=ERROR

echo "Environment:"
echo "  VLLM_ROCM_CUSTOM_PAGED_ATTN=${VLLM_ROCM_CUSTOM_PAGED_ATTN}"
echo ""
echo "Running test..."
echo ""

pytest -s -v tests/entrypoints/openai/test_prefix_cache_debug.py::test_side_by_side

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ TEST PASSED"
else
    echo "✗ TEST FAILED (exit code: $EXIT_CODE)"
fi
echo "================================================================================"

exit $EXIT_CODE
