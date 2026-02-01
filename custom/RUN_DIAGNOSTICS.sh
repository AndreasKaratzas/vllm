#!/bin/bash
#
# Comprehensive diagnostics for MI355X bfloat16 prefix caching bug
#

echo "================================================================================"
echo "DIAGNOSTIC SUITE FOR MI355X BFLOAT16 PREFIX CACHING BUG"
echo "================================================================================"
echo ""

# Check environment
python3 -c "
import torch
from vllm.platforms.rocm import on_gfx950
arch = torch.cuda.get_device_properties('cuda').gcnArchName
print(f'GPU: {arch}')
print(f'Is gfx950: {on_gfx950()}')
print(f'PyTorch version: {torch.__version__}')
print()
"

echo "================================================================================"
echo "TEST 1: PyTorch BF16 Determinism (is the bug in PyTorch/ROCm itself?)"
echo "================================================================================"
python3 test_torch_bf16_consistency.py
echo ""

echo "================================================================================"
echo "TEST 2: KV Cache Roundtrip (does cache write/read preserve values?)"
echo "================================================================================"
python3 test_cache_roundtrip.py
echo ""

echo "================================================================================"
echo "TEST 3: vLLM End-to-End (the actual bug - BF16 with prefix caching)"
echo "================================================================================"
python3 test_bfloat16_fix.py
echo ""

echo "================================================================================"
echo "ANALYSIS GUIDE"
echo "================================================================================"
echo ""
echo "If TEST 1 FAILS (PyTorch is non-deterministic with BF16):"
echo "  → This is a ROCm/gfx950 bug, not vLLM"
echo "  → Workaround: Use FP16 instead of BF16"
echo "  → Possible fix: Update ROCm to latest version"
echo ""
echo "If TEST 1 PASSES but TEST 2 FAILS (cache roundtrip corrupts data):"
echo "  → Bug is in Triton reshape_and_cache kernel"
echo "  → Need to fix tl.load/tl.store implicit conversion"
echo ""
echo "If TEST 1 and TEST 2 PASS but TEST 3 FAILS (vLLM still has bug):"
echo "  → Bug is in prefix caching logic or model computation"
echo "  → Need to trace through attention path more carefully"
echo ""
