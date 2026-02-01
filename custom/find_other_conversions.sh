#!/bin/bash
#
# Search for other potential BF16 conversion sites that might need fixing
#

echo "================================================================================"
echo "Searching for potential BF16 conversion issues in attention kernels"
echo "================================================================================"
echo ""

echo "1. Looking for .to(V.dtype) conversions..."
grep -rn "\.to(V\.dtype)" /app/vllm/vllm/v1/attention/ --include="*.py"

echo ""
echo "2. Looking for .to(tl.bfloat16) conversions..."
grep -rn "\.to(tl\.bfloat16)" /app/vllm/vllm/v1/attention/ --include="*.py"

echo ""
echo "3. Looking for .to(torch.bfloat16) conversions..."
grep -rn "\.to(torch\.bfloat16)" /app/vllm/vllm/v1/attention/ --include="*.py"

echo ""
echo "4. Looking for .to(bf16) conversions..."
grep -rn "\.to(.*bf16)" /app/vllm/vllm/v1/attention/ --include="*.py"

echo ""
echo "5. Checking for other .to() in triton kernels..."
grep -n "tl\.dot.*\.to(" /app/vllm/vllm/v1/attention/ops/triton*.py

echo ""
echo "================================================================================"
echo "Analysis complete"
echo "================================================================================"
echo ""
echo "If any results show up above (other than our fixed lines ~389 and ~755),"
echo "those might be additional conversion sites that need fixing."
echo ""
