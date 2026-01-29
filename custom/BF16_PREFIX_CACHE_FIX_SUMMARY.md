# BF16 Prefix Caching Non-Determinism Fix - Summary

## Problem

On AMD MI355X (gfx950) with BFloat16, prefix caching produced non-deterministic outputs across multiple runs with the same prompt. The bug occurred regardless of attention backend (TRITON or AITER).

## Root Causes Identified

### 1. Non-Deterministic Block Selection (Fixed)
**File**: `/app/vllm/vllm/v1/core/block_pool.py` line 69

**Problem**: `next(iter(blocks.values()))` returned arbitrary blocks when multiple blocks shared the same hash key.

**Fix**: Changed to `min(blocks.values(), key=lambda b: b.block_id)` to ensure deterministic selection of the block with the lowest ID.

### 2. Missing GPU Synchronization (Fixed)
**File**: `/app/vllm/vllm/v1/core/block_pool.py` line 276-281

**Problem**: No synchronization between GPU cache writes and reads on gfx950.

**Fix**: Added `torch.cuda.synchronize()` after caching blocks on gfx950 to ensure writes complete before blocks are marked as cached.

### 3. BF16 Precision Loss in Triton Kernels (Fixed for TRITON backend)
**File**: `/app/vllm/vllm/v1/attention/ops/triton_unified_attention.py`

**Problem**: Converting FP32 softmax probabilities to BF16 before accumulation:
```python
acc += tl.dot(P.to(V.dtype), V)  # P is FP32, V.dtype is BF16
```

This lossy conversion causes different results when tiles are processed in different orders (cache hit vs miss).

**Fix**: Keep P in FP32, upcast V to FP32 for the dot product:
```python
if IN_PRECISION is not None:
    # Keep P in FP32, upcast V to match P's dtype
    acc += tl.dot(P, V.to(P.dtype), input_precision=IN_PRECISION)
else:
    # Original behavior for other platforms
    acc += tl.dot(P.to(V.dtype), V)
```

## Changes Made

### Modified Files

1. **`/app/vllm/vllm/v1/core/block_pool.py`**
   - Line 69: Fixed non-deterministic block selection
   - Lines 276-281: Added GPU synchronization for gfx950

2. **`/app/vllm/vllm/v1/core/single_type_kv_cache_manager.py`**
   - Removed incomplete gfx950 workaround (lines 434-440)

3. **`/app/vllm/vllm/v1/attention/ops/triton_unified_attention.py`**
   - Line 106: Added `IN_PRECISION` parameter to 2D kernel signature
   - Lines 383-393: Fixed BF16 accumulation in 2D kernel
   - Line 464: Added `IN_PRECISION` parameter to 3D kernel signature
   - Lines 746-756: Fixed BF16 accumulation in 3D kernel
   - Lines 937-943: Set `IN_PRECISION="ieee"` for BF16 on gfx950
   - Line 1067: Pass `IN_PRECISION` to 2D kernel call
   - Line 1120: Pass `IN_PRECISION` to 3D kernel call

### Created Files

- `/app/vllm/debug_layer_outputs.py` - Debug script for layer-by-layer output comparison

## Test Results

### ✅ TRITON Backend (VLLM_ROCM_USE_AITER=0)

**Simple Consistency Test**:
```bash
python3 test_simple_compare.py
```
**Result**: All 3 runs produce identical tokens ✅

**Pytest**:
```bash
VLLM_ROCM_USE_AITER=0 pytest tests/entrypoints/openai/test_serving_tokens.py::test_same_response_as_chat_completions
```
**Result**: PASSED ✅

### ❌ AITER Backend (VLLM_ROCM_USE_AITER=1)

**Simple Consistency Test**:
```bash
VLLM_ROCM_USE_AITER=1 python3 test_simple_compare.py
```
**Result**: Runs 2-3 match, but Run 1 differs ❌

**Pytest**:
```bash
VLLM_ROCM_USE_AITER=1 pytest tests/entrypoints/openai/test_serving_tokens.py::test_same_response_as_chat_completions
```
**Result**: FAILED - Different outputs between `/inference/v1/generate` and `/v1/chat/completions` endpoints ❌

**Note**: AITER backend uses its own implementation in `/app/vllm/vllm/v1/attention/backends/rocm_aiter_unified_attn.py` which likely has the same BF16 precision issue but was not modified per user instructions.

## Technical Details

### Why the Fix Works

1. **Prevents lossy FP32→BF16 conversion**: By keeping softmax probabilities P in FP32 during accumulation, we avoid the precision loss that causes different tile orders to produce different results.

2. **Safe upcast for V**: Converting BF16 → FP32 is a safe operation that gains precision, unlike the lossy FP32 → BF16 downcast.

3. **IEEE precision**: Using `input_precision="ieee"` ensures the matmul operation uses full FP32 precision on gfx950.

4. **Platform-specific**: The fix only applies to gfx950 with BF16, maintaining performance on other platforms.

### Why Cache vs No-Cache Still Differs

Small logprob differences between cache hit and cache miss modes are **expected behavior**, not a bug:

- Different computation paths: `Q[31] @ K[31]` vs `Q[15] @ [K_cached[16]; K_fresh[15]]`
- Different softmax denominators and accumulation orders
- Even FP32 isn't perfectly associative: `(a + b) + c ≠ a + (b + c)`

**Important**: The fix ensures **determinism within each mode**, not identical values across modes.

## Performance Impact

- ✅ Other platforms/dtypes: No impact (uses original fast path)
- ⚠️ BF16 on gfx950: Slight slowdown due to FP32 accumulation
- **Trade-off**: Correctness > speed

## Summary

✅ **TRITON Backend**: Fully fixed, deterministic prefix caching with BF16 on gfx950
❌ **AITER Backend**: Still has non-determinism, requires similar fix in AITER-specific attention implementation
✅ **Prefix caching**: Remains enabled, working correctly with TRITON backend
✅ **Backward compatible**: Changes only affect gfx950 with BF16
