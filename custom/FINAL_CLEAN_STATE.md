# Final Clean State - Only the Actual Fix Remains

## Modified Files

### 1. `/app/vllm/vllm/v1/attention/ops/triton_unified_attention.py`

**Lines 921-923**: Sets IN_PRECISION for BF16 on gfx950
```python
if IS_GFX950 and q_dtype_is_bf16:
    IN_PRECISION = "ieee"
else:
    IN_PRECISION = None
```

**Lines 737-746** (3D kernel): The actual fix
```python
# Fix for BF16 prefix caching: when IN_PRECISION is set (e.g., "ieee" for gfx950),
# keep P in FP32 to avoid precision loss from P.to(V.dtype) conversion.
# This ensures deterministic outputs regardless of tile processing order.
if IN_PRECISION is not None:
    # Keep P in FP32, upcast V to FP32 for the dot product
    acc += tl.dot(P, V.to(tl.float32), input_precision=IN_PRECISION)
else:
    # Original behavior: convert P to V's dtype
    acc += tl.dot(P.to(V.dtype), V, input_precision=IN_PRECISION)
```

**Note**: The 2D kernel (line 383) does NOT have this fix because:
1. The 2D kernel signature doesn't have `IN_PRECISION` parameter
2. The 2D kernel's dot products don't use `input_precision`

**Status**: Need to verify which kernel is actually being used

## Removed Modifications

✓ All debug logging removed from:
- triton_attn.py
- kv_cache_manager.py

✓ Workaround reverted:
- kv_cache_manager.py (prefix caching is NOT disabled)

✓ Earlier failed fixes removed:
- No modifications to other triton kernel files
- No modifications to flash_attn.py

## What Needs Verification

1. **Which kernel is actually used?**
   - Logs show `[UNIFIED_ATTN_2D]` but the 2D kernel doesn't have the fix
   - The 3D kernel has the fix but may not be called

2. **Does the 2D kernel need IN_PRECISION added?**
   - If 2D kernel is used, need to add IN_PRECISION parameter to signature
   - Then add the same fix at line 383

3. **Test the fix**:
   ```bash
   bash test_fix.sh
   ```

## Summary

The fix is **only in the 3D kernel** (lines 737-746). If the system uses the 2D kernel (as logs suggest), the fix needs to be applied there too.

**Action needed**:
1. Determine which kernel is actually executing
2. If 2D kernel, add IN_PRECISION parameter and apply fix to line 383
3. Test to confirm no divergence
