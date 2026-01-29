# Debug Status: MI355X BFloat16 Prefix Caching Bug

## Current Situation

After identifying that the bug is **bfloat16-specific** (float16 works fine), we made precision changes but the bug **still persists**.

## Changes Made So Far

### Files Modified

1. **`/app/vllm/vllm/v1/attention/ops/chunked_prefill_paged_decode.py`**
   - Added bfloat16 dtype detection
   - Added gfx950 platform detection
   - Force `IN_PRECISION="ieee"` for bfloat16 on gfx950
   - Added debug logging to verify it's being triggered

2. **`/app/vllm/vllm/v1/attention/ops/prefix_prefill.py`**
   - Same changes as above
   - Added debug logging

## Test Results

### Float16 ✅
```
✓ Tokens MATCH (no bug)
```

### BFloat16 ❌
```
✗ FIRST RUN DIFFERS (BUG STILL PRESENT)
  Run 1: 'The European Union consists of 27 member states'
  Run 2-5: 'The European Union (EU) consists of **2'

  Position 3 logprob difference: 0.054082
```

## What We Need to Check Now

The precision fix didn't work, so we need to diagnose WHY:

### Option 1: Fix Not Being Applied
- IN_PRECISION logic not being triggered
- Wrong condition (dtype or platform detection failing)
- Different code path being used

### Option 2: Fix Applied But Insufficient
- IN_PRECISION being set correctly
- But issue is elsewhere (not in attention kernels)
- Maybe in model layers, logits processing, or sampling

### Option 3: Multiple Issues
- Need to fix OTHER Triton kernels too:
  - `triton_decode_attention.py`
  - `triton_prefill_attention.py`
  - `triton_unified_attention.py`

### Option 4: Fundamental Issue
- HuggingFace also shows divergence with bfloat16
- Might be a deeper gfx950 + bfloat16 issue
- Not fixable at vLLM level

## Next Step: Run Diagnostic

```bash
bash /app/RUN_ALL_TESTS.sh
```

This will:
1. ✅ Run test with debug logging
2. ✅ Check if IN_PRECISION is being triggered
3. ✅ Test bfloat16 (where bug is)
4. ✅ Test float16 (should work)
5. ✅ Provide analysis and next steps

## Expected Outcomes

### If Debug Messages Appear
```
🔍 GFX950 BF16 DETECTED: IN_PRECISION=ieee, dtype=torch.bfloat16
```
→ Our fix IS being applied, but it's not sufficient
→ Need to look elsewhere (different kernels, model layers, etc.)

### If Debug Messages Don't Appear
```
(no debug messages)
```
→ Our fix is NOT being applied
→ Need to debug why:
  - Is `q_dtype_is_bf16` correctly detecting bfloat16?
  - Is `IS_GFX950` correctly detecting gfx950?
  - Is different attention backend being used?

## Files Created for Debugging

- `/app/test_with_debug.py` - Test with debug logging
- `/app/RUN_ALL_TESTS.sh` - Comprehensive test suite
- `/app/test_bfloat16_fix.py` - BFloat16-specific test
- `/app/check_backend.py` - Check which backend is selected

## Possible Root Causes (Ranked)

1. **Different attention backend** - Maybe AITER or native HIP is being used instead of our Triton kernels
2. **Multiple Triton kernels need fixing** - Not just chunked_prefill_paged_decode
3. **Issue in model layers** - Problem before attention (QKV projection)
4. **Fundamental gfx950 issue** - Hardware-level bfloat16 behavior
5. **Cache write/read precision** - Issue in how KV is written/read, not computed

## Ready to Test

Run this:
```bash
bash /app/RUN_ALL_TESTS.sh
```

Then check the output and `test1_debug.log` for the debug messages.
