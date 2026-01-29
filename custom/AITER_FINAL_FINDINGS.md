# AITER Prefix Cache Bug - Final Findings

## Executive Summary

Through systematic debugging, we've identified that:
1. ✅ **TRITON backend is FIXED** - Deterministic with prefix caching
2. ✅ **Cache block selection is FIXED** - Runs 2-3 are deterministic with each other
3. ❌ **AITER backend has a bug when mixing cached + fresh KV blocks** - Run 1 differs from Runs 2-3

## The Bug

**Without prefix caching** (all runs identical):
```
Run 1: [785, 7513, 9145, 320, 38807, 8, 17167, 315, 3070, 17]
Run 2: [785, 7513, 9145, 320, 38807, 8, 17167, 315, 3070, 17]
Run 3: [785, 7513, 9145, 320, 38807, 8, 17167, 315, 3070, 17]
Output: "The European Union (EU) consists of **2"
```

**With prefix caching** (Run 1 vs Runs 2-3):
```
Run 1: [785, 7513, 9145, 320, 38807, 8, 17167, 315, 3070, 17]  ✅ CORRECT
Run 2: [785, 7513, 9145, 17167, 315, 220, 17, 22, 4462, 5302]  ❌ WRONG
Run 3: [785, 7513, 9145, 17167, 315, 220, 17, 22, 4462, 5302]  ❌ WRONG
```

Run 1 output: "The European Union (EU) consists of **2" (CORRECT - matches no-cache)
Runs 2-3 output: "The European Union consists of 27 member states" (WRONG - but deterministic)

## Cache Behavior Analysis

### Run 1 (Cold Cache)
```
[CACHE_GET] Cache MISS for hash=\xc4\x9f}\x12...
[CACHE_PUT] Cached block: id=1, hash=\xc4\x9f}\x12...
[CACHE_PUT] Cached block: id=2, hash=\x81\xc1\xeb:...
```
- Computes all blocks fresh
- Produces CORRECT output
- Caches blocks 1 and 2

### Run 2 (Warm Cache)
```
[CACHE_GET] Single block: id=1, hash=\xc4\x9f}\x12...
[CACHE_MGR] Cache HIT: 1 blocks
[CACHE_PUT] Cached block: id=4, hash=\x81\xc1\xeb:...  ← SAME HASH AS BLOCK 2!
```
- Reuses cached block 1
- Computes block 4 fresh (same hash as Run 1's block 2)
- Produces WRONG output

### Run 3 (Warm Cache)
```
[CACHE_GET] Single block: id=1, hash=\xc4\x9f}\x12...
[CACHE_MGR] Cache HIT: 1 blocks
[CACHE_PUT] Cached block: id=6, hash=\x81\xc1\xeb:...  ← SAME HASH AS BLOCK 2!
```
- Reuses cached block 1
- Computes block 6 fresh (same hash as Run 1's block 2)
- Produces WRONG output (but identical to Run 2)

## Key Insight

All runs compute blocks with the SAME content hashes:
- Block with hash `\xc4\x9f}\x12...` (block 1)
- Block with hash `\x81\xc1\xeb:...` (blocks 2, 4, 6)

BUT:
- When both blocks are computed fresh (Run 1): **CORRECT output**
- When block 1 is cached and block 2/4/6 is fresh (Runs 2-3): **WRONG output**

## Root Cause

The bug is in how the AITER attention kernel handles **MIXED cached + fresh KV blocks**.

When Run 1 computes with NO cached blocks:
- Q @ [K_fresh_block1, K_fresh_block2] → softmax → @ [V_fresh_block1, V_fresh_block2]
- Produces correct output

When Runs 2-3 compute with cached block 1:
- Q @ [K_cached_block1, K_fresh_block2] → softmax → @ [V_cached_block1, V_fresh_block2]
- Produces WRONG output

Possible issues:
1. **Block table indexing bug**: Cached vs fresh blocks might be indexed differently
2. **Tensor layout mismatch**: Cached blocks might be in different memory layout
3. **Attention metadata bug**: `attn_metadata` might not correctly describe mixed cached/fresh
4. **AITER kernel bug**: External `aiter.ops.triton.unified_attention` may not handle mixed KV correctly

## What We've Fixed

✅ **Fixed in both TRITON and AITER**:
1. Deterministic block selection (`block_pool.py` line 69)
   - Changed from `next(iter(blocks.values()))` to `min(blocks.values(), key=lambda b: b.block_id)`
2. GPU synchronization after caching (`block_pool.py` lines 282-286)
   - Added `torch.cuda.synchronize()` on gfx950 after caching blocks

✅ **Fixed in TRITON only**:
3. BF16 precision fix (`triton_unified_attention.py`)
   - Keep softmax probabilities in FP32 instead of converting to BF16
   - This fixed TRITON backend completely

## What's Still Broken

❌ **AITER backend**:
- Bug in how attention handles mixed cached + fresh KV blocks
- External library (`aiter.ops.triton.unified_attention`) - can't directly fix
- Runs 2-3 produce wrong outputs when reusing cached blocks

## Attempted Fixes That Didn't Work

1. ❌ GPU sync before attention (`rocm_aiter_unified_attn.py` line 163)
   - Added `torch.cuda.synchronize()` before `unified_attention()` call
   - Didn't fix the issue

2. ❌ Debug logging showed no obvious block selection issues
   - All runs correctly identify and use the same cached blocks
   - Block hashes are consistent

## Next Steps

### Option 1: Fix AITER Kernel (External)
- Contact AITER library maintainers
- Report bug: mixed cached/fresh KV blocks produce incorrect outputs
- Likely needs fix in `aiter/ops/triton/unified_attention.py`

### Option 2: Workaround in vLLM
- Detect when AITER backend is used on gfx950 with BF16
- Disable prefix caching only for this specific configuration
- Example:
  ```python
  if (backend == "AITER" and
      on_gfx950() and
      dtype == torch.bfloat16):
      enable_prefix_caching = False
  ```

### Option 3: Force TRITON Backend
- Since TRITON is now fixed, use it instead of AITER
- Set `VLLM_ROCM_USE_AITER=0`

## Test Commands

### Test TRITON (FIXED):
```bash
VLLM_ROCM_USE_AITER=0 python3 test_simple_compare.py
# Expected: All 3 runs identical ✅
```

### Test AITER without caching (WORKS):
```bash
VLLM_ROCM_USE_AITER=1 python3 debug_aiter_nocache.py
# Expected: All 3 runs identical ✅
```

### Test AITER with caching (BROKEN):
```bash
VLLM_ROCM_USE_AITER=1 python3 debug_aiter_cache.py
# Expected: Run 1 differs from Runs 2-3 ❌
```

## Conclusion

We successfully fixed the TRITON backend and improved determinism overall. The remaining AITER issue is in an external library and requires either:
1. Fixing the upstream `aiter` library
2. Working around it in vLLM
3. Using TRITON backend instead (recommended for now)
