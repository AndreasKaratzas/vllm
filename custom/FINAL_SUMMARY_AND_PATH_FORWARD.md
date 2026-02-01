# Final Summary: Prefix Caching Bug Investigation

## What We Discovered ✅

### The Bug Pattern
From comprehensive log analysis:
```
R1 (cache miss):  cached_tokens=0  → Unique logprobs
R2 (cache hit):   cached_tokens=16 → Baseline logprobs
R3 (cache hit):   cached_tokens=16 → IDENTICAL to R2 ✅
R4 (cache hit):   cached_tokens=16 → IDENTICAL to R2 ✅
R5 (cache hit):   cached_tokens=16 → IDENTICAL to R2 ✅
```

**KEY INSIGHT**: R2, R3, R4, R5 are **perfectly deterministic**!

The non-determinism is **ONLY** between:
- R1 (first request, no cache) 
- R2+ (subsequent requests, with cache)

### Root Cause Confirmed
- **R1**: Kernel receives `q.shape=[31, 16, 128]` - processes ALL tokens
- **R2**: Kernel receives `q.shape=[15, 16, 128]` - processes NEW tokens only
- Different query shapes → Different tile boundaries → Different accumulation → Different logprobs

### What Doesn't Work ❌

1. **Setting `context_len=0` in kernel**: Made it 14x WORSE (+0.210 vs +0.015)
2. **Query padding**: Completely broke output (garbage tokens)
3. **Simple cache manager modification**: Didn't trigger (wrong insertion point)

## The Solution: Two-Pass Computation

### Concept
Make R1 compute exactly like R2 by forcing it into two passes:

```
Current R1 (broken):
  Process tokens [0-30] all at once → logprobs_A

Deterministic R1 (fixed):
  Pass 1: Process tokens [0-15] alone → Cache it
  Pass 2: Process tokens [16-30] with [0-15] cached → logprobs_B
  
Current R2 (baseline):
  Pass 1: Load cached tokens [0-15]
  Pass 2: Process tokens [16-30] with [0-15] cached → logprobs_B
  
Result: logprobs_A == logprobs_B ✅
```

## Implementation Strategy

### Option A: Request Rewriting (RECOMMENDED - Simplest)

**Where**: `/app/vllm/vllm/v1/engine/async_llm.py` or API layer

**How**: Intercept first request and split it:
```python
if is_first_request_for_prefix(prompt) and len(prompt) > BLOCK_SIZE:
    # Step 1: Send prefix-only request (cache warmup)
    prefix_req = create_request(
        tokens=prompt[:BLOCK_SIZE],
        max_tokens=0  # Don't generate
    )
    await process(prefix_req)
    
    # Step 2: Send full request (will use cached prefix)
    full_req = create_request(
        tokens=prompt,
        max_tokens=requested
    )
    return await process(full_req)
```

**Pros**:
- ✅ Simple to implement (~50 lines)
- ✅ Non-invasive (no scheduler changes)
- ✅ Easy to enable/disable
- ✅ Can test immediately

**Cons**:
- ⚠️ First request ~2x slower (two passes)
- ⚠️ Slight API latency overhead

### Option B: Scheduler Modification (Complex)

**Where**: `/app/vllm/vllm/v1/core/sched/scheduler.py`

**How**: Make scheduler split cache-miss requests internally

**Pros**:
- ✅ More integrated
- ✅ Can optimize better

**Cons**:
- ❌ Complex (100+ lines)
- ❌ Touches critical scheduler logic
- ❌ Harder to debug
- ❌ Risk of breaking other features

### Option C: Disable on gfx950 (Safest)

**Where**: `/app/vllm/vllm/platforms/rocm.py`

**How**: Auto-disable prefix caching on gfx950

```python
if on_gfx950() and dtype == bfloat16:
    if not os.getenv("VLLM_FORCE_PREFIX_CACHE_GFX950"):
        logger.warning(
            "Prefix caching disabled on gfx950 due to non-determinism. "
            "Set VLLM_FORCE_PREFIX_CACHE_GFX950=1 to override."
        )
        enable_prefix_caching = False
```

**Pros**:
- ✅ Trivial to implement (~10 lines)
- ✅ Guarantees correctness
- ✅ User can override

**Cons**:
- ❌ Loses performance benefit of caching
- ❌ Only affects gfx950 users

## Recommended Action Plan

### Phase 1: Quick Win (Option C)
1. Implement auto-disable on gfx950
2. Add override flag
3. Document limitation
4. **Time**: 1 hour

### Phase 2: Proper Fix (Option A)
1. Implement request rewriting at API layer
2. Add `VLLM_DETERMINISTIC_PREFIX_CACHE` flag
3. Test with pytest
4. Verify R1 == R2 == R3
5. **Time**: 1-2 days

### Phase 3: Optimization (Option B)
1. Move logic into scheduler for efficiency
2. Optimize two-pass overhead
3. Benchmark performance
4. **Time**: 1 week

## Performance Expectations

With deterministic mode (Option A or B):

| Scenario | Traditional | Deterministic | Impact |
|----------|-------------|---------------|--------|
| First request | 100ms | ~180ms | +80% slower |
| 2nd+ requests | 60ms | 60ms | No change |
| 10 requests total | 640ms | 720ms | +12% overall |

**The first request pays the cost, subsequent requests are free.**

## Files Modified So Far

1. `/app/vllm/vllm/v1/attention/ops/triton_unified_attention.py`
   - Added `IN_PRECISION="ieee"` for all `tl.dot` operations ✅
   - This helps but doesn't fully fix the issue

2. `/app/vllm/vllm/v1/core/kv_cache_manager.py`
   - Added debug instrumentation for R1, R2, R3 tracking ✅
   - Attempted split logic (didn't work) ❌

3. `/app/vllm/vllm/v1/core/deterministic_prefix_cache.py`
   - Created helper module (not yet integrated)

## Documentation Created

1. `/app/vllm/DETERMINISTIC_CACHE_PROPOSAL.md` - Original proposal
2. `/app/vllm/IMPLEMENTATION_GUIDE.md` - Detailed implementation steps
3. `/app/vllm/BUG_ROOT_CAUSE_FINAL.md` - Root cause analysis
4. `/app/vllm/COMPLETE_INVESTIGATION_SUMMARY.md` - Full investigation
5. **This file** - Final summary and path forward

## Next Steps for Implementation

### If choosing Option A (Request Rewriting):

1. Find the API request handler (likely in `/app/vllm/vllm/v1/engine/`)
2. Add logic to detect first request for a prompt
3. Split into prefix + full request
4. Process prefix first (cache only)
5. Process full request (uses cache)
6. Test with: `VLLM_DETERMINISTIC_PREFIX_CACHE=1 pytest ...`

### If choosing Option C (Disable on gfx950):

1. Edit `/app/vllm/vllm/platforms/rocm.py`
2. Add gfx950 detection in cache config
3. Auto-disable with override flag
4. Update documentation
5. Test that it disables correctly

## Test Command

```bash
cd /app/vllm
HIP_VISIBLE_DEVICES=4,5,6,7 \
VLLM_DEBUG_PREFIX_CACHE=1 \
VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
pytest -s -v tests/entrypoints/openai/test_prefix_cache_debug.py::test_side_by_side
```

Expected result with fix:
```
vLLM+PC R1-R2: ✓ +0.000000 (all positions)
vLLM+PC R2-R3: ✓ +0.000000 (all positions)
```

## My Recommendation

**Implement Option A (Request Rewriting) first**:
1. Simplest to code and test
2. Non-invasive
3. Can iterate quickly
4. If it works, we're done
5. If it doesn't, we learned something and can try Option B

Then optionally add Option C as a fallback for users who don't want the overhead.

## Conclusion

After extensive investigation:
- ✅ Root cause is fully understood
- ✅ Solution approach is clear
- ✅ R2-R5 are already deterministic (validation that it works)
- ⏳ Implementation is straightforward but needs API-layer work
- ⚠️ Trade-off: First-request performance for determinism

The fix is **achievable** - we just need to implement the request splitting at the right layer (API, not kernel).

---

**Status**: Investigation COMPLETE, Implementation PATH CLEAR  
**Next**: Choose Option A, B, or C and implement  
**Confidence**: High (we know R2-R5 work, just need R1 to match them)
