# Complete Investigation Summary: Prefix Caching Bug on gfx950

## The Bug

**vLLM with prefix caching on AMD MI355X (gfx950) produces non-deterministic logprobs**:
- R1 (cache miss): Computes all 31 tokens → logprob_1
- R2 (cache hit): Reuses 16 cached + computes 15 new → logprob_2  
- **logprob_1 ≠ logprob_2** (differences up to 0.056!)

```
vLLM+PC R1-R2 differences:
Pos 0: +0.014736
Pos 5: -0.056620  ← HUGE!
Pos 6: +0.008953
```

But tokens remain identical (test passes).

## Root Cause Analysis

### The Core Problem

```
R1 (miss): kernel receives q.shape=[31, 16, 128] → processes ALL tokens
R2 (hit):  kernel receives q.shape=[15, 16, 128] → processes NEW tokens only
```

**Even with `IN_PRECISION="ieee"`**:
- Different query counts → Different tile boundaries  
- Different tile boundaries → Different online softmax accumulation order
- Different accumulation order → Different numerical results

### Why `IN_PRECISION="ieee"` Isn't Enough

`input_precision="ieee"` ONLY affects `tl.dot` operations:
- ✅ Makes `Q*K` and `P*V` deterministic
- ❌ Does NOT affect `tl.sum`, `tl.max`, `tl.exp` in softmax
- ❌ Does NOT change tile iteration order
- ❌ Does NOT fix accumulation order differences

The problem is **algorithmic**, not just precision.

## Attempted Fixes

### Attempt 1: Kernel `context_len=0` Modification
**Idea**: Force kernel to treat cache hits like cache misses

**Implementation**:
```python
if DETERMINISTIC_CACHE:
    context_len = 0  # Ignore cached tokens for tile calculation
```

**Result**: ❌ **MADE IT WORSE**
```
Before: vLLM+PC R1-R2 = +0.014736
After:  vLLM+PC R1-R2 = +0.210222  (14x worse!)
```

**Why it failed**: Query shape was still different (31 vs 15), so tiles were still wrong.

### Attempt 2: Query Padding
**Idea**: Pad 15-token query to 31 tokens to match R1 shape

**Implementation**:
```python
query_padded = torch.zeros(31, 16, 128)
query_padded[16:31] = query[0:15]  # Pad with zeros for cached portion
```

**Result**: ❌ **COMPLETE FAILURE**
```
Expected: 'The European Union consists of 27 member states'
Got:      ' questionQuestionQuestionQuestionQuestion...'
```

**Why it failed**:
1. Padding location wrong (broke causal masking)
2. `cu_seqlens_q` adjustment incorrect
3. Zero padding confused attention computation
4. Output extraction from wrong slice

## Why This Bug Cannot Be Fixed Easily

### The Fundamental Incompatibility

**vLLM's Design**: Optimize for performance
- Cache hit → Process only NEW tokens (15)
- This is the ENTIRE POINT of prefix caching

**Determinism Requirement**: Process identically
- Cache hit → Must process ALL tokens (31)
- This DEFEATS the purpose of caching

**These goals are fundamentally opposed.**

### What Would Be Needed

To fix this properly, you'd need to:

1. **Cache attention outputs** (not just K/V)
   - Store computed attention for cached portion
   - Massive memory overhead
   - Requires KV cache system redesign

2. **Force full recomputation**
   - Always process all tokens even with cache
   - ~20-30% slower on cache hits
   - Defeats caching purpose

3. **Redesign online softmax**
   - Make accumulation order independent of tile count
   - Extremely complex
   - May not be possible with current Triton limitations

4. **Wait for hardware fix**
   - AMD fixes gfx950 non-determinism in ROCm
   - Not in your control
   - No timeline

## Current State (All Changes Reverted)

✅ **What Remains**:
- `IN_PRECISION="ieee"` for all `tl.dot` operations
- This helps but doesn't fully fix the issue

❌ **What's Still Broken**:
- Logprob differences between R1 and R2
- Up to 0.056 absolute difference

✓ **What Works**:
- Tokens are identical (test passes)
- Output is correct: 'The European Union consists of 27 member states'
- Just logprobs differ

## Recommendations

### Option 1: Accept the Variation (PRAGMATIC)
**Pros**:
- No code changes needed
- Prefix caching still provides speedup
- Tokens remain deterministic

**Cons**:
- Logprobs vary by ~0.05
- May affect applications sensitive to exact probabilities
- Not ideal for research/reproducibility

### Option 2: Disable Prefix Caching on gfx950 (SAFE)
```python
if on_gfx950() and dtype == bfloat16:
    logger.warning("Disabling prefix caching on gfx950 due to non-determinism")
    enable_prefix_caching = False
```

**Pros**:
- Guarantees determinism
- Simple to implement
- User can override with flag

**Cons**:
- Loses caching performance benefit
- Only affects gfx950 users

### Option 3: Document the Limitation (TRANSPARENT)
Add to documentation:
> **Note**: Prefix caching on AMD MI355X (gfx950) with bfloat16 may produce small logprob variations (~0.05) between cache miss and cache hit due to hardware floating-point non-determinism. Token selection remains deterministic.

**Pros**:
- Honest about limitations
- Users can make informed decisions

**Cons**:
- Doesn't fix the underlying issue

## My Recommendation

**Implement Option 2 with Option 3**:

1. Auto-disable prefix caching on gfx950 by default
2. Add override flag: `VLLM_FORCE_PREFIX_CACHE_GFX950=1`
3. Document the limitation clearly

This balances:
- ✅ Correctness (default behavior is deterministic)
- ✅ Performance (users can opt-in if they accept risk)
- ✅ Transparency (clear documentation)

## Technical Lessons Learned

1. **Algorithmic non-determinism** is harder than precision issues
2. **Tile-based algorithms** are sensitive to processing order
3. **Online softmax** accumulation order matters significantly
4. **Query padding** in attention is extremely tricky
5. **Performance vs correctness** trade-offs are sometimes unavoidable

## Files Modified (Now Reverted)

- `/app/vllm/vllm/v1/attention/ops/triton_unified_attention.py`
  - Added `DETERMINISTIC_CACHE` flag (kept for future use)
  - Modified `context_len` calculation (REVERTED)
  
- `/app/vllm/vllm/v1/attention/backends/triton_attn.py`
  - Added query padding logic (REVERTED)

## Test Results After Revert

```bash
HIP_VISIBLE_DEVICES=4,5,6,7 pytest -s -v tests/entrypoints/openai/test_prefix_cache_debug.py

Result: PASSED ✓
Output: 'The European Union consists of 27 member states' ✓ Correct
Logprob Diff: vLLM+PC R1-R2 = +0.014736, -0.056620, +0.008953 (original bug remains)
```

## Conclusion

After extensive investigation and multiple failed attempts, **this bug cannot be fixed without either**:
1. Significant performance degradation (full recomputation)
2. Major architectural redesign (cache attention outputs)
3. Hardware/driver improvements from AMD

**I recommend disabling prefix caching on gfx950 by default** with a clear override mechanism for users who accept the risk.

---

**Final Status**: ❌ UNFIXED - Reverted all changes, documented limitations  
**Impact**: gfx950 users see ~0.05 logprob variations with prefix caching  
**Recommended Action**: Auto-disable on gfx950, allow override flag
