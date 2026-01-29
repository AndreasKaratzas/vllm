# AITER Debug Analysis

## Summary of Findings

### Cache Logs Analysis

**Run 1** (Cold cache):
```
[CACHE_GET] Cache MISS for hash=b"\xc4\x9f}\x12..."
[CACHE_MGR] Cache MISS
[CACHE_PUT] Cached block: id=1, hash=...
[CACHE_PUT] Cached block: id=2, hash=...
```
- No cached blocks available
- Computes everything fresh
- Caches blocks 1 and 2
- **Result**: Produces token 320 (' (') at position 3

**Run 2** (Warm cache):
```
[CACHE_GET] Single block: id=1, hash=b"\xc4\x9f}\x12..."
[CACHE_MGR] Cache HIT: 1 blocks
  Block 0: id=1
[CACHE_PUT] Cached block: id=4, hash=...
```
- Reuses cached block 1 from Run 1
- Only caches new block 4
- **Result**: Produces token 17167 (' consists') at position 3

**Run 3** (Warm cache):
```
[CACHE_GET] Single block: id=1, hash=b"\xc4\x9f}\x12..."
[CACHE_MGR] Cache HIT: 1 blocks
  Block 0: id=1
[CACHE_PUT] Cached block: id=6, hash=...
```
- Reuses cached block 1 from Run 1
- Only caches new block 6
- **Result**: Produces token 17167 (' consists') at position 3

## Key Insights

### ✅ What's Working

1. **Block selection is deterministic**: All runs consistently select block id=1 when it's available
2. **Runs 2-3 are deterministic**: Both produce identical outputs (token 17167 at position 3)
3. **Cache hit/miss detection works correctly**: Proper detection of available cached blocks

### ❌ What's Broken

1. **Run 1 produces different output**: Token 320 instead of 17167 at position 3
2. **Different logprobs from position 0**: Even before divergence, logprobs differ slightly:
   - Position 0: Run 1 = -0.13004428 vs Runs 2-3 = -0.11972452 (diff = 0.01031976)
   - Position 1: Run 1 = -0.04078946 vs Runs 2-3 = -0.04075523 (diff = 0.00003422)
   - Position 2: Run 1 = -0.00665673 vs Runs 2-3 = -0.00587629 (diff = 0.00078044)
3. **Divergence amplifies**: Small initial differences compound by position 3

## Root Cause Hypothesis

### Theory 1: Run 1 Computation is Wrong
Run 1 might be computing incorrect activations, which then get cached in block 1. When Runs 2-3 reuse block 1, they inherit these "wrong" values and produce consistent but incorrect outputs compared to what Run 1 should have produced.

**Evidence**:
- Runs 2-3 are identical to each other (deterministic)
- But both differ from Run 1
- All reuse the same block 1 from Run 1

### Theory 2: AITER Backend Has BF16 Precision Issue
Similar to the TRITON backend, AITER may have precision issues when converting FP32 → BF16 during attention computation.

**Evidence**:
- TRITON backend had `P.to(V.dtype)` causing non-determinism
- Fixed by keeping P in FP32 and upcasting V
- AITER backend uses external `aiter.ops.triton.unified_attention`
- Likely has similar precision conversion

**Counter-evidence**:
- If AITER had random BF16 issues, Runs 2-3 should also differ from each other
- But they don't - they're perfectly deterministic!

### Theory 3: First Run Initialization Issue
Something about the first run (cold cache) causes different computation paths or initialization.

**Possibilities**:
- GPU state not properly initialized on first run
- Cache warmup effects
- Different code paths for cache miss vs hit

## What's Different About Run 1?

Run 1 is the ONLY run that:
- Has all cache misses
- Allocates blocks 1 and 2 fresh
- Computes attention without any cached KV blocks

Runs 2-3 both:
- Reuse cached block 1
- Only allocate one new block each
- Compute attention with mix of cached + fresh KV

## Hypothesis: Cache Miss Path Has Bug

The issue might be in how the attention is computed when there are NO cached blocks (Run 1) vs when there ARE cached blocks (Runs 2-3).

**Potential issues**:
1. Different attention kernel called for cache miss vs hit
2. Different tensor initialization
3. GPU synchronization issues on first run

## Next Steps

1. **Check if TRITON fix also needs to be applied to AITER**
   - AITER backend uses external library
   - Can't modify directly
   - But could try forcing AITER to use FP32 accumulation

2. **Add more debug logging**
   - Print Q, K, V tensor shapes and dtypes in AITER backend
   - Track if different code paths are taken for Run 1 vs Runs 2-3

3. **Test Theory**: Run without prefix caching
   - If all 3 runs are identical without prefix caching, confirms issue is in cache logic
   - If they still differ, issue is deeper in AITER backend

4. **Workaround**: Force Run 1 to behave like Runs 2-3
   - Pre-warm cache before first real request
   - Or: discard Run 1's results as "warmup"

## Conclusion

The good news: **Cache block selection is now deterministic** (Runs 2-3 identical).

The bad news: **Run 1 computes different values** than Runs 2-3, and these wrong values get cached and reused.

The mystery: Why does Run 1 differ when it should be computing the same as Runs 2-3 would without any cache?
