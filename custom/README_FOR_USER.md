# Investigation Complete

## Summary

I investigated the prefix caching non-determinism bug on AMD MI355X (gfx950) and attempted multiple fixes. **Unfortunately, none of them worked.**

## What I Found

The bug is caused by **different query shapes** being passed to the attention kernel:
- **Cache miss** (R1): Processes 31 tokens
- **Cache hit** (R2): Processes 15 new tokens (16 cached)

This leads to:
- Different tile boundaries
- Different accumulation orders  
- Different logprob results (up to 0.056 difference)

## What I Tried

1. **`context_len=0` in kernel** → Made it 14x WORSE
2. **Query padding to match shapes** → Completely BROKE output (garbage tokens)

Both approaches failed because they're fundamentally incompatible with vLLM's architecture.

## Why It Can't Be Fixed Easily

To get identical results, you need:
- **Same query shape** (31 vs 31, not 31 vs 15)
- **Same tile processing order**
- **Same accumulation order**

But this means **always processing all tokens**, which defeats the entire purpose of prefix caching (performance).

## Current Status

**All my changes have been REVERTED**. The system is back to baseline:
- ✅ Tokens are identical (test passes)
- ✅ Output is correct
- ❌ Logprobs vary by ~0.05 between cache miss/hit

## My Recommendation

**Auto-disable prefix caching on gfx950** with an override flag:

```python
if on_gfx950() and dtype == bfloat16:
    if not os.getenv("VLLM_FORCE_PREFIX_CACHE_GFX950"):
        logger.warning("Disabling prefix caching on gfx950 due to non-determinism")
        enable_prefix_caching = False
```

This:
- ✅ Guarantees correctness by default
- ✅ Allows users to opt-in if they accept the risk
- ✅ Only affects gfx950 (not other GPUs)

## Documentation Created

For full details, see:
- `/app/vllm/COMPLETE_INVESTIGATION_SUMMARY.md` - Technical analysis
- `/app/vllm/FINAL_STATUS.md` - Test results and attempts
- `/app/vllm/BUG_ROOT_CAUSE_FINAL.md` - Root cause explanation

## Next Steps

If you want to pursue this further, you would need to:
1. Redesign the KV cache system to cache attention outputs (major refactoring)
2. OR accept the ~0.05 logprob variations as acceptable
3. OR disable prefix caching on gfx950

I'm sorry I couldn't deliver a working fix. The problem is deeper than what can be solved with kernel modifications alone.
