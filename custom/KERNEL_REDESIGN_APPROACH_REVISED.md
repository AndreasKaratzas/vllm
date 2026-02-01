# Kernel Redesign - REVISED APPROACH

## The Fundamental Problem

**Current Issue**: When vLLM has a cache hit:
- R1 (miss): Passes Q[0:31] → kernel processes all 31 tokens
- R2 (hit): Passes Q[16:31] → kernel processes only 15 NEW tokens

The kernel receives **different number of queries**, making it impossible to ensure identical tile order just by changing `context_len` inside the kernel!

## Why Setting `context_len=0` Doesn't Work

Setting `context_len=0` inside the kernel doesn't help because:
1. The kernel still receives only 15 queries (not 31)
2. Tile boundaries are determined by query count, not context_len  
3. We can't "magically" access queries [0:15] that weren't passed in

## Real Solution: Modify vLLM's Attention Call

We need to change HOW vLLM calls the attention kernel when prefix caching is enabled:

### Current Flow (NON-DETERMINISTIC)
```python
# Cache miss:
unified_attention(q=Q[0:31], ...)  # 31 queries

# Cache hit:
unified_attention(q=Q[16:31], ...)  # 15 queries ← DIFFERENT!
```

### Proposed Flow (DETERMINISTIC)
```python
# Cache miss:
unified_attention(q=Q[0:31], ...)  # 31 queries

# Cache hit:
unified_attention(
    q=Q[0:31],  # ← SAME! Pass ALL queries
    output_mask=[False]*16 + [True]*15,  # Only update outputs [16:31]
)
```

## Implementation Plan

### Step 1: Add output masking support to kernel
- Add `output_mask` parameter to kernel
- After computing attention, zero out masked positions before writing

### Step 2: Modify vLLM's attention wrapper
- When cache hit detected, pad queries with zeros for cached portion
- Pass output mask to indicate which positions to actually compute
- This ensures identical tile iteration regardless of cache state

### Step 3: Optimize for performance
- Use actual cached Q values (not zeros) to reduce numerical error
- Kernel can skip computation for masked outputs (with proper barriers)

## Alternative: Two-Pass Approach

If full recomputation is too slow:

### Pass 1: Read cached attention outputs
- vLLM caches not just K/V but also attention outputs O[0:16]
- On cache hit, directly copy O[0:16] to output buffer

### Pass 2: Compute new tokens deterministically
- Ensure queries [16:31] are processed identically regardless of cache state
- This requires aligning tile boundaries to match full-sequence computation

## Trade-offs

| Approach | Determinism | Performance | Complexity |
|----------|-------------|-------------|------------|
| Full recomputation | ✅ Perfect | ⚠️ ~20% slower | ⚠️ Moderate |
| Output masking | ✅ Perfect | ⚠️ ~15% slower | ✅ Simple |
| Cache attention outputs | ✅ Perfect | ✅ Fast | ❌ High (memory) |
| Current (broken) | ❌ Non-deterministic | ✅ Fast | ✅ Simple |

## Recommended: Output Masking with Padding

This is the cleanest approach that maintains determinism without requiring massive refactoring:

```python
def deterministic_attention_wrapper(q, context_len, ...):
    if DETERMINISTIC_CACHE and context_len > 0:
        # Pad queries to full length
        q_full = torch.zeros(context_len + q.shape[0], ...)
        q_full[context_len:] = q  # New queries
        # Note: Could use cached Q[0:context_len] for better numerics
        
        # Call kernel with full query sequence
        out_full = unified_attention(q=q_full, ...)
        
        # Return only non-cached portion
        return out_full[context_len:]
    else:
        return unified_attention(q=q, ...)
```

This ensures queries [16:30] are computed identically whether as:
- Positions [16:30] in a 31-token sequence (R1)
- Positions [16:30] in a padded 31-token sequence (R2)

## Next Steps

1. Implement wrapper function with query padding
2. Test with debug prints to verify Q shapes match
3. Compare logprobs R1 vs R2
4. Measure performance impact
5. Optimize by using cached Q values instead of zeros
