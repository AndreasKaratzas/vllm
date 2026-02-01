# Kernel Redesign Plan: Ensure Identical Tile Order

## Problem Statement

Current behavior:
- **R1 (cache miss)**: Kernel processes 31 queries, tiles iterate 0→31
- **R2 (cache hit)**: Kernel processes 15 queries, tiles iterate 0→31 but with context_len=16

Even with identical K/V values and `IN_PRECISION="ieee"`, the tile processing differs because:
1. Query positions are offset by context_len
2. Tile boundaries align differently  
3. Online softmax accumulation order differs

## Root Cause in Kernel

```python
# Line 171 in triton_unified_attention.py
context_len = seq_len - cur_batch_query_len

# For R1: context_len = 31 - 31 = 0
# For R2: context_len = 31 - 15 = 16
```

This `context_len` offset affects:
- Tile iteration start/end
- Query absolute positions  
- Causal masking calculations
- Tile accumulation order

## Solution Approach

### Option 1: Force Full Recomputation (SIMPLE but SLOW)
Always pass the full query sequence, mark cached outputs as "discard".

**Pros**: Guarantees identical computation  
**Cons**: Defeats caching purpose, no speedup

### Option 2: Deterministic Tile Ordering (COMPLEX but EFFICIENT)
Redesign kernel to process tiles in a deterministic order regardless of cache state.

**Key idea**: Make tile iteration and accumulation independent of `context_len`.

### Option 3: Two-Pass Approach (HYBRID)
1. **Pass 1**: Load cached attention outputs (if any)
2. **Pass 2**: Compute NEW tokens with identical tile order

**Pros**: Preserves speedup, ensures determinism  
**Cons**: Requires caching attention outputs, not just K/V

## Recommended Approach: Option 2 (Deterministic Tile Ordering)

### Changes Required

1. **Normalize query positions** to always start from 0
2. **Adjust tile iteration** to use absolute sequence positions
3. **Ensure accumulation order** is independent of context_len

### Implementation Plan

```python
# Current code (line 171):
context_len = seq_len - cur_batch_query_len

# NEW: Add a "deterministic mode" flag for prefix caching
if DETERMINISTIC_CACHE_MODE:
    # Force tile iteration to match full sequence computation
    # by treating all tiles the same regardless of context_len
    virtual_context_len = 0  # Pretend no cache for tile iteration
else:
    virtual_context_len = context_len  # Normal behavior
```

### Detailed Changes

#### 1. Add deterministic flag to kernel signature
```python
@triton.jit
def attention_decode_kernel(
    # ... existing params ...
    DETERMINISTIC_CACHE: tl.constexpr,  # NEW
):
```

#### 2. Modify tile iteration to be cache-independent
```python
if DETERMINISTIC_CACHE:
    # Process tiles as if full sequence (0 to seq_len)
    # Mask out already-cached query positions in output
    effective_context_len = 0
else:
    effective_context_len = context_len
```

#### 3. Add output masking for cached positions
```python
# After computing attention output
if DETERMINISTIC_CACHE:
    # Zero out outputs for cached positions
    is_cached = query_pos < context_len
    acc = tl.where(is_cached[:, None], 0.0, acc)
```

## Performance Impact

- **Cache miss (R1)**: No change
- **Cache hit (R2)**: ~20-30% slower (processes all queries but discards cached)
- **Correctness**: 100% deterministic ✅

## Trade-off Analysis

| Metric | Current | Option 1 | Option 2 | Option 3 |
|--------|---------|----------|----------|----------|
| Determinism | ❌ | ✅ | ✅ | ✅ |
| Speedup on cache hit | ✅ | ❌ | ⚠️ | ✅ |
| Implementation complexity | Simple | Trivial | Moderate | High |
| Memory overhead | Low | Low | Low | High |

## Next Steps

1. Implement Option 2 with DETERMINISTIC_CACHE flag
2. Test with debug prints to verify tile order
3. Measure performance impact
4. Compare logprobs R1 vs R2
5. If determinism achieved, consider optimizations

## Alternative: Hybrid Approach

Only use deterministic mode for **prefill** (where sequences are short):
- Prefill: Use deterministic mode (full recomputation)
- Decode: Use normal mode (single token, no cache hit issues)

This minimizes performance impact while ensuring determinism.
