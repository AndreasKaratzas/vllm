# Deterministic Cache Proposal

## Analysis of R1, R2, R3 Differences

### Cache Hit Pattern
From the logs:
- **R1**: `cached_tokens=0` → MISS (computes all 31 tokens)
- **R2**: `cached_tokens=16` → HIT (reuses 16, computes 15)  
- **R3**: `cached_tokens=16` → HIT (same as R2)
- **R4**: `cached_tokens=16` → HIT (same as R2)
- **R5**: `cached_tokens=16` → HIT (same as R2)

### The Key Observation

**R2, R3, R4, R5 are IDENTICAL to each other** (all cache hits)
- Same cached_tokens=16
- Same computation pattern
- **Logprobs are identical across R2-R5** ✓

**But R1 is DIFFERENT from R2-R5**:
- R1: cached_tokens=0 (miss)
- R2: cached_tokens=16 (hit)
- **This is where the non-determinism occurs**

## Insight: The Problem is R1 vs R2, NOT R2 vs R3!

The bug is:
```
First request (R1, miss):  Different logprobs
Second request (R2, hit):  Different logprobs
Third+ requests (R3-R5):   IDENTICAL to R2 ✓
```

**So once we get a cache hit, everything becomes deterministic!**

The non-determinism is **ONLY** between:
1. First computation (no cache)
2. Subsequent computations (with cache)

## Proposed Solution: Deterministic Cache Mode

### Concept: "Warm-up" the Cache Deterministically

Instead of trying to make R1 == R2, we can:
1. **Force R1 to also use cache** (even though it's the first request)
2. Pre-populate cache with zeros or a canonical value
3. This makes R1 behave like R2 from the start

### Implementation

```python
class DeterministicPrefixCache:
    """
    Ensures first request behavior matches subsequent requests
    by pre-populating cache with canonical values.
    """
    
    def __init__(self):
        self.canonical_cache = {}  # token_ids -> pre-computed K/V
    
    def get_or_create_cache(self, token_ids, block_size):
        """
        For first request: Create cache with CANONICAL computation
        For subsequent: Use existing cache
        
        This ensures R1 computes the same way as R2+
        """
        key = tuple(token_ids)
        
        if key not in self.canonical_cache:
            # FIRST REQUEST: Compute in TWO passes
            # Pass 1: Compute prefix (tokens 0-15) ALONE
            prefix_kv = self._compute_prefix(token_ids[:16])
            
            # Pass 2: Compute suffix (tokens 16-30) WITH prefix cached
            full_output = self._compute_with_cache(
                token_ids, 
                cached_kv=prefix_kv,
                cache_len=16
            )
            
            # Store for future use
            self.canonical_cache[key] = prefix_kv
            
            return full_output
        else:
            # SUBSEQUENT REQUESTS: Use cached prefix
            return self._compute_with_cache(
                token_ids,
                cached_kv=self.canonical_cache[key],
                cache_len=16
            )
```

### Key Idea

**Make R1 compute in TWO passes**:
1. **Pass 1**: Compute tokens 0-15 as if they were a separate request
2. **Cache it**
3. **Pass 2**: Compute tokens 16-30 WITH cached 0-15

This makes R1 computationally identical to R2!

## Detailed Algorithm

### Traditional (Non-deterministic)
```
R1: Process tokens [0-30] all at once → logprob_A
R2: Process tokens [16-30] with [0-15] cached → logprob_B
    logprob_A ≠ logprob_B ❌
```

### Deterministic Mode
```
R1: 
  Step 1: Process tokens [0-15] alone → cache them
  Step 2: Process tokens [16-30] with [0-15] cached → logprob_X
  
R2:
  Step 1: Load cached tokens [0-15]
  Step 2: Process tokens [16-30] with [0-15] cached → logprob_Y
  
  logprob_X == logprob_Y ✓
```

## Implementation Strategy

### Option A: Modify Request Splitting

```python
def process_request_deterministic(tokens, enable_prefix_cache):
    if not enable_prefix_cache:
        return process_normal(tokens)
    
    # Always split into prefix + suffix
    prefix_len = find_cache_boundary(tokens)  # e.g., 16
    
    if prefix_len == 0:
        # No natural split point, process normally
        return process_normal(tokens)
    
    # ALWAYS do two-pass computation
    # Pass 1: Process prefix
    prefix_output = process_tokens(
        tokens[:prefix_len],
        cache=None
    )
    
    # Cache the K/V from prefix
    cache_kv(tokens[:prefix_len], prefix_output.kv)
    
    # Pass 2: Process suffix with cached prefix
    suffix_output = process_tokens(
        tokens[prefix_len:],
        cache=get_cached_kv(tokens[:prefix_len])
    )
    
    return merge_outputs(prefix_output, suffix_output)
```

### Option B: Pre-warm Cache

```python
def enable_deterministic_prefix_cache(vllm_instance):
    """
    Pre-populate cache with canonical computations
    for common prefixes.
    """
    # Detect common prompt patterns
    common_prefixes = detect_common_prefixes(
        vllm_instance.recent_requests
    )
    
    for prefix_tokens in common_prefixes:
        if len(prefix_tokens) >= block_size:
            # Compute and cache prefix
            _ = vllm_instance.compute(
                prefix_tokens,
                cache_only=True  # Don't return output
            )
```

## Advantages

✅ **Truly deterministic**: R1 == R2 == R3
✅ **No query padding needed**: Each pass is natural
✅ **Works with existing architecture**: Just splits requests differently
✅ **Maintains performance**: After first request, it's the same speed

## Disadvantages

⚠️ **First request ~2x slower**: Requires two passes
⚠️ **More complex**: Need to manage request splitting
⚠️ **Memory**: Briefly holds both prefix and suffix outputs

## Configuration

```python
class VllmConfig:
    # New option
    deterministic_prefix_cache: bool = False
    deterministic_cache_block_size: int = 16  # Split point
```

Usage:
```bash
vllm serve model \
    --enable-prefix-caching \
    --deterministic-prefix-cache \
    --deterministic-cache-block-size 16
```

## Expected Results

With deterministic mode enabled:
```
vLLM+PC R1-R2: ✓ +0.000000 (all positions)
vLLM+PC R2-R3: ✓ +0.000000 (all positions)
```

## Performance Impact

| Scenario | Traditional | Deterministic | Difference |
|----------|-------------|---------------|------------|
| R1 (first) | 100% (fast) | ~180% (2 passes) | -80% slower |
| R2 (cache hit) | 60% (fast) | 60% (fast) | Same |
| R3+ (cache hit) | 60% (fast) | 60% (fast) | Same |

**Amortized**: If you do 10 requests with same prefix:
- Traditional: 100% + 9×60% = 640%
- Deterministic: 180% + 9×60% = 720%
- **Only ~12% slower overall**

## Implementation Plan

1. **Phase 1**: Add request splitting logic
   - Split requests at cache block boundaries
   - Handle prefix-only and suffix-only passes

2. **Phase 2**: Modify cache manager
   - Track "canonical" first computations
   - Ensure subsequent requests use same cache

3. **Phase 3**: Test and benchmark
   - Verify logprobs are identical
   - Measure performance impact
   - Compare with traditional mode

4. **Phase 4**: Add configuration option
   - Make it opt-in
   - Document trade-offs

## Next Steps

Would you like me to implement this **deterministic cache** approach? 

The key insight is:
- **Stop trying to make R1 fast** (single-pass)
- **Make R1 match R2** (two-pass with caching)
- **R2+ stay fast** (cache hits)

This trades first-request performance for determinism, which seems like a reasonable trade-off for users who need reproducibility.
