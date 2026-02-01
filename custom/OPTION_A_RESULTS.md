# Option A Implementation: Results and Findings

## Summary

Attempted to implement deterministic prefix caching via request splitting at the API layer. **Implementation encountered fundamental architectural issues** that prevent it from working at this level.

## What Was Implemented

### Files Modified
1. `/app/vllm/vllm/v1/engine/async_llm.py`
   - Added `DeterministicPrefixCacheTracker` class
   - Tracks first-time prompt prefixes
   - Tags requests for potential splitting

2. `/app/vllm/vllm/v1/core/deterministic_prefix_cache.py`
   - Helper module for deterministic caching logic (created but not fully integrated)

### Approach Attempted
```python
# For first-time request with tokens [0-30]:
# Pass 1: Send prefix [0-15] as warmup request
await send_prefix_warmup_request(tokens[:16])
await wait_for_completion()

# Pass 2: Send full request [0-30] (uses cached prefix)
await send_actual_request(tokens[:31])
return actual_response  # Return this to client
```

## Why It Doesn't Work

### Core Problem: HTTP Semantics Violation

**One HTTP Request MUST Equal One HTTP Response**

When client sends:
```
POST /v1/completions HTTP/1.1
{ "prompt": "...", "max_tokens": 10 }
```

The client expects exactly ONE response. But our implementation creates:
1. Prefix warmup request (internal)
2. Actual request (what client wants)

This breaks the request/response mapping:
- Output processor gets confused about which response to return
- Async queues don't know which output belongs to which "real" request
- Client receives malformed or missing response (`KeyError: 'choices'`)

### Technical Issues Encountered

1. **Async Flow Corruption**
   ```python
   # This blocks and breaks the async flow:
   async for output in prefix_queue:
       if output.finished:
           break
   ```

2. **Queue Management**
   - Creating temporary queues for warmup requests
   - Waiting for them to complete
   - Then processing actual request
   - **All of this happens INSIDE a single `_add_request()` call**
   - But the caller expects immediate return, not blocking

3. **Response Routing**
   - System doesn't know which response to send to HTTP client
   - Prefix warmup response? Actual response? Both?
   - Output processor wasn't designed for this use case

## Test Results

```bash
$ pytest test_prefix_cache_debug.py::test_side_by_side

Running 5 requests...
F  # Failed on FIRST request

KeyError: 'choices'  # Response malformed
```

Only 1 request ran instead of 5, and it returned invalid response.

## Why This Level Is Wrong

The **API/HTTP layer** is responsible for:
- Receiving HTTP requests
- Routing to engine
- Returning HTTP responses
- **1:1 mapping is fundamental**

Request splitting should happen at:
- **Scheduler level**: Can batch/split internally, returns ONE final result
- **KV Cache Manager level**: Can compute in chunks, invisible to upper layers

NOT at:
- ❌ API layer: Breaks HTTP semantics
- ❌ Request handler: Expected to be 1:1

## Correct Implementation Levels

### Option 1: Scheduler-Level (Complex but Proper)
```
HTTP Request → AsyncLLM → Scheduler 
                            ↓
                    [Detects first-time prefix]
                            ↓
                    Schedule: prefix batch → wait → suffix batch
                            ↓
                    Return combined result
                            ↑
HTTP Response ← AsyncLLM ←  ┘
```

**Pros**: Transparent to API, proper architecture
**Cons**: Complex, touches critical scheduler code (~200+ lines)

### Option 2: KV Cache Manager (Simpler)
```
Request processing → Prefill → KV Cache Manager
                                        ↓
                        [First time? Split computation]
                                        ↓
                        Compute prefix → Cache → Compute suffix
                                        ↓
                        Return logprobs to prefill
```

**Pros**: Localized changes, doesn't affect scheduling
**Cons**: Still somewhat complex, affects attention flow

### Option 3: Disable on gfx950 (Simplest)
```python
# In rocm.py
if is_gfx950() and dtype == bfloat16:
    if not os.getenv("VLLM_FORCE_PREFIX_CACHE"):
        logger.warning("Prefix caching disabled on gfx950 due to non-determinism")
        enable_prefix_caching = False
```

**Pros**: 10 lines, guaranteed correct, user can override
**Cons**: Loses performance benefit of caching

### Option 4: Manual Workaround (Documentation)
Document for users:
```python
# For deterministic results on gfx950, manually warm cache:
# Step 1: Cache the prefix
client.completions.create(
    prompt=tokens[:16],
    max_tokens=1  # Just to cache
)

# Step 2: Full request (uses cache)
result = client.completions.create(
    prompt=tokens,  # Full prompt
    max_tokens=10
)
```

**Pros**: No code changes, works today
**Cons**: User burden, not automatic

## Recommended Path Forward

### Immediate (Today)
Implement **Option 3** (disable on gfx950):
- Simple, safe, correct
- 15 minutes to implement
- Users can override if they want

### Future (Proper Fix)
Implement **Option 1** (scheduler-level) or **Option 2** (KV cache level):
- Requires careful design
- Needs extensive testing
- 1-2 weeks of work

### Documentation
Add **Option 4** to docs regardless:
- Helps users who need determinism NOW
- Works on any hardware
- Good debugging technique

## Current Code State

The `/app/vllm/vllm/v1/engine/async_llm.py` file now has:
- ✅ `DeterministicPrefixCacheTracker` class (working)
- ✅ Request tagging logic (working)
- ❌ Request splitting disabled (causes failures)

**Code is safe to commit** - it adds infrastructure but doesn't break anything.

## Key Insight

> **The non-determinism is ONLY between R1 (cache miss) and R2+ (cache hits). R2, R3, R4, R5 are perfectly deterministic with each other.**

This means:
- The caching mechanism itself works correctly
- The problem is specifically the different computation paths for first request
- Once cache is warm, everything is deterministic

Therefore:
- Warming the cache (manually or automatically) solves the problem
- We just need a clean way to do it

## Conclusion

Option A (request splitting at API layer) is **architecturally incompatible** with vLLM's design.

**Recommended action**: Implement Option 3 (disable on gfx950) as immediate fix, then plan Option 1 or 2 as proper long-term solution.

The investigation was valuable - we now understand:
1. Why the bug occurs (different tile processing for R1 vs R2+)
2. Where it occurs (attention kernel accumulation)
3. What works (R2-R5 are identical)
4. What doesn't work (API-level splitting)
5. What will work (scheduler or cache-level splitting)

**Status**: Option A attempted and understood. Ready to pivot to Option 3 or continue to scheduler-level implementation.
