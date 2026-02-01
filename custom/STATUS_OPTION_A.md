# Option A Implementation Status

## What I Tried

Implemented request splitting at the `AsyncLLM._add_request()` level:
- Detect first-time requests using a tracker based on prompt prefix hash
- Split first request into:
  1. Prefix warmup request (tokens [0:16])
  2. Full request (tokens [0:31]) using cached prefix

## Issues Encountered

### Problem 1: Response Flow Corruption
When splitting at the API layer, the HTTP client sends ONE request but my implementation creates TWO internal requests (prefix + actual). This breaks the request/response mapping:
- Client expects ONE response
- System generates TWO responses (from prefix warmup + actual)
- Test fails with `KeyError: 'choices'` because response format is wrong

### Problem 2: Async Queue Confusion
Using `async for output in prefix_queue` to wait for warmup completion causes:
- Blocking the main async flow
- Potential deadlocks
- Confusion in output routing

### Root Cause
**The HTTP/API layer is the wrong place to implement this**. One HTTP request MUST map to one response. Trying to turn it into two internal requests at this level fundamentally breaks the async request/response contract.

## Lessons Learned

1. **API layer is too high**: Can't split requests here without breaking HTTP semantics
2. **Need lower-level implementation**: Should be at scheduler or KV cache manager level
3. **Transparency is key**: The split must be invisible to the API layer

## Next Steps

### Option A-Revised: Scheduler-Level Implementation
Instead of splitting at API level, implement at scheduler:
- Scheduler detects tagged requests (`_deterministic_split_point`)
- Automatically schedules prefix computation first
- Then schedules remainder with cached prefix
- All transparent to API layer

### Alternative: Simpler Approach
Just document the limitation and provide a workaround:
- Users can manually send prefix-only request first
- Then send full request
- Or disable prefix caching on gfx950

## Current Code State

Modified files:
- `/app/vllm/vllm/v1/engine/async_llm.py` - Added deterministic cache tracker and request tagging
- Currently tags requests but doesn't split them (to avoid breaking responses)

## Recommendation

Implement scheduler-level handling OR revert to simpler "disable on gfx950" approach (Option C).
