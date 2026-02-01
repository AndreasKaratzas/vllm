# Deterministic Prefix Cache: Implementation Guide

## Goal

Make R1 (cache miss) produce identical logprobs to R2+ (cache hits) by forcing R1 to compute in two passes that match R2's pattern.

## The Key Insight

From log analysis:
- R1: `cached_tokens=0` (miss) → Different logprobs
- R2: `cached_tokens=16` (hit) → Baseline logprobs  
- R3-R5: `cached_tokens=16` (hit) → **IDENTICAL to R2** ✅

**The non-determinism is ONLY between R1 and R2+**. R2-R5 are already fully deterministic!

## Why This Approach Works

Current R1 computation:
```
Input: tokens [0-30] (31 total)
Kernel: q.shape=[31, 16, 128] → Single pass, all at once
Tiles: [0-15], [16-31] → Accumulated in order
Result: logprobs_A
```

Current R2+ computation:
```
Input: tokens [16-30] (15 new, 16 cached)
Kernel: q.shape=[15, 16, 128] → Single pass, new tokens only
Tiles: [0-15] (from cache), then [16-31] (computed)
Result: logprobs_B

logprobs_A ≠ logprobs_B ❌
```

**Proposed R1 computation (deterministic mode)**:
```
Pass 1: tokens [0-15] alone
  - q.shape=[16, 16, 128]
  - Compute and cache

Pass 2: tokens [16-30] with [0-15] cached  
  - q.shape=[15, 16, 128] ← SAME as R2!
  - Same tile structure as R2
  - Same accumulation order as R2
  
Result: logprobs_X == logprobs_B ✅
```

## Implementation Steps

### Step 1: Add Deterministic Mode Detection

**File**: `/app/vllm/vllm/v1/core/kv_cache_manager.py`

**Location**: In `get_computed_blocks()` method, after line 194

**Add**:
```python
# Check if we should use deterministic two-pass mode
deterministic_mode = os.getenv("VLLM_DETERMINISTIC_PREFIX_CACHE", "0") == "1"
block_size = self.block_size

if deterministic_mode and num_new_computed_tokens == 0 and request.num_tokens > block_size:
    # This is a CACHE MISS - force it to behave like a cache hit
    # by pretending we have cached blocks
    split_point = (request.num_tokens // block_size) * block_size
    
    if split_point >= block_size:
        # Mark this request for two-pass processing
        if not hasattr(request, '_deterministic_split_point'):
            request._deterministic_split_point = split_point
            request._needs_prefix_warmup = True
            
            if os.getenv("VLLM_DEBUG_PREFIX_CACHE", "0") == "1":
                print(f"[DETERMINISTIC] Request {request.request_id[-12:]} "
                      f"will use two-pass: [0:{split_point}] then [{split_point}:{request.num_tokens}]")
```

### Step 2: Modify Scheduling to Handle Two-Pass

**File**: `/app/vllm/vllm/v1/core/sched/scheduler.py`

**Location**: In `schedule()` method

**Add logic to split requests**:
```python
def schedule(self):
    # ... existing code ...
    
    for request in waiting_requests:
        if hasattr(request, '_needs_prefix_warmup') and request._needs_prefix_warmup:
            # This request needs two-pass processing
            split_point = request._deterministic_split_point
            
            # PASS 1: Process prefix only
            prefix_request = self._create_prefix_request(request, split_point)
            self._schedule_request(prefix_request)
            
            # Mark that prefix is being processed
            request._prefix_processed = False
            request._needs_prefix_warmup = False
            continue
        
        if hasattr(request, '_prefix_processed') and not request._prefix_processed:
            # Prefix is still being processed, skip for now
            continue
        
        # Normal scheduling
        self._schedule_request(request)
```

### Step 3: Handle Prefix Completion

**Add callback when prefix completes**:
```python
def _on_prefix_complete(self, request, prefix_output):
    # Prefix pass is done, cache the K/V
    self.kv_cache_manager.cache_blocks(request, request._deterministic_split_point)
    
    # Mark prefix as complete
    request._prefix_processed = True
    
    # Now the suffix can be processed with cached prefix
    # (it will naturally use cached K/V)
```

### Step 4: Environment Variable Configuration

**Add to documentation**:
```bash
# Enable deterministic prefix caching
export VLLM_DETERMINISTIC_PREFIX_CACHE=1

# Set block size for splitting (default: 16)
export VLLM_DETERMINISTIC_CACHE_BLOCK_SIZE=16

# Run vLLM
vllm serve model --enable-prefix-caching
```

## Simpler Alternative: Request Rewriting

If modifying the scheduler is too complex, we can rewrite requests at the API layer:

**File**: `/app/vllm/vllm/v1/engine/async_llm.py`

**Add before adding request**:
```python
async def add_request(self, request):
    if self.deterministic_prefix_cache and len(request.prompt_token_ids) > self.block_size:
        # Split into two requests
        split_point = (len(request.prompt_token_ids) // self.block_size) * self.block_size
        
        # Request 1: Prefix only (cache warmup)
        prefix_request = Request(
            request_id=f"{request.request_id}_prefix",
            prompt_token_ids=request.prompt_token_ids[:split_point],
            max_tokens=0,  # Don't generate, just cache
            ...
        )
        await self.engine_core.add_request(prefix_request)
        await self._wait_for_request_complete(prefix_request.request_id)
        
        # Request 2: Full request (will use cached prefix)
        await self.engine_core.add_request(request)
    else:
        # Normal path
        await self.engine_core.add_request(request)
```

## Testing

Run with deterministic mode:
```bash
cd /app/vllm
HIP_VISIBLE_DEVICES=4,5,6,7 \
VLLM_DEBUG_PREFIX_CACHE=1 \
VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
pytest -s tests/entrypoints/openai/test_prefix_cache_debug.py
```

Expected results:
```
vLLM+PC R1-R2: ✓ +0.000000 (all positions)
vLLM+PC R2-R3: ✓ +0.000000 (all positions)
```

## Performance Impact

- **First request (R1)**: ~80-100% slower (two passes instead of one)
- **Subsequent requests (R2+)**: No change (already using cache)
- **Amortized over 10 requests**: ~10-15% slower overall

## Current Status

- ✅ Diagnostic code added to track R1, R2, R3
- ✅ Confirmed R2-R5 are identical (problem is only R1 vs R2)
- ⏳ Implementation in progress
- ❌ Not yet tested

## Files to Modify

1. `/app/vllm/vllm/v1/core/kv_cache_manager.py` - Detect and mark requests for splitting
2. `/app/vllm/vllm/v1/core/sched/scheduler.py` - Handle two-pass scheduling
3. `/app/vllm/vllm/v1/engine/async_llm.py` - (Alternative) Request rewriting at API layer

## Recommendation

Start with the **Request Rewriting** approach (Alternative) because:
- ✅ Simpler to implement
- ✅ Less invasive (doesn't touch scheduler)
- ✅ Easier to test and debug
- ✅ Can be disabled with single flag

Then if that works, optimize by moving logic deeper into scheduler.
