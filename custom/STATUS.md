# Kernel Redesign Implementation Status

## What I've Done

### 1. Added DETERMINISTIC_CACHE flag to all kernels
- ✅ `kernel_unified_attention_2d` (2D prefill kernel)
- ✅ `kernel_unified_attention_3d` (3D parallel softmax kernel)  
- ✅ `attention_decode_kernel` (decode kernel)

### 2. Modified context_len calculation
In all 3 kernels, added logic:
```python
actual_context_len = seq_len - cur_batch_query_len

if DETERMINISTIC_CACHE:
    context_len = 0  # Force identical tile order
    true_context_len = actual_context_len
else:
    context_len = actual_context_len
    true_context_len = actual_context_len
```

### 3. Added flag to unified_attention wrapper
- Detects gfx950 + bfloat16
- Sets `DETERMINISTIC_CACHE = True` by default (env var: `VLLM_DETERMINISTIC_CACHE`)
- Passes flag to all kernel launches

## Current Limitation

**The approach won't fully work** because:
- vLLM passes different query tensor shapes to the kernel:
  - R1 (miss): q.shape = [31, 16, 128]
  - R2 (hit): q.shape = [15, 16, 128]
- Setting `context_len=0` inside the kernel doesn't change the fact that we're processing different numbers of queries

## Next Step Required

We need to modify the **wrapper** (not just the kernel) to:
1. Pad queries to full length when cache hit detected
2. Pass output mask to indicate which positions to compute
3. Ensure identical query shapes regardless of cache state

This requires changes in `/app/vllm/vllm/v1/attention/backends/triton_attn.py` around line 490.

## Files Modified
1. `/app/vllm/vllm/v1/attention/ops/triton_unified_attention.py` - Kernels + wrapper
2. `/app/vllm/PREFIX_CACHE_FILES.md` - Documentation  
3. `/app/vllm/KERNEL_REDESIGN_PLAN.md` - Original plan
4. `/app/vllm/KERNEL_REDESIGN_APPROACH_REVISED.md` - Revised approach
5. `/app/vllm/BUG_ROOT_CAUSE_FINAL.md` - Root cause analysis

## Test Status
❌ Compilation error: Need to fix 3D kernel signature

## Performance Impact (Estimated)
- With current approach: ~20-30% slower on cache hits
- With full redesign (padding): ~15-25% slower

## Correctness Status
🔄 In progress - kernel modifications done, but wrapper changes needed for full fix
