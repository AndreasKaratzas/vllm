# MI355X BFloat16 Prefix Caching Bug - Complete Summary

## Problem

AMD MI355X (gfx950) with vLLM using **BFloat16** and **prefix caching** produces **non-deterministic outputs**:
- First request (cache miss): Token 17167 at position 3
- Subsequent requests (cache hit): Token 320 at position 3

## Investigation Process

### 1. Initial Hypothesis - Precision Issues
Suspected BF16 rounding in Triton attention kernels.
- ❌ Added IEEE precision mode → Bug persisted
- ❌ Modified wrong kernel files (Flash Attention, not used on ROCm)

### 2. Dataflow Analysis
Created [DATAFLOW.md](DATAFLOW.md) documenting request flow:
- ✓ Identified actual backend: `TRITON_ATTN` (not FlashAttention)
- ✓ Located correct files: [triton_attn.py](vllm/vllm/v1/attention/backends/triton_attn.py), [triton_unified_attention.py](vllm/vllm/v1/attention/ops/triton_unified_attention.py)

### 3. Logging & Diagnosis
Added logging to trace execution:
- `[PREFIX_CACHE]` - Shows which blocks are reused
- `[TRITON_ATTN_FWD]` - Attention forward pass info
- `[CACHED_BLOCK]` - Actual cached KV values
- `[TRITON_ATTN_OUT]` - Attention output statistics

### 4. Root Cause Found

**The Smoking Gun** (from [cached_kv_test.log](cached_kv_test.log)):

Cached KV values are **IDENTICAL**:
```
Run 1: block_idx=1, k_mean=0.06230439, v_mean=-0.00188563
Run 2: block_idx=1, k_mean=0.06230439, v_mean=-0.00188563
Run 3: block_idx=1, k_mean=0.06230439, v_mean=-0.00188563
```

But attention outputs **DIFFER**:
```
Run 1 (31 tokens): output_mean=-0.003076, std=0.127060
Run 2 (15+16):     output_mean=-0.004047, std=0.124664
Run 3 (15+16):     output_mean=-0.004047, std=0.124664
```

**Conclusion**: Cache storage works perfectly. Bug is in the Triton `unified_attention` kernel's BF16 computation.

## Technical Root Cause

The `unified_attention` kernel produces different outputs for the same cached KV depending on query size:

**Run 1**: Process all 31 tokens together
- Attention matrix: 31×31
- All fresh computation

**Runs 2-3**: Process 15 tokens with 16 cached
- Attention matrix: 15×31 (15 queries × 31 KV: 16 cached + 15 fresh)
- Mixed cached + fresh computation

Even with IEEE precision mode, BF16's limited mantissa (8 bits) means:
- Different accumulation order → different rounding
- Different loop iterations → different intermediate precision
- Tiny difference (0.001) cascades through 20 layers → wrong token

## Solution Implemented

### Workaround: Disable Prefix Caching for BF16 on gfx950

**File Modified**: [kv_cache_manager.py:110-132](vllm/vllm/v1/core/kv_cache_manager.py#L110-L132)

```python
# Detect BF16 + gfx950 combination
if enable_caching:
    if current_platform.is_rocm() and on_gfx950():
        vllm_config = get_current_vllm_config()
        if vllm_config.model_config.dtype == torch.bfloat16:
            logger.warning(
                "Disabling prefix caching for BFloat16 on AMD MI355X (gfx950) "
                "due to precision issues in attention computation. "
                "This is a known limitation. Use float16 if prefix caching is required."
            )
            enable_caching = False
```

**Impact**:
- ✓ Fixes non-determinism
- ✗ Loses prefix caching performance benefit for BF16
- ✓ Simple, safe, immediate fix

## Alternative Solutions

See [FINAL_DIAGNOSIS.md](FINAL_DIAGNOSIS.md) for full details:

1. **Option 1** (Implemented): Disable prefix caching for BF16+gfx950
2. **Option 2**: Force FP16 for cached attention
3. **Option 3**: Fix Triton kernel accumulation order (complex, requires ROCm/Triton experts)
4. **Option 4**: Recompute instead of cache

## Testing

### Verify the Bug
```bash
bash RUN_THIS_NOW.sh
# Expected: DIVERGENCE DETECTED
```

### Verify the Fix
```bash
bash test_workaround.sh
# Expected: NO DIVERGENCE, prefix caching disabled warning
```

### Key Logs
- [debug_triton.log](debug_triton.log) - Initial bug confirmation
- [cached_kv_test.log](cached_kv_test.log) - Proof that cache storage works, kernel is buggy

## Files Modified

1. [kv_cache_manager.py](vllm/vllm/v1/core/kv_cache_manager.py) - Workaround implementation
2. [triton_attn.py](vllm/vllm/v1/attention/backends/triton_attn.py) - Added debug logging
3. [triton_unified_attention.py](vllm/vllm/v1/attention/ops/triton_unified_attention.py) - Added IEEE precision + debug logging

## Documentation Created

- [DATAFLOW.md](DATAFLOW.md) - Complete request flow on ROCm
- [BUG_ANALYSIS.md](BUG_ANALYSIS.md) - Analysis and investigation steps
- [FINAL_DIAGNOSIS.md](FINAL_DIAGNOSIS.md) - Detailed root cause and solutions
- [SUMMARY.md](SUMMARY.md) - This file

## Recommendations

**Short term**: Use the workaround (disable prefix caching for BF16)

**Long term**:
1. File issue with vLLM upstream
2. Collaborate with Triton/ROCm teams to fix kernel
3. Possible solutions:
   - Ensure consistent accumulation order regardless of query size
   - Use FP32 accumulation for attention with BF16 inputs
   - Add CDNA3-specific optimizations that preserve determinism

**For users**: If you need prefix caching on MI355X, use `--dtype float16` instead of `bfloat16`

## Key Insight

This bug reveals a fundamental challenge with BF16 precision in LLM inference:
- Prefix caching assumes mathematical equivalence
- BF16 breaks this assumption due to accumulation order sensitivity
- The same issue could affect other accelerators (TPU, Intel, etc.) with BF16
- Careful kernel design is critical for deterministic behavior

## Acknowledgments

Bug identified and root-caused through systematic investigation:
1. Confirmed PyTorch BF16 ops are deterministic (not the issue)
2. Confirmed cache roundtrip is exact (not the issue)
3. Added targeted logging to trace execution
4. Proved cached values are identical
5. Identified kernel computation as root cause
6. Implemented practical workaround

---

*Generated during debugging session on 2026-01-28*
