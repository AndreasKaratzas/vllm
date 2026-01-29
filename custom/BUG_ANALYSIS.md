# MI355X BFloat16 Prefix Caching Bug - Analysis

## Bug Confirmation

✗ **Bug persists after IEEE precision fix**

Divergence at token position 3:
- Run 1: token=17167
- Runs 2+: token=320

## Root Cause Analysis

### The Smoking Gun (from debug_triton.log)

**Run 1 (cache miss - all fresh computation):**
```
[PREFIX_CACHE] num_tokens=31, num_computed_tokens=0, block_hashes=([],)
[TRITON_ATTN_FWD] num_tokens=31, max_query_len=31, max_seq_len=31
[KV_WRITE] key_shape=[31, 8, 128]
[TRITON_ATTN_OUT] output_mean=-0.003076, output_std=0.127060  (layer 0)
```
- Processes ALL 31 prompt tokens together
- 31x31 attention matrix
- Writes all 31 tokens to cache

**Runs 2-3 (cache hit - partial fresh computation):**
```
[PREFIX_CACHE] num_tokens=31, num_computed_tokens=16, block_hashes=[KVCacheBlock(...)]
[TRITON_ATTN_FWD] num_tokens=15, max_query_len=15, max_seq_len=31
[KV_WRITE] key_shape=[15, 8, 128]
[TRITON_ATTN_OUT] output_mean=-0.004047, output_std=0.124664  (layer 0)
```
- Reuses first 16 tokens from cache (from Run 1)
- Only processes remaining 15 tokens (16-30)
- 15x31 attention matrix (15 query tokens attend to 31 KV tokens: 16 cached + 15 fresh)
- Writes only 15 new tokens to cache

**Layer 0 outputs already differ!**
- Run 1: `output_mean=-0.003076`
- Runs 2-3: `output_mean=-0.004047`
- Difference: ~0.001 (this cascades through 20 layers → wrong token)

## Why This Happens

### The Core Issue

When prefix caching reuses the first N tokens, the model should produce IDENTICAL outputs for the remaining tokens. But it doesn't:

1. **Run 1 path**: Compute attention for all 31 tokens together
   - Q, K, V are all fresh
   - Attention: Q[31] @ K[31]^T → P[31x31] @ V[31]

2. **Runs 2-3 path**: Compute attention for only 15 tokens, reading 16 from cache
   - Q[15] is fresh (for tokens 16-30)
   - K[31], V[31] are mixed: 16 cached + 15 fresh
   - Attention: Q[15] @ K[31]^T → P[15x31] @ V[31]

Even though K[0:16] and V[0:16] are IDENTICAL (cached from Run 1), the attention computation produces different outputs.

### Possible Causes

#### 1. ✓ BF16 Precision (partially mitigated with IEEE mode)
- We added `IN_PRECISION="ieee"` to force IEEE precision
- Bug still persists → IEEE alone is not enough

#### 2. ⚠️ Different Attention Kernel Code Paths
- Run 1: Full prefill (31 tokens)
- Runs 2-3: Partial prefill (15 tokens) with cached context
- The `unified_attention` kernel may handle these cases differently
- Different code paths → different BF16 rounding behavior

#### 3. ⚠️ KV Cache Read/Write Precision Issue
- Cached KV might not be stored/loaded with sufficient precision
- But: `test_cache_roundtrip.py` passed (exact roundtrip)
- However: This only tested simple tensors, not the actual vLLM cache structure

#### 4. ⚠️ Block Table Indexing Issue
- The `unified_attention` kernel uses `block_table` to locate cached KV
- Possible off-by-one or indexing bug when mixing cached + fresh KV
- This could cause wrong KV values to be read → wrong outputs

## What We've Tried

1. ✓ Added IEEE precision to Triton kernels (still fails)
2. ✓ Added logging to trace execution (found the divergence point)
3. ✓ Verified PyTorch BF16 ops are deterministic (passed)
4. ✓ Verified simple cache roundtrip works (passed)

## Next Steps

### Priority 1: Check if KV Cache Storage is Exact

The `test_cache_roundtrip.py` test was too simple. We need to test the ACTUAL vLLM cache write/read:

```python
# Test:
# 1. Run 1: Write 31 tokens to vLLM KV cache
# 2. Extract cached KV for tokens 0-15
# 3. Run 2: Write 15 tokens, reading first 16 from cache
# 4. Compare: Do cached tokens 0-15 have IDENTICAL values in both runs?
```

If cached values differ → Cache storage bug (block table, data layout, etc.)
If cached values match → Attention kernel bug (different computation path)

### Priority 2: Add Detailed KV Cache Read Logging

Add logging to see EXACTLY what KV values are read from cache:

```python
# In unified_attention kernel or before it's called:
# Log the actual cached KV values being read
# Compare Run 1 vs Runs 2-3
```

### Priority 3: Check Block Table Construction

The `block_table` tells the attention kernel where to find cached KV blocks. If this is wrong, the kernel reads garbage:

```python
# In kv_cache_manager.py:
# Log the block_table for each run
# Verify it points to the correct cache blocks
```

### Priority 4: Test with FP16

Run the SAME test with `--dtype float16`:
- If FP16 works → Confirms this is a BF16-specific issue
- If FP16 also fails → Might be a more fundamental caching bug

### Priority 5: Disable Prefix Caching for BF16

If we can't fix it quickly, add a workaround:

```python
# In config or cache manager:
if dtype == torch.bfloat16 and is_gfx950():
    enable_caching = False
    logger.warning("Disabling prefix caching for BF16 on gfx950 due to known bug")
```

## Key Questions to Answer

1. Are the cached KV values EXACTLY the same when read back?
2. Does the attention kernel use a different code path for partial prefill?
3. Is the block_table correctly pointing to cached blocks?
4. Does FP16 have the same issue?

## Files to Investigate Next

1. `/app/vllm/vllm/v1/attention/ops/triton_unified_attention.py` - The attention kernel itself
2. `/app/vllm/vllm/v1/core/kv_cache_manager.py` - How blocks are allocated and tracked
3. `/app/vllm/vllm/v1/attention/backends/triton_attn.py` - How KV cache is passed to kernel
