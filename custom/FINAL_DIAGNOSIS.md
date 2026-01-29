# MI355X BFloat16 Prefix Caching Bug - Final Diagnosis

## Bug Confirmed

**Status**: ✗ BUG ACTIVE
**Platform**: AMD MI355X (gfx950) with ROCm
**Dtype**: BFloat16
**Symptom**: First request produces different output than subsequent cached requests

**Divergence**:
- Run 1: Token at position 3 = **17167**
- Runs 2-3: Token at position 3 = **320**

## Root Cause Identified

### The Smoking Gun

**Cached KV values are IDENTICAL across all runs:**

| Run | Type | Cached Block K Mean | Cached Block V Mean |
|-----|------|---------------------|---------------------|
| 1 | Fresh (31 tokens) | 0.06230439 | -0.00188563 |
| 2 | Cached (15+16) | 0.06230439 | -0.00188563 |
| 3 | Cached (15+16) | 0.06230439 | -0.00188563 |

**But attention outputs DIFFER:**

| Run | Type | Layer 0 Output Mean | Layer 0 Output Std |
|-----|------|--------------------|--------------------|
| 1 | Fresh (31 tokens) | -0.003076 | 0.127060 |
| 2 | Cached (15+16) | -0.004047 | 0.124664 |
| 3 | Cached (15+16) | -0.004047 | 0.124664 |

**Difference**: `|-0.003076 - (-0.004047)| = 0.000971` in layer 0
→ Cascades through 20 layers → Wrong token at position 3

### Conclusion

✓ **Cache storage/retrieval works perfectly** - KV values preserved exactly
✗ **Bug is in the Triton attention kernel** - `unified_attention()` produces different outputs for same KV depending on query size

## Technical Details

### Execution Paths

**Run 1 (cache miss):**
```
Prompt: 31 tokens → All fresh computation
Attention: Q[31] @ K[31]ᵀ → P[31×31] @ V[31]
KV Cache: Write all 31 tokens to block_id=1
Result: output_mean = -0.003076
```

**Runs 2-3 (cache hit):**
```
Prompt: 31 tokens → Reuse first 16, compute 15
Attention: Q[15] @ K[31]ᵀ → P[15×31] @ V[31]
           where K[0:16] and V[0:16] come from cache
KV Cache: Read 16 cached, write 15 new
Result: output_mean = -0.004047
```

### Why Different Outputs?

Even though K and V are identical, the attention kernel produces different results because:

1. **Different attention matrix dimensions**:
   - Run 1: 31×31 (all fresh)
   - Runs 2-3: 15×31 (mixed cached + fresh)

2. **Different code paths in Triton kernel**:
   - The `unified_attention` kernel may handle prefill differently based on query length
   - Different loop iterations, block sizes, or memory access patterns

3. **BF16 accumulation order matters**:
   - BF16 has limited precision (8-bit mantissa)
   - Dot products accumulated in different order → different rounding
   - Even with IEEE precision mode enabled (line 921 in triton_unified_attention.py)

## Affected Code

**File**: `/app/vllm/vllm/v1/attention/ops/triton_unified_attention.py`

**Kernel**: `kernel_unified_attention_2d` (lines 96-402)

This kernel is called for both:
- Full prefill (31 tokens)
- Partial prefill (15 tokens with cached context)

**Key operations** (BF16-sensitive):
- Line 320: `S += scale * tl.dot(Q, K, input_precision=IN_PRECISION)`
- Line 384: `acc += tl.dot(P.to(V.dtype), V, input_precision=IN_PRECISION)`

## Attempted Fixes (Failed)

1. ✗ Added `IN_PRECISION="ieee"` for BF16 on gfx950 (line 921)
   - Forces IEEE precision in Triton matmuls
   - Bug persists

2. ✗ Modified other attention kernels (decode, prefill)
   - These weren't being used

3. ✗ Added precision fixes to FlashAttention backend
   - ROCm doesn't use FlashAttention

## Why This is Hard to Fix

The issue is **fundamentally architectural**:

1. Prefix caching assumes: `attn(Q_all, K_all, V_all) = attn(Q_new, [K_cached; K_new], [V_cached; V_new])`

2. In exact arithmetic, this holds

3. With BF16:
   - Associativity doesn't hold: `(a + b) + c ≠ a + (b + c)`
   - Different accumulation order → different rounding → different result
   - Even tiny differences (0.001) cascade through deep networks

4. The Triton kernel processes blocks differently depending on query size:
   - 31 queries: One processing pattern
   - 15 queries: Different processing pattern
   - Same cached KV, different BF16 rounding → different outputs

## Potential Solutions

### Option 1: Disable Prefix Caching for BF16 on gfx950

**Pros**: Simple, safe, works immediately
**Cons**: Performance loss from prefix caching

```python
# In vllm/config/cache.py or kv_cache_manager.py
if dtype == torch.bfloat16 and platform.is_gfx950():
    logger.warning("Disabling prefix caching for BF16 on gfx950 due to precision issues")
    enable_prefix_caching = False
```

### Option 2: Force FP16 for Cached Attention on gfx950

**Pros**: Maintains prefix caching, better precision
**Cons**: Performance overhead, complexity

```python
# In triton_attn.py
if use_cached_kv and dtype == bfloat16 and is_gfx950:
    # Upcast to FP16 for attention when using cache
    q = q.to(torch.float16)
    key_cache = key_cache.to(torch.float16)
    value_cache = value_cache.to(torch.float16)
    # Compute attention in FP16
    # Downcast output back to BF16
```

### Option 3: Modify Triton Kernel for Consistent Accumulation

**Pros**: Root cause fix
**Cons**: Very complex, requires deep Triton/CDNA3 knowledge

Ensure the attention kernel uses the **same accumulation order** regardless of query size:
- Force same block sizes
- Force same loop iteration patterns
- Use same reduction strategies

### Option 4: Recompute Instead of Cache for BF16

**Pros**: Simple, maintains determinism
**Cons**: Defeats purpose of prefix caching

Instead of caching KV:
- Always recompute from scratch
- Use prefix caching only for deduplication, not for saving compute

## Recommended Action

**Short term**: Option 1 (disable prefix caching for BF16+gfx950)
- Safe, simple, immediate fix
- Document as known limitation

**Long term**: Option 3 (fix Triton kernel)
- File issue with vLLM team
- Requires collaboration with Triton/ROCm experts
- May need CDNA3-specific tuning

## Test to Verify Fix

```bash
# Should produce IDENTICAL outputs for all runs
python3 test_simple_compare.py

# Expected output after fix:
# ✓ All runs produced same tokens: True
```

## References

- Test logs: [cached_kv_test.log](cached_kv_test.log)
- Dataflow analysis: [DATAFLOW.md](DATAFLOW.md)
- Bug analysis: [BUG_ANALYSIS.md](BUG_ANALYSIS.md)
- Affected kernel: [triton_unified_attention.py:96-402](vllm/vllm/v1/attention/ops/triton_unified_attention.py#L96-L402)
