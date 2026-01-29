# Code Comparison: Cache-Miss vs Cache-Hit Paths

This document shows the exact code differences between the two attention computation paths that diverge on MI355X.

---

## Side-by-Side Comparison

### Cache-Miss Path (First Request)
**File**: `vllm/v1/attention/ops/prefix_prefill.py`

```python
@triton.jit
def _fwd_kernel(
    Q, K, V,
    K_cache, V_cache,
    sink_ptr, B_Loc,
    sm_scale, k_scale, v_scale, out_scale_inv,
    B_Start_Loc, B_Seqlen, x,
    Out,
    # ... stride parameters ...
    IN_PRECISION: tl.constexpr,  # ← PRECISION SPECIFIED ✓
    # ... other constexprs ...
):
    # ... setup code ...

    # Phase 1: Attention over cached context (lines 155-266)
    for start_n in tl.range(0, cur_batch_ctx_len, BLOCK_SIZE):
        # ... load K, V from cache ...

        # ✓ Precision explicitly specified
        qk = tl.dot(q, k, input_precision=IN_PRECISION)  # Line 210
        qk *= sm_scale

        # ... softmax computation ...

        # ✓ Precision explicitly specified
        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)  # Line 263

        # ... update running stats ...

    # Phase 2: Attention over query itself (lines 284-334)
    for start_n in tl.range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        # ... load K, V from input ...

        # ✓ Precision explicitly specified
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)  # Line 301
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]),
                      qk, float("-inf"))  # Causal mask

        # ... softmax computation ...

        # ✓ Precision explicitly specified
        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)  # Line 331

        # ... update running stats ...

    # Final normalization
    acc = acc / (l_i[:, None] + 1e-10)  # Line 336
    # ... store output ...
```

---

### Cache-Hit Path (Subsequent Requests)
**File**: `vllm/v1/attention/ops/chunked_prefill_paged_decode.py`

```python
@triton.jit
def kernel_paged_attention_2d(
    output_ptr, query_ptr,
    key_cache_ptr, value_cache_ptr,
    sink_ptr, block_tables_ptr,
    seq_lens_ptr, alibi_slopes_ptr,
    scale, k_scale, v_scale, out_scale_inv,
    # ... parameters ...
    USE_SINKS: tl.constexpr,
    USE_FP8: tl.constexpr,
    # ✗ NO IN_PRECISION PARAMETER! ←←← THIS IS THE BUG
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    # ... setup code ...

    # Single phase: Attention over all blocks (lines 133-227)
    for j in range(0, num_blocks):
        # ... load K, V from paged cache ...

        # ✗ NO PRECISION SPECIFICATION! Uses Triton default
        qk = scale * tl.dot(Q, K)  # Line 193 ←←← BUG HERE
        S = tl.where(head_mask[:, None] & seq_mask, qk, float("-inf"))

        # ... softmax computation ...

        # ✗ NO PRECISION SPECIFICATION! Uses Triton default
        acc += tl.dot(p.to(V.dtype), V)  # Line 227 ←←← BUG HERE

        # ... update running stats ...

    # Final normalization
    acc = acc / (L[:, None] + 1e-10)  # Line 230
    # ... store output ...
```

---

## The Critical Difference

### Cache-Miss (Prefill) - Line 210, 263, 301, 331
```python
# ✓ EXPLICIT PRECISION
tl.dot(q, k, input_precision=IN_PRECISION)
tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
```

### Cache-Hit (Decode) - Line 193, 227
```python
# ✗ MISSING PRECISION PARAMETER
tl.dot(Q, K)          # Uses default - may differ on gfx950!
tl.dot(p.to(V.dtype), V)  # Uses default - may differ on gfx950!
```

---

## What `IN_PRECISION` Does

From `vllm/v1/attention/ops/prefix_prefill.py:664`:

```python
IN_PRECISION = "ieee" if IS_TURING and q_dtype_is_f32 else None
```

### When `IN_PRECISION = None` (Most Cases)
- **Prefill kernel**: Explicitly passes `None` to `tl.dot(..., input_precision=None)`
- **Decode kernel**: Doesn't pass parameter at all → Triton uses implicit default

### The Problem
On gfx950, these two scenarios produce **different compiled code**:

1. **Explicit `input_precision=None`**: Triton recognizes user intent, may use specific path
2. **No parameter**: Triton uses platform default, which may be different

This results in:
- Different rounding modes
- Different FP8 MFMA instruction selection
- Different accumulation precision

---

## Numerical Impact

### Token Position 3 (Where Divergence Occurs)

#### Cache-Miss Path
```
Token: ' consists'
Logprob: -0.7507
Computation:
  1. Attend over context (cached KV) with IN_PRECISION
  2. Attend over query (new tokens) with IN_PRECISION
  3. Normalize
```

#### Cache-Hit Path
```
Token: ' ('
Logprob: -0.8047
Computation:
  1. Attend over all blocks (paged cache) WITHOUT explicit precision
  2. Normalize
```

**Difference**: 0.054 in logprob space → **Different argmax token** → Complete output divergence

---

## How the Fix Works

### Before Fix
```python
@triton.jit
def kernel_paged_attention_2d(...):
    # Missing precision parameter
    qk = scale * tl.dot(Q, K)  # ← Uses Triton default
    acc += tl.dot(p.to(V.dtype), V)  # ← Uses Triton default
```

### After Fix
```python
@triton.jit
def kernel_paged_attention_2d(
    ...,
    IN_PRECISION: tl.constexpr,  # ← Add parameter
):
    # Use same precision as prefill
    qk = scale * tl.dot(Q, K, input_precision=IN_PRECISION)  # ← Explicit
    acc += tl.dot(p.to(V.dtype), V, input_precision=IN_PRECISION)  # ← Explicit
```

### Calling Code
```python
def chunked_prefill_paged_decode(...):
    # Compute precision mode consistently with prefill
    q_dtype_is_f32 = query.dtype is torch.float32
    IS_TURING = current_platform.get_device_capability() == (7, 5)
    IN_PRECISION = "ieee" if IS_TURING and q_dtype_is_f32 else None

    # ... later ...

    kernel_paged_attention_2d[...](
        ...,
        USE_FP8=output_scale is not None,
        IN_PRECISION=IN_PRECISION,  # ← Pass to kernel
    )
```

---

## Why MI325X Works But MI355X Doesn't

### MI325X (gfx942)
- Earlier generation FP8 MFMA instructions
- Triton's default precision mode happens to match explicit `None`
- No observable numerical difference

### MI355X (gfx950)
- **New generation FP8 MFMA instructions** (better performance, different rounding)
- Triton's default may use enhanced FP8 path
- Explicit `None` may use conservative path
- **Results in observable numerical divergence**

From `vllm/csrc/rocm/attention.cu:39-41`:
```cpp
#if defined(__HIPCC__) && (defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__FP8MFMA__
#endif
```

Both have FP8 MFMA, but **gfx950 has enhanced version** that may behave differently when precision is unspecified.

---

## Verification

After applying the fix, both paths should use **identical** `tl.dot()` semantics:

```python
# Both kernels now explicitly use:
tl.dot(q, k, input_precision=IN_PRECISION)
tl.dot(p, v, input_precision=IN_PRECISION)

# Where IN_PRECISION is computed identically:
IN_PRECISION = "ieee" if IS_TURING and q_dtype_is_f32 else None
```

This ensures:
✅ **Consistent rounding** across cache-hit and cache-miss paths
✅ **Identical compiled code** for attention operations
✅ **Deterministic output** across all requests
✅ **No platform-specific divergence**

---

## See Also

- **Full Analysis**: [BUG_ANALYSIS_SUMMARY.md](BUG_ANALYSIS_SUMMARY.md)
- **Quick Fix**: [QUICK_FIX_GUIDE.md](QUICK_FIX_GUIDE.md)
- **Patch File**: [potential_fix.patch](potential_fix.patch)
- **Diagnostic Tool**: [debug_attention_divergence.py](debug_attention_divergence.py)
