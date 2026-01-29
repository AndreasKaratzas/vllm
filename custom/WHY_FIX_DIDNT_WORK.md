# Why The Fix Didn't Work (Yet)

## TL;DR

**The fix was applied correctly**, but vLLM is using the **native HIP C++ kernel** instead of the **Triton kernel** where our fix was applied!

## The Two Kernel Paths

vLLM has **TWO different implementations** for paged attention on ROCm:

### Path 1: Native HIP C++ Kernel (Bypasses Our Fix)
**File**: C++ code in `vllm/csrc/rocm/`
**Function**: `ops.paged_attention_rocm()`
**Used when**:
- `VLLM_ROCM_CUSTOM_PAGED_ATTN=1` (default)
- Block size is power-of-2 (16, 32, 64, etc.)
- GPU is gfx9 family (including gfx950)
- Other conditions met

**Issue**: This is compiled C++/HIP code. Our Python/Triton `IN_PRECISION` fix **doesn't apply here**!

### Path 2: Triton Kernel (Has Our Fix)
**File**: `vllm/v1/attention/ops/chunked_prefill_paged_decode.py`
**Function**: `kernel_paged_attention_2d()`
**Used when**:
- `VLLM_ROCM_CUSTOM_PAGED_ATTN=0` (forced)
- OR block size is non-power-of-2 (like 544)

**Status**: ✅ Our `IN_PRECISION` fix is applied here!

---

## Code That Makes The Decision

**File**: `/app/vllm/vllm/v1/attention/ops/chunked_prefill_paged_decode.py:338-400`

```python
# Line 340: Check if custom kernel should be used
use_custom = use_rocm_custom_paged_attention(
    query.dtype, head_size, block_size, ...
)

# Line 357: Check if block size is power of 2
is_pow2 = block_size > 0 and (block_size & (block_size - 1) == 0)
if not is_pow2:
    use_custom = False  # Force Triton for non-power-of-2

# Line 361: BRANCH POINT
if use_custom:
    # ❌ Uses native HIP C++ kernel (ops.paged_attention_rocm)
    # Our IN_PRECISION fix doesn't apply!
    ops.paged_attention_rocm(...)
else:
    # ✅ Uses Triton kernel (kernel_paged_attention_2d)
    # Our IN_PRECISION fix DOES apply!
    kernel_paged_attention_2d[...](
        ...,
        IN_PRECISION=IN_PRECISION,  # ← Our fix
    )
```

---

## Why Your Test Still Failed

Your test likely used:
- **dtype**: `float16` (default)
- **block_size**: 16 or 32 (default, power-of-2)
- **VLLM_ROCM_CUSTOM_PAGED_ATTN**: `1` (default)

This means: **Native HIP kernel was used** → Our fix didn't apply!

---

## Solutions

### Solution 1: Force Triton Kernel (Recommended)

Disable the custom ROCm paged attention to force Triton path:

```bash
export VLLM_ROCM_CUSTOM_PAGED_ATTN=0

# Then run your test
cd /app/vllm
pytest -s -v tests/entrypoints/openai/test_prefix_cache_debug.py::test_side_by_side
```

**Or use the provided script:**
```bash
bash /app/test_with_triton_kernel.sh
```

### Solution 2: Use Non-Power-of-2 Block Size

This forces Triton even with custom paged attention enabled:

```bash
# Add to test server args
--block-size 544
```

**Or modify the test:**
```python
base_args = [
    "--dtype", "float16",
    "--max-model-len", "512",
    "--enforce-eager",
    "--generation-config", "vllm",
    "--max_num_seqs", "1",
    "--block-size", "544",  # ← Add this
]
```

### Solution 3: Fix the Native HIP Kernel (Advanced)

The native HIP kernel also needs a similar precision fix, but it's in C++/HIP:

**File**: `vllm/csrc/rocm/attention.cu`

This requires:
1. Modifying C++ code
2. Recompiling the extension
3. Understanding HIP/MFMA instructions

**Not recommended** unless you're familiar with HIP/CUDA kernel development.

---

## Diagnostic

Run this to see which kernel path your configuration uses:

```bash
python /app/check_kernel_path.py
```

This will show:
- Current GPU architecture
- Environment variable settings
- Which kernel would be used for different configurations
- Whether your IN_PRECISION fix applies

---

## Expected Results After Forcing Triton

Once Triton kernel is forced (Solution 1 or 2), you should see:

```
--- VLLM WITH PREFIX CACHING ---
  Deterministic: ✓ All 5 runs identical
```

**All runs should match:**
```
vLLM+PC Run 1: 'The European Union consists of 27 member states'
vLLM+PC Run 2: 'The European Union consists of 27 member states'
vLLM+PC Run 3: 'The European Union consists of 27 member states'
vLLM+PC Run 4: 'The European Union consists of 27 member states'
vLLM+PC Run 5: 'The European Union consists of 27 member states'
```

---

## Additional Finding: HuggingFace Divergence

Your test also revealed:

```
--- HUGGINGFACE ---
  HF+cache == HF-cache: ✗
    With cache:    'The European Union (EU) consists of **2'
    Without cache: 'The European Union consists of 27 member states'
```

This is **separate from the vLLM bug** and suggests:
1. HuggingFace also has numerical differences between cache/no-cache paths
2. This might be expected behavior (different KV cache implementations)
3. OR it's a similar precision issue in transformers on gfx950

This explains why:
- vLLM+PC Run 2-5 match HF+cache (both use cached KV)
- vLLM-PC and vLLM+PC Run 1 match HF-cache (both compute from scratch)

---

## Summary

| Configuration | Kernel Used | Fix Applies? | Status |
|---------------|-------------|--------------|--------|
| **Default** (VLLM_ROCM_CUSTOM_PAGED_ATTN=1) | Native HIP C++ | ❌ No | Bug present |
| **VLLM_ROCM_CUSTOM_PAGED_ATTN=0** | Triton | ✅ Yes | **Should work** |
| **block-size=544** | Triton | ✅ Yes | **Should work** |

---

## Next Steps

1. **Run diagnostic:**
   ```bash
   python /app/check_kernel_path.py
   ```

2. **Test with forced Triton:**
   ```bash
   bash /app/test_with_triton_kernel.sh
   ```

3. **If still fails**, check:
   - Triton version compatibility
   - Whether code was recompiled after changes
   - Any caching of compiled kernels

4. **If succeeds with Triton**, you can either:
   - Keep using `VLLM_ROCM_CUSTOM_PAGED_ATTN=0` in production
   - OR investigate fixing the native HIP kernel as well

---

**Created**: 2026-01-28
**Issue**: Prefix caching non-determinism on MI355X (gfx950)
**Status**: Fix applied to Triton kernel, needs kernel path verification
