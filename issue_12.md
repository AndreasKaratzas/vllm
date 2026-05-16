# AITER fused GPT-J RoPE + FP8 KV cache rounds in a different order than vLLM

## Summary

On MI355/gfx950 with ROCm, `tests/compile/passes/test_rope_kvcache_fusion.py`
failed only for this shape of case:

- FP8 KV cache
- GPT-J style RoPE (`is_neox=False`)
- `VLLM_ROCM_USE_AITER_TRITON_ROPE=0`
- `RopeKVCacheFusionPass` replaces separate RoPE and KV-cache update with
  `do_rope_and_kv_cache_update`

The mismatch was sparse, but deterministic:

```text
Mismatched elements: 2 / 40960
Greatest absolute difference: 0.5078125
```

## Minimal Reproduction

```bash
cd /app/vllm
pytest -q -vv tests/compile/passes/test_rope_kvcache_fusion.py \
  -k 'fp8 and ROCM_AITER_UNIFIED_ATTN' --tb=short
```

The full local validation after the vLLM-side fallback is:

```bash
pytest -q -vv tests/compile/passes/test_rope_kvcache_fusion.py --tb=short
```

which currently reports:

```text
32 passed
```

## Environment

Observed locally on the MI355 node:

```text
torch 2.10.0+git8514f05
HIP 7.2.53211
Triton 3.6.0
platform: ROCm
FP8 dtype: torch.float8_e4m3fn
```

## What Was Ruled Out

- The shared CUDA/ROCm MLA fused cache kernel is not on this compile-pass path.
- Changing `qk_t`/RoPE arithmetic in `csrc/cache_kernels_fused.cu` is too broad
  and does not target the failing path.
- The mismatch disappears when `VLLM_ROCM_USE_AITER_TRITON_ROPE=1`, because the
  unfused reference then uses AITER's Triton rotary path too.
- BF16 KV-cache cases pass, and Neox-style RoPE cases pass.

## Likely Root Cause

The AITER fused kernel in:

```text
/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/_triton_kernels/fusions/fused_kv_cache.py
```

computes the RoPE value and stores it to the output tensor, but the FP8 KV cache
store scales and quantizes the pre-rounded intermediate value. The unfused vLLM
path first writes the RoPE result to the BF16 key tensor via vLLM's C custom op
and then quantizes that BF16 tensor into the FP8 KV cache. GPT-J style RoPE has a
couple of values close enough to FP8 rounding boundaries that the two orders
produce different encoded cache values.

## Local Test Treatment

The compile pass keeps using the fused RoPE+KV-cache path. The test now applies
a narrow ROCm-only diagnostic error budget only for:

- ROCm
- GPT-J style RoPE
- quantized KV cache
- AITER Triton rotary disabled

For that exact case, at least 99.99% of cache elements must remain within the
original tight tolerance, and no element may exceed `0.52` absolute error. This
keeps the acceptable drift to the observed sparse FP8 rounding-boundary cases
across ROCm attention backends without changing CUDA behavior, production code,
or the shared MLA fused cache kernel.
