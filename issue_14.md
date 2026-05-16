# ROCm AITER Unified Cross-Attention WER Drift

## Summary

`CohereLabs/cohere-transcribe-03-2026` uses encoder-decoder
cross-attention and is a useful ASR correctness probe for variable encoder
lengths. On ROCm, forcing `ROCM_AITER_UNIFIED_ATTN` produces a noticeably
higher WER than the CUDA/Triton baseline and than ROCm `TRITON_ATTN`.

This should be investigated by the AITER attention team rather than hidden by
having the AITER backend call the vLLM Triton backend internally.

## Observed Results

On the tiny filtered Earnings22 validation dataset:

- CUDA baseline is around `11.88` WER with expected WER `11.92`.
- ROCm `TRITON_ATTN` is around `11.76`-`11.78` WER after fixing the Triton
  encoder-decoder path to propagate `causal=False` into the unified
  attention mask.
- ROCm `ROCM_AITER_UNIFIED_ATTN` is higher, around `12.77` WER.

Local MI355 runs measured:

- `TRITON_ATTN`: `11.783094865937237`
- `TRITON_ATTN`: `11.759047733557773`
- `ROCM_AITER_UNIFIED_ATTN`: `12.769027293495249`

## Reproduction

Run the transcription correctness test on ROCm:

```bash
PYTHONPATH=/app/vllm pytest -q -s \
  tests/entrypoints/speech_to_text/correctness/test_transcription_api_correctness.py \
  -k 'cohere-rocm-aiter-unified-attn or cohere-rocm-triton-attn' \
  --tb=short
```

The test now keeps these as two separate ROCm rows:

- `--attention-backend=TRITON_ATTN`
- `--attention-backend=ROCM_AITER_UNIFIED_ATTN`

## What Was Excluded

- Audio loading / WAV serialization was checked by round-tripping the dataset
  arrays through the same in-memory WAV path used by the API test; samples were
  unchanged.
- Swapping the local Cohere processor path for the HF remote processor did not
  explain the WER drift.
- Changing frontend prompt punctuation modes can alter WER, but that would be
  a behavior change rather than an attention backend correctness fix.

## Suspected Area

The remaining difference is in the ROCm AITER unified encoder-decoder
cross-attention path. The investigation should focus on non-causal paged
attention semantics and any kernel-side handling that differs from the Triton
unified cross-attention path.
