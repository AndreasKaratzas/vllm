# ROCm AITER tiny-mixtral top-k parity intermittently fails on MI355

## Summary

The AMD language-model CI shard intermittently fails vLLM/Hugging Face top-k
parity for `TitanML/tiny-mixtral` on MI355 when ROCm AITER Linear/MoE are
enabled and the existing test disables only the AITER RMSNorm path.

The model is intentionally tiny and has very flat logits, so greedy token
differences are expected. The failure condition is narrower: at one generation
step, Hugging Face's selected token is not present in vLLM's reported top-k.

## Environment

- GPU: AMD MI355 / gfx950
- Model: `TitanML/tiny-mixtral`
- vLLM attention backend selected in the local repro: `ROCM_ATTN`
- vLLM MoE backend selected in the local repro: `ROCm AITER Unquantized MoE`
- Local vLLM version string observed in repro:
  `v0.21.1rc1.dev74+g67f58ce23.d20260518`
- Buildkite failure version string:
  `v0.21.1rc1.dev125+g95b14e2b7`

Environment variables used by the repro:

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_RMSNORM=0
export VLLM_ROCM_USE_SKINNY_GEMM=0
```

`VLLM_ROCM_USE_AITER_LINEAR` and `VLLM_ROCM_USE_AITER_MOE` are intentionally
left unset, so they follow default AITER enablement.

## Buildkite Failure

The failing shard was:

```text
mi355_1: Language Models Tests (Standard)
```

The failing tests were:

```text
models/language/generation/test_common.py::test_models[True-True-5-32-TitanML/tiny-mixtral]
models/language/generation/test_common.py::test_models[False-True-5-32-TitanML/tiny-mixtral]
```

Representative failure from Buildkite:

```text
Matched prefix:
[1562, 10381, 22228, 31360, 8393, 15630, 1343, 11919, 528,
 21687, 1343, 528, 16247, 30887, 1562, 8393, 30887, 1562,
 1562, 30887, 28029, 30887, 1562, 10650, 22397, 30887,
 30887, 30887, 30887, 30887, 13635]

Hugging Face top token:
  id=9833, decoded token="Image", logprob=-7.774135589599609

vLLM top-5:
  id=22397, decoded token="avirus", rank=1, logprob=-7.725228309631348
  id=29024, rank=2, logprob=-7.959603309631348
  id=1334,  rank=3, logprob=-8.084603309631348
  id=16956, rank=4, logprob=-8.115853309631348
  id=18106, rank=5, logprob=-8.162728309631348
```

Hugging Face had vLLM's selected token in its top-k, but vLLM did not have the
Hugging Face selected token in its top-k. That is what violates the parity test.

## Reproduction Commands

Exact full-shard command that reproduced the failure locally once:

```bash
VLLM_TEST_GROUP_NAME=mi355_1-language-models-tests-standard \
VLLM_ALLOW_DEPRECATED_BEAM_SEARCH=1 \
HIP_VISIBLE_DEVICES=5 CUDA_VISIBLE_DEVICES=5 \
pytest -v -s tests/models/language -m 'core_model and (not slow_test)' --tb=short
```

Focused standalone diagnostic script:

```bash
HIP_VISIBLE_DEVICES=5 CUDA_VISIBLE_DEVICES=5 \
python tools/vllm-rocm/aiter_tiny_mixtral_repro.py --repeat 3 --no-fail
```

For failure hunting, run without `--no-fail` and increase repeats:

```bash
HIP_VISIBLE_DEVICES=5 CUDA_VISIBLE_DEVICES=5 \
python tools/vllm-rocm/aiter_tiny_mixtral_repro.py --repeat 50
```

The script mirrors the relevant test settings:

- `TitanML/tiny-mixtral`
- prompts from `tests/prompts/example.txt`
- `max_tokens=32`
- `logprobs=5`
- vLLM `max_model_len=1024`
- vLLM `block_size=16`
- vLLM `enable_chunked_prefill=False`
- ROCm HF SDP guard: flash and mem-efficient SDP disabled, math SDP enabled
- AITER enabled globally, AITER RMSNorm disabled

## Expected Result

Generated tokens may diverge because the model has very flat logits, but each
divergence should remain top-k compatible:

- Hugging Face selected token should appear in vLLM's top-k for that step.
- vLLM selected token should appear in Hugging Face's top-k for that step.

The pytest rows should pass without disabling AITER Linear or AITER MoE.

## Actual Result

The full CI-like language shard intermittently fails because Hugging Face's
selected token is absent from vLLM's top-k for at least one step.

Disabling AITER Linear/MoE makes the row pass, but that avoids the backend under
test and is not a useful fix.

## Local Repro Run Result

Command run locally on MI355:

```bash
HIP_VISIBLE_DEVICES=5 CUDA_VISIBLE_DEVICES=5 \
python tools/vllm-rocm/aiter_tiny_mixtral_repro.py --repeat 3 --no-fail
```

Observed local result:

```text
Environment:
  VLLM_ROCM_USE_AITER=1
  VLLM_ROCM_USE_AITER_LINEAR=<unset>
  VLLM_ROCM_USE_AITER_MOE=<unset>
  VLLM_ROCM_USE_AITER_RMSNORM=0
  VLLM_ROCM_USE_SKINNY_GEMM=0
  HIP_VISIBLE_DEVICES=5
  CUDA_VISIBLE_DEVICES=5

Loaded 8 prompt(s)
Result: PASS
```

The local run produced several generated-token divergences, but all observed
divergences were top-k compatible across 3 repeats. Examples:

```text
Prompt 0, step 12:
  HF token id=8393
  vLLM token id=16247
  HF token in vLLM top-k: True
  vLLM token in HF top-k: True

Prompt 1, step 5:
  HF token id=23431
  vLLM token id=976
  HF token in vLLM top-k: True
  vLLM token in HF top-k: True

Prompt 6, step 24:
  HF token id=3338
  vLLM token id=15648
  HF token in vLLM top-k: True
  vLLM token in HF top-k: True
```

This suggests the failure is intermittent or depends on exact shard ordering,
prior tests, cache state, or warmup/JIT state.

## Notes

- The local repro confirms AITER is active:
  `Using ROCm AITER Unquantized MoE backend`.
- The attention backend falls back to `ROCM_ATTN` for this decoder path.
- The AITER sampler logs that it falls back to PyTorch-native for per-request
  generators, so the suspected surface is more likely AITER Linear/MoE or a
  surrounding ordering/warmup interaction than sampler top-k itself.
- The failure is not fixed by changing the test threshold; it is a backend
  parity issue that should be investigated with the minimal script above.
