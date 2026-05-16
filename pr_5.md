# PR 5: ROCm AITER Eagle Spec Decode

Commit message: `Fix ROCm AITER Eagle spec decode`

## Summary

Fix ROCm AITER Eagle correctness failures by keeping DeepSeek target and drafter MLA backends aligned, and by avoiding duplicate ROCm AITER RMSNorm+quant fusion pattern registration when the `quant_fp8` custom op is disabled.

## Files

| File | Diff |
|---|---|
| `tests/v1/e2e/spec_decode/test_spec_decode.py` | Build `speculative_config` explicitly and pass `attention_backend: TRITON_MLA` to the DeepSeek drafter when the target test path uses the same fallback. |
| `vllm/compilation/passes/fusion/rocm_aiter_fusion.py` | Register both AITER/native quant matcher variants only when `quant_fp8` is enabled; otherwise register the single native variant to avoid duplicate Inductor patterns. |
| `tests/compile/passes/test_fusion.py` | Add a ROCm AITER regression test for constructing the RMSNorm+quant fusion pass with `rms_norm` enabled and `quant_fp8` disabled. |

## Buildkite Test Groups

| Group | Hardware section | Command |
|---|---|---|
| `Spec Decode Eagle` | MI300 single GPU | `pytest -v -s v1/e2e/spec_decode -k "eagle_correctness"` |
| `V1 e2e (4 GPUs)` | MI300 4 GPUs, optional | `pytest -v -s v1/e2e/spec_decode/test_spec_decode.py -k "eagle_correctness_heavy"` |

## Pytest Commands

```bash
cd /app/vllm/tests

env -u VLLM_WORKER_MULTIPROC_METHOD CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.. pytest -q -s \
  'v1/e2e/spec_decode/test_spec_decode.py::test_eagle_correctness_light[ROCM_AITER_FA-deepseek_eagle]'

env -u VLLM_WORKER_MULTIPROC_METHOD CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.. pytest -q -s \
  'v1/e2e/spec_decode/test_spec_decode.py::test_eagle_correctness_medium[ROCM_AITER_FA-qwen3_eagle3]'

env -u VLLM_WORKER_MULTIPROC_METHOD CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.. pytest -q -s \
  'v1/e2e/spec_decode/test_spec_decode.py::test_eagle_correctness_medium[ROCM_AITER_FA-llama3_eagle3]'

env -u VLLM_WORKER_MULTIPROC_METHOD CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.. pytest -q -s \
  'v1/e2e/spec_decode/test_spec_decode.py::test_eagle_correctness_heavy[ROCM_AITER_FA-llama3_eagle]'

cd /app/vllm
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. pytest -q -s \
  tests/compile/passes/test_fusion.py::test_aiter_fusion_rmsnorm_quant_without_quant_fp8_custom_op
```

## Local Notes

`deepseek_eagle` and `qwen3_eagle3` passed locally. The two Llama cases are gated Hugging Face models and this shell has no `HF_TOKEN`, so local reproduction stops at `401 Unauthorized` before vLLM startup.
