# PR 1: Uniproc Distributed Shutdown

Commit message: `Destroy distributed state during uniproc shutdown`

## Summary

Fix `UniProcExecutor.shutdown()` so it tears down vLLM model-parallel and distributed state after shutting down the driver worker. This addresses single-process V1 engine cleanup paths that can leave NCCL/RCCL process groups alive until interpreter exit.

## Files

| File | Diff |
|---|---|
| `vllm/v1/executor/uniproc_executor.py` | Import `destroy_model_parallel` and `destroy_distributed_environment`; call both in a `finally` block during shutdown. |
| `tests/v1/executor/test_executor.py` | Add a unit test proving worker shutdown is followed by model-parallel and distributed teardown. |

## Buildkite Test Groups

| Group | Hardware section | Command |
|---|---|---|
| `Language Models Tests (Standard)` | MI355 and MI300 | `pytest -v -s models/language -m 'core_model and (not slow_test)'` |
| `V1 Core + KV + Metrics` | MI355, MI300, MI250 | `pytest -v -s v1/executor` |
| `Entrypoints Integration (Pooling)` | MI355 and MI300 | `pytest -v -s entrypoints/pooling` |

## Pytest Commands

```bash
PYTHONPATH=. pytest -q tests/v1/executor/test_executor.py::test_uniproc_executor_shutdown_destroys_distributed

cd /app/vllm/tests
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.. pytest -q -s 'models/language/generation/test_common.py::test_models[False-False-5-32-openai-community/gpt2]'
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.. pytest -q -s 'models/language/generation/test_common.py::test_models[True-True-5-32-TitanML/tiny-mixtral]'

PYTHONPATH=.. pytest -q -s entrypoints/pooling/embed/test_correctness_mteb.py::test_mteb_embed
PYTHONPATH=.. pytest -q -s entrypoints/pooling/scoring/test_cross_encoder_online_vision.py -k TRITON_ATTN
```
