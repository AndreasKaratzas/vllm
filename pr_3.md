# PR 3: Conservative Memory Profiling

Commit message: `Handle VRAM increases during memory profiling`

## Summary

Replace the fatal memory-profiling assertion with conservative accounting. If free VRAM increases during profiling, reserve that increase as non-KV memory so KV cache sizing cannot be inflated by transient ROCm/runtime or process memory release.

## Files

| File | Diff |
|---|---|
| `vllm/v1/worker/gpu_worker.py` | Add `_finalize_memory_profiling_result`; reserve upward free-memory deltas as non-KV memory and warn instead of asserting. |
| `tests/v1/worker/test_worker_memory_snapshot.py` | Add CPU-only unit coverage for normal accounting and increased-free-memory accounting. |

## Buildkite Test Groups

| Group | Hardware section | Command |
|---|---|---|
| `LM Eval Qwen3-5 Models (B200-MI355)` | MI355, 2 GPUs | `pytest -s -v evals/gsm8k/test_gsm8k_correctness.py --config-list-file=configs/models-qwen35-mi355.txt` |
| `V1 Core + KV + Metrics` | MI355, MI300, MI250 | `pytest -v -s v1/worker` |
| `Distributed Tests (8xH100-8xMI300)` | MI300, 8 GPUs | `torchrun --nproc-per-node=8 ../examples/features/torchrun/torchrun_dp_example_offline.py --tp-size=2 --pp-size=1 --dp-size=4 --enable-ep` |

## Pytest Commands

```bash
cd /app/vllm
PYTHONPATH=. pytest -q \
  tests/v1/worker/test_worker_memory_snapshot.py::test_memory_profile_free_increase_is_reserved_as_non_kv_memory \
  tests/v1/worker/test_worker_memory_snapshot.py::test_memory_profile_without_free_increase_preserves_accounting

cd /app/vllm/tests
printf '%s\n' /app/vllm/tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml > /tmp/qwen35-mxfp4.txt

VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONPATH=.. pytest -q -s \
  evals/gsm8k/test_gsm8k_correctness.py \
  --config-list-file=/tmp/qwen35-mxfp4.txt
```

## Cold AITER Repro Prep

```bash
find /usr/local/lib/python3.12/dist-packages/aiter/jit -maxdepth 3 \
  \( -name 'module_moe_ck2stages_fp4x2_fp4x2_preshuffle_on_b16_silu_per_1x32_mulWeightStage2_*' \
     -o -name 'lock_module_moe_ck2stages_fp4x2_fp4x2_preshuffle_on_b16_silu_per_1x32_mulWeightStage2_' \) \
  -print -exec rm -rf {} +
```
