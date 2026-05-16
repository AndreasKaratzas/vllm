# PR 2: Qwen GSM8K Startup Budget

Commit message: `Extend Qwen MXFP4 GSM8K startup budget`

## Summary

Increase the startup wait for the Qwen3.5 MXFP4 GSM8K config. Cold starts on MI355 can spend several minutes in torch/AITER compilation before `/health` is available.

## Files

| File | Diff |
|---|---|
| `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml` | Add `startup_max_wait_seconds: 1200`. |

## Buildkite Test Groups

| Group | Hardware section | Command |
|---|---|---|
| `LM Eval Qwen3-5 Models (B200-MI355)` | MI355, 2 GPUs | `pytest -s -v evals/gsm8k/test_gsm8k_correctness.py --config-list-file=configs/models-qwen35-mi355.txt` |

## Pytest Commands

```bash
cd /app/vllm/tests

printf '%s\n' /app/vllm/tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml > /tmp/qwen35-mxfp4.txt

VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONPATH=.. pytest -q -s \
  evals/gsm8k/test_gsm8k_correctness.py \
  --config-list-file=/tmp/qwen35-mxfp4.txt

VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONPATH=.. pytest -s -v \
  evals/gsm8k/test_gsm8k_correctness.py \
  --config-list-file=configs/models-qwen35-mi355.txt
```
