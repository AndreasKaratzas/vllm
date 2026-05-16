# Consolidated ROCm CI Regression Fixes

Commit message: `Fix ROCm CI regressions across workers, kernels, and tests`

## Summary

This file consolidates the five existing PR-sized notes plus the current local
diff inventory. The work targets the new ROCm CI regressions seen across V1
worker startup/shutdown, Qwen3.5 GSM8K TP2 startup, Granite tool calls, AITER
Eagle spec decode, FP8 kernel numerical checks, spawned-worker logits processor
tests, and model initialization smoke coverage.

Use `scripts/run_rocm_regression_groups.sh` for local validation. Every mode
writes one log per test group plus `summary.tsv` under `raw_logs/<timestamp>/`.
Use `bk-list` to print the Buildkite-shaped affected groups before running
them.

## Existing PR Bundles

| PR note | Commit message | Primary regression area |
|---|---|---|
| `pr_1.md` | `Destroy distributed state during uniproc shutdown` | Language model / pooling hangs from lingering distributed state. |
| `pr_2.md` | `Extend Qwen MXFP4 GSM8K startup budget` | Cold Qwen3.5 MXFP4 TP2 startup exceeding health-check budget. |
| `pr_3.md` | `Handle VRAM increases during memory profiling` | Memory profiling assertion when free VRAM increases during warmup. |
| `pr_4.md` | `Guide Granite tool-call formatting` | Granite 3.0 returning no parsed tool calls. |
| `pr_5.md` | `Fix ROCm AITER Eagle spec decode` | ROCm AITER Eagle spec decode correctness and fusion setup. |

## Current Diff Inventory

| File | Diff |
|---|---|
| `scripts/run_rocm_regression_groups.sh` | Add structured ROCm regression runner with `quick`, `full`, `long`, `all`, `pr1` through `pr5`, and `prs` modes; stores logs under `raw_logs`. |
| `tests/compile/passes/test_fusion.py` | Add ROCm AITER regression coverage for constructing RMSNorm+quant fusion when `rms_norm` is enabled and `quant_fp8` is disabled. |
| `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml` | Add `startup_max_wait_seconds: 1200` for cold torch/AITER compilation. |
| `tests/kernels/core/test_layernorm.py` | Compare fused RMSNorm FP8 quant output with FP8-aware helper instead of exact float equality. |
| `tests/kernels/core/test_rotary_embedding_mla_cache_fused.py` | Compare FP8 cache payloads by FP8 code drift and relax non-FP8 cache tolerance. |
| `tests/kernels/core/test_vit_fp8_quant.py` | Use FP8-aware comparisons for contiguous, non-contiguous, and skip-scale ViT quant cases. |
| `tests/kernels/quant_utils.py` | Add `assert_fp8_codes_close` and `assert_fp8_quant_close` helpers for sparse adjacent-code drift. |
| `tests/models/registry.py` | Point gated Gemma3n and Llama4 example models at accessible `unsloth` mirrors. |
| `tests/models/test_initialization.py` | Disable V1 multiprocessing inside initialization subprocesses so KV-cache monkeypatching is effective. |
| `tests/models/utils.py` | Preserve single-expert MoE configs in dummy HF overrides and support `num_experts_per_token`. |
| `tests/tool_use/test_tool_calls.py` | Apply model-specific system prompts to tool-call and tool-result tests. |
| `tests/tool_use/utils.py` | Add Granite 3.0 system prompt that requests `<\|tool_call\|>` JSON output only. |
| `tests/utils.py` | Include subprocess stdout and stderr in `spawn_new_process_for_each_test` failures. |
| `tests/utils_/test_spawn_decorator.py` | Add regression coverage proving spawned-test failure output includes child stdout/stderr. |
| `tests/v1/e2e/spec_decode/test_spec_decode.py` | Build speculative config explicitly and align DeepSeek drafter MLA backend with target fallback. |
| `tests/v1/executor/test_executor.py` | Add unit coverage for UniProcExecutor distributed teardown on shutdown. |
| `tests/v1/logits_processors/test_custom_offline.py` | Use a discoverable dummy dist-info entrypoint instead of process-local metadata monkeypatching. |
| `tests/v1/logits_processors/test_custom_online.py` | Make spawned servers see dummy entrypoints via `PYTHONPATH`, clear logits-processor cache, and cover both entrypoint and FQCN paths. |
| `tests/v1/logits_processors/utils.py` | Add `install_dummy_entrypoint` helper that creates a temporary dist-info entrypoint visible to spawned workers. |
| `tests/v1/worker/test_worker_memory_snapshot.py` | Add CPU-only coverage for normal memory profiling and increased-free-memory conservative accounting. |
| `vllm/v1/executor/uniproc_executor.py` | Destroy model-parallel and distributed environments in `UniProcExecutor.shutdown()` after worker shutdown. |
| `vllm/v1/worker/gpu_worker.py` | Replace fatal free-memory-increase assertion with conservative non-KV memory accounting and a warning. |

## Supporting Docs

| File | Purpose |
|---|---|
| `pr_1.md` | Split-PR note for uniproc distributed shutdown. |
| `pr_2.md` | Split-PR note for Qwen3.5 GSM8K startup budget. |
| `pr_3.md` | Split-PR note for conservative memory profiling. |
| `pr_4.md` | Split-PR note for Granite tool-call formatting. |
| `pr_5.md` | Split-PR note for ROCm AITER Eagle spec decode. |
| `pr_all.md` | Consolidated note for all current diffs. |

## Buildkite Groups To See Green

| Buildkite group | Covered by local mode | Notes |
|---|---|---|
| `Language Models Tests (Standard)` | `pr1`, partial `quick` | Language generation smoke and cleanup validation. |
| `Entrypoints Integration (Pooling)` | `pr1` | MTEB embedding and TRITON attention vision scoring smoke. |
| `LM Eval Qwen3.5 Models B200/MI355` | `long`, `pr2`, `pr3` | Requires two visible GPUs for TP2. |
| `Async Engine / Inputs / Utils / Worker` | `quick`, `pr3` | Memory profiling accounting tests. |
| `V1 Core + KV + Metrics` | `quick`, `pr1` | Executor shutdown coverage. |
| `Entrypoints Integration (API Server 2)` | `pr4` | Granite tool-call tests. |
| `Spec Decode Eagle` | `long`, `pr5` | Eagle correctness under ROCm AITER FA. |
| `PyTorch Compilation Passes Unit Tests` | `quick`, `pr5` | RMSNorm+quant fusion pass construction. |
| `Kernels Core Operation Test` | `quick`, `full` | FP8 drift-sensitive kernel checks. |
| `V1 Sample + Logits` | `quick`, `full` | Spawn-safe custom logits processor coverage. |
| `Basic Models Tests (Initialization)` | `quick` | Small initialization subset. |
| `Basic Models Tests (Extra Initialization)` | `quick`, `full` | Arctic spotcheck plus optional shard runs. |

## Local Runner Commands

```bash
cd /app/vllm

# Fast affected subset, single GPU unless overridden.
HIP_VISIBLE_DEVICES=0 scripts/run_rocm_regression_groups.sh quick

# Broader affected files and model-initialization shards.
HIP_VISIBLE_DEVICES=0 scripts/run_rocm_regression_groups.sh full

# Slow model-serving checks. TP2 GSM8K needs two visible GPUs.
CLEAR_AITER_CACHE=1 HIP_VISIBLE_DEVICES=0,1 \
  scripts/run_rocm_regression_groups.sh long

# Run the five existing PR bundles as separate named groups.
CLEAR_AITER_CACHE=1 HIP_VISIBLE_DEVICES=0,1 \
  scripts/run_rocm_regression_groups.sh prs

# Print the affected Buildkite groups and exact CI command lists.
scripts/run_rocm_regression_groups.sh bk-list

# Run the MI355 affected Buildkite groups locally.
CLEAR_AITER_CACHE=1 HIP_VISIBLE_DEVICES=0,1 \
  scripts/run_rocm_regression_groups.sh bk-mi355

# Run all affected Buildkite-shaped groups locally.
CLEAR_AITER_CACHE=1 HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  scripts/run_rocm_regression_groups.sh bk-affected
```

## PR-Specific Runner Modes

| Mode | What it runs |
|---|---|
| `pr1` | Uniproc shutdown unit, GPT-2 language smoke, pooling MTEB and TRITON attention smoke. |
| `pr2` | Qwen3.5 MXFP4 GSM8K TP2 startup with optional AITER cache clear. |
| `pr3` | Memory profiling unit tests plus Qwen3.5 MXFP4 GSM8K TP2 repro path. |
| `pr4` | Granite tool-call and parallel tool-call tests. |
| `pr5` | ROCm AITER RMSNorm+quant fusion test and Eagle correctness tests. |
| `prs` | Runs `pr1` through `pr5` in order. |
| `bk-list` | Prints affected Buildkite groups and command lists without running tests. |
| `bk-mi355` | Runs the affected MI355 Buildkite command lists. |
| `bk-mi300` | Runs the affected MI300 Buildkite command lists. |
| `bk-model-init` | Runs the Buildkite model initialization groups, including both local shards. |
| `bk-distributed` | Runs the 8-GPU distributed Buildkite command. |
| `bk-affected` | Runs all affected Buildkite-shaped groups. |

## Patch Stat

```text
 tests/compile/passes/test_fusion.py                | 31 +++++++++++
 .../gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-TP2.yaml   |  1 +
 tests/kernels/core/test_layernorm.py               |  8 +--
 .../core/test_rotary_embedding_mla_cache_fused.py  | 16 ++----
 tests/kernels/core/test_vit_fp8_quant.py           | 15 ++++--
 tests/kernels/quant_utils.py                       | 62 ++++++++++++++++++++++
 tests/models/registry.py                           |  6 +--
 tests/models/test_initialization.py                |  4 ++
 tests/models/utils.py                              |  8 ++-
 tests/tool_use/test_tool_calls.py                  | 18 ++++---
 tests/tool_use/utils.py                            |  7 +++
 tests/utils.py                                     | 15 ++++--
 tests/utils_/test_spawn_decorator.py               | 17 ++++++
 tests/v1/e2e/spec_decode/test_spec_decode.py       | 18 ++++---
 tests/v1/executor/test_executor.py                 | 24 +++++++++
 tests/v1/logits_processors/test_custom_offline.py  | 19 +++----
 tests/v1/logits_processors/test_custom_online.py   | 59 ++++++++++++++------
 tests/v1/logits_processors/utils.py                | 40 ++++++++++++++
 tests/v1/worker/test_worker_memory_snapshot.py     | 59 +++++++++++++++++++-
 vllm/v1/executor/uniproc_executor.py               |  9 +++-
 vllm/v1/worker/gpu_worker.py                       | 61 +++++++++++++++------
 21 files changed, 410 insertions(+), 87 deletions(-)
```

## Notes

`raw_logs/` is local test output and should stay out of the PR unless a specific
log excerpt is needed for debugging. For cold AITER repros, set
`CLEAR_AITER_CACHE=1`; for Qwen3.5 TP2, expose two GPUs with
`HIP_VISIBLE_DEVICES=0,1`.
