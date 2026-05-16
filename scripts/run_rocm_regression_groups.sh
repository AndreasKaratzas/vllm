#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Run the ROCm regression groups touched by the recent CI fixes and keep
# timestamped local logs under raw_logs/.
#
# Usage:
#   scripts/run_rocm_regression_groups.sh quick
#   scripts/run_rocm_regression_groups.sh full
#   scripts/run_rocm_regression_groups.sh long
#   scripts/run_rocm_regression_groups.sh all
#   scripts/run_rocm_regression_groups.sh pr1
#   scripts/run_rocm_regression_groups.sh pr2
#   scripts/run_rocm_regression_groups.sh pr3
#   scripts/run_rocm_regression_groups.sh pr4
#   scripts/run_rocm_regression_groups.sh pr5
#   scripts/run_rocm_regression_groups.sh prs
#   scripts/run_rocm_regression_groups.sh bk-list
#   scripts/run_rocm_regression_groups.sh bk-mi355
#   scripts/run_rocm_regression_groups.sh bk-mi300
#   scripts/run_rocm_regression_groups.sh bk-model-init
#   scripts/run_rocm_regression_groups.sh bk-distributed
#   scripts/run_rocm_regression_groups.sh bk-affected
#
# Useful environment variables:
#   RAW_LOG_ROOT=raw_logs/rocm        Where logs are written.
#   HIP_VISIBLE_DEVICES=0            GPU selection for single-GPU checks.
#   HIP_VISIBLE_DEVICES=0,1          GPU selection for TP2 checks.
#   CLEAR_AITER_CACHE=1              Clear AITER JIT cache before long evals.

set -uo pipefail

MODE="${1:-quick}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RAW_LOG_ROOT="${RAW_LOG_ROOT:-${ROOT_DIR}/raw_logs}"
LOG_DIR="${RAW_LOG_ROOT}/${RUN_ID}"
SUMMARY_FILE="${LOG_DIR}/summary.tsv"

mkdir -p "${LOG_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export VLLM_ALLOW_DEPRECATED_BEAM_SEARCH="${VLLM_ALLOW_DEPRECATED_BEAM_SEARCH:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

if [[ -z "${HIP_VISIBLE_DEVICES:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES=0
fi

cat >"${LOG_DIR}/env.txt" <<EOF
run_id=${RUN_ID}
mode=${MODE}
root_dir=${ROOT_DIR}
raw_log_root=${RAW_LOG_ROOT}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-}
VLLM_ALLOW_DEPRECATED_BEAM_SEARCH=${VLLM_ALLOW_DEPRECATED_BEAM_SEARCH:-}
PYTHONPATH=${PYTHONPATH:-}
EOF

printf "status\tseconds\tgroup\tlog\n" >"${SUMMARY_FILE}"

FAILURES=0

usage() {
  sed -n '1,36p' "$0"
}

run_group() {
  local group="$1"
  local buildkite_label="$2"
  local description="$3"
  local command="$4"
  local log_file="${LOG_DIR}/${group}.log"
  local start
  local status
  local elapsed

  start="$(date +%s)"

  {
    echo "================================================================================"
    echo "Group: ${group}"
    echo "Buildkite label: ${buildkite_label}"
    echo "Description: ${description}"
    echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Working directory: ${ROOT_DIR}"
    echo "Command:"
    echo "  ${command}"
    echo "================================================================================"
    cd "${ROOT_DIR}"
    eval "${command}"
  } 2>&1 | tee "${log_file}"

  status="${PIPESTATUS[0]}"
  elapsed="$(($(date +%s) - start))"

  if [[ "${status}" -eq 0 ]]; then
    printf "PASS\t%s\t%s\t%s\n" "${elapsed}" "${group}" "${log_file}" \
      | tee -a "${SUMMARY_FILE}"
  else
    printf "FAIL(%s)\t%s\t%s\t%s\n" "${status}" "${elapsed}" "${group}" \
      "${log_file}" | tee -a "${SUMMARY_FILE}"
    FAILURES=$((FAILURES + 1))
  fi
}

clear_aiter_cache_if_requested() {
  if [[ "${CLEAR_AITER_CACHE:-0}" == "1" ]]; then
    echo "Clearing AITER JIT cache before long ROCm evals"
    rm -rf /usr/local/lib/python3.12/dist-packages/aiter/jit/build/* \
      /usr/local/lib/python3.12/dist-packages/aiter/jit/*.so
  fi
}

run_quick_groups() {
  run_group \
    "spawn-decorator-diagnostics" \
    "supporting test utility" \
    "Verifies spawned test failures include child stdout/stderr." \
    "pytest -v -s tests/utils_/test_spawn_decorator.py"

  run_group \
    "basic-models-initialization-small" \
    "Basic Models Tests (Initialization)" \
    "Runs the small initialization subset that exposed Llama4/Gemma3n issues." \
    "pytest -v -s tests/models/test_initialization.py::test_can_initialize_small_subset"

  run_group \
    "basic-models-initialization-large-spotcheck" \
    "Basic Models Tests (Extra Initialization) %N" \
    "Spot-checks an extra-initialization MoE model that failed via profile/compile." \
    "pytest -v -s 'tests/models/test_initialization.py::test_can_initialize_large_subset[ArcticForCausalLM]'"

  run_group \
    "v1-logits-processors" \
    "V1 Sample + Logits" \
    "Runs the custom logits processor entrypoint/server cases fixed for spawned workers." \
    "pytest -v -s tests/v1/logits_processors/test_custom_offline.py::test_custom_logitsprocs tests/v1/logits_processors/test_custom_offline.py::test_rejects_custom_logitsprocs tests/v1/logits_processors/test_custom_online.py::test_custom_logitsprocs tests/v1/logits_processors/test_custom_online.py::test_invalid_custom_logitsproc_arg"

  run_group \
    "kernels-core-fp8-focused" \
    "Kernels Core Operation Test" \
    "Runs the FP8 numerical drift cases: layernorm, ViT FP8 quant, and one MLA rope/cache FP8 representative." \
    "pytest -v -s tests/kernels/core/test_layernorm.py tests/kernels/core/test_vit_fp8_quant.py 'tests/kernels/core/test_rotary_embedding_mla_cache_fused.py::test_concat_and_cache_mla_rope_fused[cuda:0-0-64-64-512-fp8-128-128-42-False-dtype0]'"

  run_group \
    "compile-passes-fusion-focused" \
    "PyTorch Compilation Passes Unit Tests" \
    "Runs the ROCm AITER RMSNorm+quant custom-op regression test." \
    "pytest -v -s tests/compile/passes/test_fusion.py::test_aiter_fusion_rmsnorm_quant_without_quant_fp8_custom_op"

  run_group \
    "v1-worker-memory-profiling" \
    "V1 Core + KV + Metrics" \
    "Runs memory profiling regression tests for increased free-memory handling." \
    "pytest -v -s tests/v1/worker/test_worker_memory_snapshot.py"

  run_group \
    "v1-executor-shutdown" \
    "V1 Core + KV + Metrics" \
    "Runs uniprocess executor shutdown cleanup coverage." \
    "pytest -v -s tests/v1/executor/test_executor.py"
}

run_full_groups() {
  run_group \
    "kernels-core-affected-files-full" \
    "Kernels Core Operation Test" \
    "Runs all affected kernel-core files, including the full MLA rope/cache matrix." \
    "pytest -v -s tests/kernels/core/test_layernorm.py tests/kernels/core/test_vit_fp8_quant.py tests/kernels/core/test_rotary_embedding_mla_cache_fused.py"

  run_group \
    "v1-logits-processors-full" \
    "V1 Sample + Logits" \
    "Runs the full logits processor directory from the Buildkite group." \
    "pytest -v -s tests/v1/logits_processors"

  run_group \
    "basic-models-extra-initialization-shard0" \
    "Basic Models Tests (Extra Initialization) %N" \
    "Runs shard 0 of the long extra initialization group." \
    "pytest -v -s tests/models/test_initialization.py -k 'not test_can_initialize_small_subset' --num-shards=2 --shard-id=0"

  run_group \
    "basic-models-extra-initialization-shard1" \
    "Basic Models Tests (Extra Initialization) %N" \
    "Runs shard 1 of the long extra initialization group." \
    "pytest -v -s tests/models/test_initialization.py -k 'not test_can_initialize_small_subset' --num-shards=2 --shard-id=1"
}

run_long_groups() {
  run_group \
    "spec-decode-eagle" \
    "Spec Decode Eagle" \
    "Runs the full Eagle correctness group." \
    "pytest -v -s tests/v1/e2e/spec_decode/test_spec_decode.py -k 'eagle_correctness'"

  clear_aiter_cache_if_requested
  run_group \
    "gsm8k-qwen35-mxfp4-tp2" \
    "LM Eval Qwen3.5 Models B200/MI355" \
    "Runs the Qwen3.5 MXFP4 GSM8K config; set CLEAR_AITER_CACHE=1 to force AITER rebuild." \
    "pytest -v -s tests/evals/gsm8k/test_gsm8k_correctness.py --config-list-file=tests/evals/gsm8k/configs/models-qwen35-mi355.txt"
}

run_pr1_groups() {
  run_group \
    "pr1-uniproc-shutdown" \
    "V1 Core + KV + Metrics" \
    "PR 1: proves UniProcExecutor shutdown destroys model-parallel and distributed state." \
    "pytest -v -s tests/v1/executor/test_executor.py::test_uniproc_executor_shutdown_destroys_distributed"

  run_group \
    "pr1-language-gpt2-smoke" \
    "Language Models Tests (Standard)" \
    "PR 1: smoke test for the language-model group that was hanging during cleanup." \
    "pytest -v -s 'tests/models/language/generation/test_common.py::test_models[False-False-5-32-openai-community/gpt2]'"

  run_group \
    "pr1-pooling-smoke" \
    "Entrypoints Integration (Pooling)" \
    "PR 1: smoke tests for pooling entrypoints that can leave distributed state alive." \
    "pytest -v -s tests/entrypoints/pooling/embed/test_correctness_mteb.py::test_mteb_embed tests/entrypoints/pooling/scoring/test_cross_encoder_online_vision.py -k TRITON_ATTN"
}

run_pr2_groups() {
  clear_aiter_cache_if_requested
  run_group \
    "pr2-qwen35-gsm8k-startup" \
    "LM Eval Qwen3.5 Models B200/MI355" \
    "PR 2: validates the extended startup budget for the Qwen3.5 MXFP4 TP2 server." \
    "pytest -v -s tests/evals/gsm8k/test_gsm8k_correctness.py --config-list-file=tests/evals/gsm8k/configs/models-qwen35-mi355.txt"
}

run_pr3_groups() {
  run_group \
    "pr3-memory-profiling-unit" \
    "Async Engine / Inputs / Utils / Worker" \
    "PR 3: validates conservative memory accounting when free VRAM increases during profiling." \
    "pytest -v -s tests/v1/worker/test_worker_memory_snapshot.py::test_memory_profile_free_increase_is_reserved_as_non_kv_memory tests/v1/worker/test_worker_memory_snapshot.py::test_memory_profile_without_free_increase_preserves_accounting"

  clear_aiter_cache_if_requested
  run_group \
    "pr3-qwen35-gsm8k-memory-profile" \
    "LM Eval Qwen3.5 Models B200/MI355" \
    "PR 3: reproduces the TP2 startup path that hit the free-memory increase assertion." \
    "pytest -v -s tests/evals/gsm8k/test_gsm8k_correctness.py --config-list-file=tests/evals/gsm8k/configs/models-qwen35-mi355.txt"
}

run_pr4_groups() {
  run_group \
    "pr4-granite-tool-calls" \
    "Entrypoints Integration (API Server 2)" \
    "PR 4: checks Granite emits the expected tool-call format." \
    "pytest -v -s tests/tool_use/test_tool_calls.py --models granite-3.0-8b"

  run_group \
    "pr4-granite-parallel-tool-calls" \
    "Entrypoints Integration (API Server 2)" \
    "PR 4: checks Granite parallel tool-call coverage with the same system prompt." \
    "pytest -v -s tests/tool_use/test_parallel_tool_calls.py --models granite-3.0-8b"
}

run_pr5_groups() {
  run_group \
    "pr5-aiter-rmsnorm-quant-fusion" \
    "PyTorch Compilation Passes Unit Tests" \
    "PR 5: validates ROCm AITER RMSNorm+quant pass construction without quant_fp8." \
    "pytest -v -s tests/compile/passes/test_fusion.py::test_aiter_fusion_rmsnorm_quant_without_quant_fp8_custom_op"

  run_group \
    "pr5-spec-decode-eagle" \
    "Spec Decode Eagle" \
    "PR 5: runs the Eagle correctness cases that regressed on ROCm AITER FA." \
    "pytest -v -s tests/v1/e2e/spec_decode/test_spec_decode.py -k 'eagle_correctness and ROCM_AITER_FA'"
}

run_pr_groups() {
  run_pr1_groups
  run_pr2_groups
  run_pr3_groups
  run_pr4_groups
  run_pr5_groups
}

print_buildkite_affected_groups() {
  cat <<'EOF'
Affected Buildkite groups mirrored by this script:

MI355 groups:
  Entrypoints Integration (API Server 2)
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    pytest -v -s entrypoints/serve/instrumentator
    PYTHONPATH=/vllm-workspace pytest -v -s entrypoints/rpc
    pytest -v -s tool_use

  Entrypoints Integration (Pooling)
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    pytest -v -s entrypoints/pooling

  LM Eval Qwen3-5 Models (B200-MI355)
    pytest -s -v evals/gsm8k/test_gsm8k_correctness.py --config-list-file=configs/models-qwen35-mi355.txt

  Language Models Tests (Standard)
    pip freeze | grep -E 'torch'
    pytest -v -s models/language -m 'core_model and (not slow_test)'

  V1 Core + KV + Metrics
    uv pip install --system -r /vllm-workspace/requirements/kv_connectors_rocm.txt
    pytest -v -s -m 'not cpu_test' v1/core
    pytest -v -s v1/executor
    pytest -v -s v1/kv_offload
    pytest -v -s v1/worker
    pytest -v -s -m 'not cpu_test' v1/kv_connector/unit
    pytest -v -s -m 'not cpu_test' v1/metrics
    pip install -U git+https://github.com/robertgshaw2-redhat/lm-evaluation-harness.git@streaming-api
    pytest -v -s entrypoints/openai/correctness/test_lmeval.py::test_lm_eval_accuracy_v1_engine

  V1 Sample + Logits
    pytest -v -s v1/sample
    pytest -v -s v1/logits_processors
    pytest -v -s v1/test_oracle.py
    pytest -v -s v1/test_request.py
    pytest -v -s v1/test_outputs.py

MI300 groups:
  Async Engine, Inputs, Utils, Worker
    pytest -v -s detokenizer
    pytest -v -s -m 'not cpu_test' multimodal
    pytest -v -s utils_

  Kernels Core Operation Test
    pytest -v -s kernels/core --ignore=kernels/core/test_minimax_reduce_rms.py  kernels/test_concat_mla_q.py kernels/test_top_k_per_row.py

  PyTorch Compilation Passes Unit Tests
    pytest -s -v compile/passes --ignore compile/passes/distributed

  Spec Decode Eagle
    pytest -v -s v1/e2e/spec_decode -k "eagle_correctness"

  V1 Core + KV + Metrics
    same command list as MI355 V1 Core + KV + Metrics

  V1 Sample + Logits
    same command list as MI355 V1 Sample + Logits

Model initialization groups:
  Basic Models Tests (Initialization)
    pytest -v -s models/test_initialization.py::test_can_initialize_small_subset

  Basic Models Tests (Extra Initialization) %N
    pytest -v -s models/test_initialization.py -k 'not test_can_initialize_small_subset' --num-shards=$BUILDKITE_PARALLEL_JOB_COUNT --shard-id=$BUILDKITE_PARALLEL_JOB

Distributed group:
  Distributed Tests (8xH100-8xMI300)
    torchrun --nproc-per-node=8 ../examples/features/torchrun/torchrun_dp_example_offline.py --tp-size=2 --pp-size=1 --dp-size=4 --enable-ep
EOF
}

run_buildkite_api_server_2_group() {
  run_group \
    "bk-entrypoints-api-server-2" \
    "Entrypoints Integration (API Server 2)" \
    "Buildkite command list for API Server 2, adapted to the local checkout path." \
    "cd tests && export VLLM_WORKER_MULTIPROC_METHOD=spawn && pytest -v -s entrypoints/serve/instrumentator && PYTHONPATH=${ROOT_DIR} pytest -v -s entrypoints/rpc && pytest -v -s tool_use"
}

run_buildkite_pooling_group() {
  run_group \
    "bk-entrypoints-pooling" \
    "Entrypoints Integration (Pooling)" \
    "Buildkite command list for pooling entrypoint integration." \
    "cd tests && export VLLM_WORKER_MULTIPROC_METHOD=spawn && pytest -v -s entrypoints/pooling"
}

run_buildkite_qwen35_group() {
  clear_aiter_cache_if_requested
  run_group \
    "bk-lm-eval-qwen35-mi355" \
    "LM Eval Qwen3-5 Models (B200-MI355)" \
    "Buildkite command list for Qwen3.5 GSM8K TP2 eval." \
    "cd tests && pytest -s -v evals/gsm8k/test_gsm8k_correctness.py --config-list-file=configs/models-qwen35-mi355.txt"
}

run_buildkite_language_standard_group() {
  run_group \
    "bk-language-models-standard" \
    "Language Models Tests (Standard)" \
    "Buildkite command list for standard core language-model tests." \
    "cd tests && pip freeze | grep -E 'torch' && pytest -v -s models/language -m 'core_model and (not slow_test)'"
}

run_buildkite_v1_core_group() {
  run_group \
    "bk-v1-core-kv-metrics" \
    "V1 Core + KV + Metrics" \
    "Buildkite command list for V1 core, executor, worker, KV, metrics, and lmeval coverage." \
    "cd tests && uv pip install --system -r ${ROOT_DIR}/requirements/kv_connectors_rocm.txt && pytest -v -s -m 'not cpu_test' v1/core && pytest -v -s v1/executor && pytest -v -s v1/kv_offload && pytest -v -s v1/worker && pytest -v -s -m 'not cpu_test' v1/kv_connector/unit && pytest -v -s -m 'not cpu_test' v1/metrics && pip install -U git+https://github.com/robertgshaw2-redhat/lm-evaluation-harness.git@streaming-api && pytest -v -s entrypoints/openai/correctness/test_lmeval.py::test_lm_eval_accuracy_v1_engine"
}

run_buildkite_v1_sample_logits_group() {
  run_group \
    "bk-v1-sample-logits" \
    "V1 Sample + Logits" \
    "Buildkite command list for V1 sample, logits processors, oracle, request, and outputs." \
    "cd tests && pytest -v -s v1/sample && pytest -v -s v1/logits_processors && pytest -v -s v1/test_oracle.py && pytest -v -s v1/test_request.py && pytest -v -s v1/test_outputs.py"
}

run_buildkite_async_utils_group() {
  run_group \
    "bk-async-engine-inputs-utils-worker" \
    "Async Engine, Inputs, Utils, Worker" \
    "Buildkite command list for detokenizer, multimodal, and utils_." \
    "cd tests && pytest -v -s detokenizer && pytest -v -s -m 'not cpu_test' multimodal && pytest -v -s utils_"
}

run_buildkite_kernels_core_group() {
  run_group \
    "bk-kernels-core-operation" \
    "Kernels Core Operation Test" \
    "Buildkite command list for kernels/core and adjacent kernel tests." \
    "cd tests && pytest -v -s kernels/core --ignore=kernels/core/test_minimax_reduce_rms.py kernels/test_concat_mla_q.py kernels/test_top_k_per_row.py"
}

run_buildkite_compile_passes_group() {
  run_group \
    "bk-pytorch-compilation-passes" \
    "PyTorch Compilation Passes Unit Tests" \
    "Buildkite command list for compile/passes, excluding distributed tests." \
    "cd tests && pytest -s -v compile/passes --ignore compile/passes/distributed"
}

run_buildkite_spec_decode_eagle_group() {
  run_group \
    "bk-spec-decode-eagle" \
    "Spec Decode Eagle" \
    "Buildkite command list for Eagle correctness." \
    "cd tests && pytest -v -s v1/e2e/spec_decode -k 'eagle_correctness'"
}

run_buildkite_model_init_groups() {
  run_group \
    "bk-basic-models-initialization" \
    "Basic Models Tests (Initialization)" \
    "Buildkite command list for the small model initialization subset." \
    "cd tests && pytest -v -s models/test_initialization.py::test_can_initialize_small_subset"

  run_group \
    "bk-basic-models-extra-initialization-shard0" \
    "Basic Models Tests (Extra Initialization) %N" \
    "Local shard 0 equivalent of the Buildkite parallel model initialization group." \
    "cd tests && pytest -v -s models/test_initialization.py -k 'not test_can_initialize_small_subset' --num-shards=2 --shard-id=0"

  run_group \
    "bk-basic-models-extra-initialization-shard1" \
    "Basic Models Tests (Extra Initialization) %N" \
    "Local shard 1 equivalent of the Buildkite parallel model initialization group." \
    "cd tests && pytest -v -s models/test_initialization.py -k 'not test_can_initialize_small_subset' --num-shards=2 --shard-id=1"
}

run_buildkite_distributed_groups() {
  run_group \
    "bk-distributed-8gpu" \
    "Distributed Tests (8xH100-8xMI300)" \
    "Buildkite command list for the 8-GPU torchrun DP/TP/EP example." \
    "cd tests && torchrun --nproc-per-node=8 ../examples/features/torchrun/torchrun_dp_example_offline.py --tp-size=2 --pp-size=1 --dp-size=4 --enable-ep"
}

run_buildkite_mi355_groups() {
  run_buildkite_api_server_2_group
  run_buildkite_pooling_group
  run_buildkite_qwen35_group
  run_buildkite_language_standard_group
  run_buildkite_v1_core_group
  run_buildkite_v1_sample_logits_group
}

run_buildkite_mi300_groups() {
  run_buildkite_async_utils_group
  run_buildkite_kernels_core_group
  run_buildkite_compile_passes_group
  run_buildkite_spec_decode_eagle_group
  run_buildkite_v1_core_group
  run_buildkite_v1_sample_logits_group
  run_buildkite_language_standard_group
}

run_buildkite_affected_groups() {
  run_buildkite_mi355_groups
  run_buildkite_mi300_groups
  run_buildkite_model_init_groups
  run_buildkite_distributed_groups
}

case "${MODE}" in
  quick)
    run_quick_groups
    ;;
  full)
    run_quick_groups
    run_full_groups
    ;;
  long)
    run_long_groups
    ;;
  all)
    run_quick_groups
    run_full_groups
    run_long_groups
    ;;
  pr1)
    run_pr1_groups
    ;;
  pr2)
    run_pr2_groups
    ;;
  pr3)
    run_pr3_groups
    ;;
  pr4)
    run_pr4_groups
    ;;
  pr5)
    run_pr5_groups
    ;;
  prs)
    run_pr_groups
    ;;
  bk-list)
    print_buildkite_affected_groups
    exit 0
    ;;
  bk-mi355)
    run_buildkite_mi355_groups
    ;;
  bk-mi300)
    run_buildkite_mi300_groups
    ;;
  bk-model-init)
    run_buildkite_model_init_groups
    ;;
  bk-distributed)
    run_buildkite_distributed_groups
    ;;
  bk-affected)
    run_buildkite_affected_groups
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac

echo
echo "Logs: ${LOG_DIR}"
echo "Summary: ${SUMMARY_FILE}"
column -t -s $'\t' "${SUMMARY_FILE}" || cat "${SUMMARY_FILE}"

if [[ "${FAILURES}" -ne 0 ]]; then
  exit 1
fi
