#!/bin/bash

# shellcheck disable=SC2329  # Every function in this file is only ever reached
# transitively through handle_amd_runner_exit, the EXIT trap handler - shellcheck's
# unused-function check doesn't do full reachability analysis from an indirectly
# invoked entry point, so it flags the whole diagnostic-collection call chain as
# dead code even though it runs live on every nonzero exit.

# This script runs ROCm tests either directly in a native CI pod or inside the
# corresponding Docker container. Multi-node tests continue to use Docker.
#
# Multi-node detection: Instead of matching on fragile group names, we detect
# multi-node jobs structurally by looking for the bracket command syntax
# "[node0_cmds] && [node1_cmds]" or via the NUM_NODES environment variable.
#
###############################################################################
# QUOTING / COMMAND PASSING
#
# Passing commands as positional arguments ($*) is fragile when the command
# string itself contains double quotes, e.g.:
#
#   bash run-amd-test.sh "export FLAGS="value" && pytest -m "not slow""
#
# The outer shell resolves the nested quotes *before* this script runs, so
# the script receives mangled input it cannot fully recover.
#
# Preferred: pass commands via the VLLM_TEST_COMMANDS environment variable:
#
#   export VLLM_TEST_COMMANDS='export FLAGS="value" && pytest -m "not slow"'
#   bash run-amd-test.sh
#
# Single-quoted assignment preserves all inner double quotes verbatim.
# The $* path is kept for backward compatibility but callers should migrate.
###############################################################################
set -o pipefail

: "${BUILDKIT_PROGRESS:=plain}"
: "${TERM:=xterm-256color}"
: "${FORCE_COLOR:=1}"
: "${CLICOLOR_FORCE:=1}"
: "${PY_COLORS:=1}"
: "${ROCM_DOCKER_TTY:=1}"
: "${PYTHONFAULTHANDLER:=1}"
: "${PYTEST_TIMEOUT:=2400}"

# Preserve runner-only values before clear_ci_orchestration_env removes them
# from the environment inherited by the test process.
amd_diagnostics_expected_gpu_count="${VLLM_CI_EXPECTED_GPU_COUNT:-1}"
amd_diagnostics_dir="${VLLM_CI_DIAGNOSTICS_DIR:-artifacts/amd-gpu-diagnostics}"
amd_diagnostics_checkout_root="${BUILDKITE_BUILD_CHECKOUT_PATH:-$(pwd -P)}"
amd_diagnostics_execution_mode="${VLLM_CI_EXECUTION_MODE:-single-node}"
amd_diagnostics_test_group="${VLLM_TEST_GROUP_NAME:-${BUILDKITE_STEP_KEY:-unknown}}"
amd_diagnostics_workspace="${VLLM_CI_WORKSPACE:-/vllm-workspace}"
amd_diagnostics_pod_name="${VLLM_CI_K8S_POD_NAME:-${POD_NAME:-${HOSTNAME:-}}}"
amd_diagnostics_k8s_namespace="${VLLM_CI_K8S_NAMESPACE:-${POD_NAMESPACE:-unknown}}"
amd_diagnostics_k8s_node_name="${VLLM_CI_K8S_NODE_NAME:-${NODE_NAME:-}}"
amd_diagnostics_collected=0
amd_diagnostics_memory_events_path=""
amd_diagnostics_probe_budget_seconds=25
amd_diagnostics_command_timeout_seconds=5
amd_diagnostics_upload_timeout_seconds=20
if [[ " ${PYTEST_ADDOPTS:-} " != *" --color"* ]]; then
  PYTEST_ADDOPTS="${PYTEST_ADDOPTS:+${PYTEST_ADDOPTS} }--color=yes"
fi
if [[ " ${PYTEST_ADDOPTS:-} " != *" --durations="* ]]; then
  PYTEST_ADDOPTS="${PYTEST_ADDOPTS:+${PYTEST_ADDOPTS} }--durations=25"
fi
if [[ " ${PYTEST_ADDOPTS:-} " != *" --durations-min="* ]]; then
  PYTEST_ADDOPTS="${PYTEST_ADDOPTS:+${PYTEST_ADDOPTS} }--durations-min=1.0"
fi
# Dump stacks after 25 minutes, then stop an individual test after 40 minutes.
if [[ " ${PYTEST_ADDOPTS:-} " != *" faulthandler_timeout="* ]]; then
  PYTEST_ADDOPTS="${PYTEST_ADDOPTS:+${PYTEST_ADDOPTS} }-o faulthandler_timeout=1500"
fi
if [[ " ${PYTEST_ADDOPTS:-} " != *" --timeout-method="* &&
  " ${PYTEST_ADDOPTS:-} " != *" --timeout-method "* ]]; then
  PYTEST_ADDOPTS="${PYTEST_ADDOPTS:+${PYTEST_ADDOPTS} }--timeout-method=thread"
fi
export BUILDKIT_PROGRESS TERM FORCE_COLOR CLICOLOR_FORCE PY_COLORS PYTEST_ADDOPTS PYTEST_TIMEOUT ROCM_DOCKER_TTY
export PYTHONFAULTHANDLER

# Export Python path for commands that run directly on the host. Containerized
# tests set this to /vllm-workspace below so spawned Python processes do not
# depend on their current working directory.
export PYTHONPATH="${PYTHONPATH:-..}"

ci_started_at=$SECONDS

###############################################################################
# Helper Functions
###############################################################################

report_docker_usage() {
  echo "--- Docker usage"
  docker system df || true
}

clear_ci_orchestration_env() {
  unset -v \
    VLLM_TEST_GROUP_NAME \
    VLLM_CI_REQUIRE_PERSISTENT_HF_CACHE \
    VLLM_CI_ARTIFACT_STEP \
    VLLM_TEST_CACHE \
    VLLM_CI_EXECUTION_MODE \
    VLLM_CI_WORKSPACE \
    VLLM_CI_REQUIRE_WORKSPACE_MOUNT \
    VLLM_TEST_COMMANDS \
    VLLM_CI_BRANCH \
    ROCM_BASE_DOCKERFILE \
    CI_BASE_DOCKERFILE \
    VLLM_CI_DOCKER_DISABLED \
    VLLM_CI_DIAGNOSTICS_DIR \
    VLLM_CI_EXPECTED_GPU_COUNT \
    VLLM_CI_K8S_POD_NAME \
    VLLM_CI_K8S_NAMESPACE \
    VLLM_CI_K8S_NODE_NAME \
    VLLM_CI_RESULTS_ROOT \
    VLLM_ALLOW_DEPRECATED_BEAM_SEARCH
}

amd_ci_teardown_log() {
  local event=$1
  shift

  printf '[amd-ci-teardown] event=%s' "${event}" >&2
  if (($#)); then
    printf ' %s' "$@" >&2
  fi
  printf '\n' >&2
}

run_docker_with_ci_timeout() {
  local raw_timeout=${CONTAINER_TIMEOUT_S:-0}
  local configured_timeout=0
  local elapsed=0
  local remaining_timeout=0
  local status=0

  if [[ ! "${raw_timeout}" =~ ^(0|[1-9][0-9]{0,5})$ ]]; then
    amd_ci_teardown_log configuration_error \
      "reason=invalid_timeout" "expected=integer_0_to_604800"
    return 2
  fi
  configured_timeout=$((10#${raw_timeout}))
  if ((configured_timeout > 604800)); then
    amd_ci_teardown_log configuration_error \
      "reason=invalid_timeout" "expected=integer_0_to_604800"
    return 2
  fi

  remaining_timeout=${configured_timeout}
  if ((configured_timeout > 0)); then
    if ! command -v timeout >/dev/null 2>&1; then
      amd_ci_teardown_log configuration_error \
        "reason=timeout_command_not_found"
      return 127
    fi
    elapsed=$((SECONDS - ci_started_at))
    if ((elapsed >= configured_timeout)); then
      amd_ci_teardown_log deadline_exhausted \
        "mode=docker" "configured_s=${configured_timeout}" "elapsed_s=${elapsed}"
      return 124
    fi
    remaining_timeout=$((configured_timeout - elapsed))
  fi

  amd_ci_teardown_log workload_started \
    "mode=docker" "timeout_s=${remaining_timeout}"
  if ((remaining_timeout == 0)); then
    "$@" || status=$?
  else
    # Docker runs interactively, so keep it in the foreground process group.
    # --verbose records both TERM and any KILL escalation caused by timeout.
    timeout --verbose --foreground --signal=TERM --kill-after=10s \
      "${remaining_timeout}s" "$@" || status=$?
  fi
  amd_ci_teardown_log workload_finished "mode=docker" "status=${status}"
  return "${status}"
}

is_native_runtime() {
  [[ "${AMD_CI_RUNTIME:-}" == "native" || "${NATIVE_CI:-}" == "true" ]]
}

validate_native_workspace() {
  local workspace_dir="${VLLM_CI_WORKSPACE:-/vllm-workspace}"
  local workspace_real=""
  local checkout_real=""
  local workspace_mount=""

  mkdir -p "${workspace_dir}" || return 1
  workspace_real=$(readlink -m "${workspace_dir}") || return 1
  if [[ -n "${BUILDKITE_BUILD_CHECKOUT_PATH:-}" ]]; then
    checkout_real=$(readlink -m "${BUILDKITE_BUILD_CHECKOUT_PATH}") || return 1
    if [[ "${checkout_real}" == "${workspace_real}" \
      || "${checkout_real}" == "${workspace_real}/"* \
      || "${workspace_real}" == "${checkout_real}/"* ]]; then
      echo "Refusing to replace ${workspace_real}; it overlaps the Buildkite checkout ${checkout_real}" >&2
      return 1
    fi
  fi
  if [[ "${VLLM_CI_REQUIRE_WORKSPACE_MOUNT:-1}" == "1" ]]; then
    if ! command -v findmnt >/dev/null 2>&1; then
      echo "findmnt is required to verify the native workspace mount" >&2
      return 1
    fi
    workspace_mount=$(findmnt -n -T "${workspace_real}" -o TARGET 2>/dev/null || true)
    if [[ "$(readlink -m "${workspace_mount:-/}")" != "${workspace_real}" ]]; then
      echo "Native CI requires a dedicated volume mounted at ${workspace_real}" >&2
      return 1
    fi
  fi
}

# Usage: overlay_python_only_source <commit> <wheel> <workspace_dir>
overlay_python_only_source() {
  local recorded_commit="$1"
  local wheel="$2"
  local workspace_dir="$3"
  local checkout="${BUILDKITE_BUILD_CHECKOUT_PATH:-}"
  local checkout_commit=""

  if [[ -z "${checkout}" || ! -d "${checkout}" ]]; then
    echo "Python-only native CI requires BUILDKITE_BUILD_CHECKOUT_PATH" >&2
    return 1
  fi
  if ! git -c "safe.directory=${checkout}" -C "${checkout}" \
    rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Buildkite checkout is not a Git worktree: ${checkout}" >&2
    return 1
  fi
  checkout_commit=$(
    git -c "safe.directory=${checkout}" -C "${checkout}" rev-parse HEAD
  ) || return 1
  if [[ "${checkout_commit}" != "${recorded_commit}" ]]; then
    echo "Buildkite checkout ${checkout_commit} does not match test image commit ${recorded_commit}" >&2
    return 1
  fi

  # setup.py normally derives this from .git via setuptools-scm. The native
  # source overlay deliberately excludes Git metadata, so preserve the exact
  # version from the already installed, artifact-matched wheel.
  VLLM_VERSION_OVERRIDE=$(
    python3 -c 'import importlib.metadata as m; print(m.version("vllm"))'
  ) || return 1
  export VLLM_VERSION_OVERRIDE
  VLLM_PRECOMPILED_WHEEL_LOCATION="${wheel}"
  export VLLM_PRECOMPILED_WHEEL_LOCATION
  echo "INFO: native Python-only wheel=${VLLM_PRECOMPILED_WHEEL_LOCATION}"

  echo "--- Overlaying full source checkout for Python-only compilation"
  # Archive the verified commit instead of copying the worktree so dirty or
  # untracked agent files cannot contaminate the artifact-matched workspace.
  git -c "safe.directory=${checkout}" -C "${checkout}" \
    archive --format=tar "${recorded_commit}" \
    | tar --no-same-owner -C "${workspace_dir}" -xf - || return 1
  for required_source in setup.py pyproject.toml vllm; do
    if [[ ! -e "${workspace_dir}/${required_source}" ]]; then
      echo "Full source checkout is missing ${required_source}" >&2
      return 1
    fi
  done
}

# Native pods run the per-build test image: the wheel is already installed and
# the test workspace is staged under /opt/vllm-ci/workspace.
prepare_native_workspace() {
  local test_commands="${1:-}"
  local workspace_dir="${VLLM_CI_WORKSPACE:-/vllm-workspace}"
  local staged_dir="/opt/vllm-ci/workspace"
  local image_commit=""
  local wheel_name=""

  validate_native_workspace || return 1
  image_commit=$(tr -d '\r\n' < /opt/vllm-ci/commit.txt 2>/dev/null || true)
  if [[ -z "${BUILDKITE_COMMIT:-}" || "${image_commit}" != "${BUILDKITE_COMMIT}" ]]; then
    echo "Image was built for ${image_commit:-<missing>}, not ${BUILDKITE_COMMIT:-unset}" >&2
    return 1
  fi

  echo "--- Preparing ${workspace_dir} from ${staged_dir}"
  find "${workspace_dir}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + || return 1
  cp -a "${staged_dir}/." "${workspace_dir}/" || return 1
  if [[ ! -d "${workspace_dir}/tests" ]]; then
    echo "Failed to stage the native test workspace" >&2
    return 1
  fi

  if [[ "${test_commands}" == *python_only_compile.sh* ]]; then
    # The image does not keep the wheel; download it from the image build.
    wheel_name=$(tr -d '\r\n' < /opt/vllm-ci/wheel-filename.txt) || return 1
    artifact_work_dir=$(mktemp -d -t vllm-rocm-wheel.XXXXXX) || return 1
    buildkite-agent artifact download "artifacts/vllm-rocm-wheel/${wheel_name}*" \
      "${artifact_work_dir}" --step "${VLLM_CI_ARTIFACT_STEP:-image-build-amd}" || return 1
    (cd "${artifact_work_dir}/artifacts/vllm-rocm-wheel" \
      && sha256sum -c "${wheel_name}.sha256") || return 1
    overlay_python_only_source "${image_commit}" \
      "${artifact_work_dir}/artifacts/vllm-rocm-wheel/${wheel_name}" \
      "${workspace_dir}" || return 1
  fi
}

initialize_native_environment() {
  local job_id="${BUILDKITE_JOB_ID:-${BUILDKITE_PARALLEL_JOB:-local}}"
  local job_id_suffix=""
  local native_root=""
  local hf_fstype=""
  local hf_mount=""

  if [[ "$(id -u)" -ne 0 ]]; then
    echo "Native ROCm CI currently requires the CI image to run as root" >&2
    return 1
  fi

  job_id="${job_id//[^A-Za-z0-9_.-]/_}"
  job_id_suffix="${job_id##*-}"
  job_id_suffix="${job_id_suffix:0:12}"
  native_root="/tmp/vllm-native-${job_id}"
  TMPDIR="/tmp/vllm-${job_id_suffix}/tmp"
  VLLM_RPC_BASE_PATH="/tmp"
  TORCHINDUCTOR_CACHE_DIR="${native_root}/cache/torchinductor"
  TRITON_CACHE_DIR="${native_root}/cache/triton"
  VLLM_CACHE_ROOT="${native_root}/cache/vllm"
  XDG_CACHE_HOME="${native_root}/cache/xdg"
  : "${HF_HOME:=/home/buildkite-agent/huggingface}"
  # datasets uses POSIX locks that are unsupported by the shared HF NFS cache.
  # Keep processed datasets job-local while retaining the persistent Hub cache.
  HF_DATASETS_CACHE="${native_root}/cache/huggingface/datasets"
  # openai-harmony downloads its tiktoken vocab on first use and caches it under
  # $TMPDIR by default, which is job-local; keep it with the persistent Hub cache.
  TIKTOKEN_RS_CACHE_DIR="${HF_HOME}/tiktoken-rs-cache"
  : "${HF_HUB_DOWNLOAD_TIMEOUT:=300}"
  : "${HF_HUB_ETAG_TIMEOUT:=60}"
  if [[ "${VLLM_CI_EXPECTED_GPU_COUNT:-1}" == "0" ]]; then
    # CPU-only native jobs intentionally reuse the ROCm wheel. Make that target
    # explicit so platform selection does not depend on wheel metadata.
    VLLM_TARGET_DEVICE=cpu
    export VLLM_TARGET_DEVICE
  fi
  export TMPDIR VLLM_RPC_BASE_PATH
  export TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR VLLM_CACHE_ROOT XDG_CACHE_HOME
  export HF_HOME HF_DATASETS_CACHE HF_HUB_DOWNLOAD_TIMEOUT HF_HUB_ETAG_TIMEOUT
  export TIKTOKEN_RS_CACHE_DIR
  export PYTORCH_ROCM_ARCH=""

  mkdir -p "${TMPDIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "${TRITON_CACHE_DIR}" \
    "${VLLM_CACHE_ROOT}" \
    "${XDG_CACHE_HOME}" \
    "${HF_HOME}" \
    "${TIKTOKEN_RS_CACHE_DIR}" \
    "${HF_DATASETS_CACHE}" || return 1

  echo "Native compile caches: VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT} TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"

  if [[ "${VLLM_CI_REQUIRE_PERSISTENT_HF_CACHE:-0}" == "1" ]]; then
    if ! command -v findmnt >/dev/null 2>&1; then
      echo "findmnt is required to verify the native Hugging Face cache mount" >&2
      return 1
    fi
    hf_mount=$(findmnt -n -T "${HF_HOME}" -o TARGET 2>/dev/null || true)
    if [[ -z "${hf_mount}" || "${hf_mount}" == "/" ]]; then
      echo "Native CI requires a persistent volume mounted at or above ${HF_HOME}" >&2
      return 1
    fi
  fi

  if command -v findmnt >/dev/null 2>&1; then
    hf_fstype=$(findmnt -n -T "${HF_HOME}" -o FSTYPE 2>/dev/null || true)
  fi
  if [[ "${hf_fstype}" == nfs || "${hf_fstype}" == nfs4 ]]; then
    # Keep hf-xet state local and avoid vectored writes on shared NFS.
    export HF_XET_CACHE="${native_root}/cache/hf-xet"
    export HF_XET_HIGH_PERFORMANCE=0
    export HF_XET_RECONSTRUCTION_USE_VECTORED_WRITE=0
    mkdir -p "${HF_XET_CACHE}" || return 1
    echo "Configured hf-xet for shared ${hf_fstype} cache at ${HF_HOME}"
  fi
}

check_dpx_gpu_exclusivity() {
  local devices=(/dev/dri/renderD*)
  local device_id lock_status

  if [[ "${#devices[@]}" -ne 1 || ! -c "${devices[0]}" ]]; then
    echo "DPX guard requires exactly one render device; refusing to start tests." >&2
    return 1
  fi
  # This queue mounts the same host-local HF cache into every pod on the node.
  if ! mountpoint -q "${HF_HOME}"; then
    echo "DPX guard requires the shared HF cache mount; refusing to start tests." >&2
    return 1
  fi
  device_id=$(stat -Lc '%t-%T' "${devices[0]}") || return 1
  mkdir -p "${HF_HOME}/.vllm-dpx-locks" || return 1
  # Hold the descriptor through workload and teardown; never delete the file.
  exec {dpx_gpu_lock_fd}>>"${HF_HOME}/.vllm-dpx-locks/${device_id}.lock" || return 1
  flock -n -E 75 "${dpx_gpu_lock_fd}" && return 0
  lock_status=$?
  if [[ "${lock_status}" -eq 75 ]]; then
    echo "DPX GPU collision: ${devices[0]} (${device_id}) is already locked by another CI job; refusing to start tests." >&2
  else
    echo "DPX GPU lock failed (status ${lock_status}); refusing to start tests." >&2
  fi
  return 1
}

run_native_preflight() {
  local expected_gpus="${VLLM_CI_EXPECTED_GPU_COUNT:-1}"

  if [[ ! "${expected_gpus}" =~ ^[0-9]+$ ]]; then
    echo "Invalid VLLM_CI_EXPECTED_GPU_COUNT=${expected_gpus}" >&2
    return 1
  fi

  python3 -c "import encodings, importlib.metadata as im, importlib.util as iu; [im.version(d) for d in ('transformers', 'torch', 'ray', 'sympy', 'markupsafe', 'vllm')]; missing=[m for m in ('torch.utils.model_zoo', 'transformers.models.nomic_bert', 'ray.dag', 'sympy.physics', 'markupsafe._speedups') if iu.find_spec(m) is None]; assert not missing, missing" || return 1

  if [[ "${expected_gpus}" == "0" ]]; then
    echo "Native CPU-only AMD job: skipping ROCm device validation"
    return 0
  fi

  echo "--- ROCm info"
  rocminfo || return 1
  VLLM_CI_EXPECTED_GPU_COUNT="${expected_gpus}" python3 - <<'PY'
import os

import torch

expected = int(os.environ["VLLM_CI_EXPECTED_GPU_COUNT"])
assert torch.version.hip, "PyTorch is not a ROCm build"
assert torch.cuda.is_available(), "ROCm GPU is not available to PyTorch"
actual = torch.cuda.device_count()
assert actual == expected, f"Expected {expected} ROCm GPU(s), found {actual}"
PY
}

is_multi_node() {
  local cmds="$1"
  # Primary signal: NUM_NODES environment variable set by the pipeline
  if [[ "${NUM_NODES:-1}" -gt 1 ]]; then
    return 0
  fi
  # Fallback: detect the bracket syntax structurally
  # Pattern: [...] && [...] (per-node command arrays)
  if [[ "$cmds" == *'] && ['* ]]; then
    return 0
  fi
  return 1
}

append_failure_diagnostic_section() {
  local log_file=$1
  local title=$2

  printf '\n===============================================================================\n' \
    >> "${log_file}"
  printf '%s\n' "${title}" >> "${log_file}"
  printf '===============================================================================\n' \
    >> "${log_file}"
}

append_failure_diagnostic_note() {
  local log_file=$1
  shift
  printf '%s\n' "$@" >> "${log_file}"
}

run_failure_diagnostic() {
  local log_file=$1
  local probe_deadline=$2
  shift 2
  local command_status=0
  local command_timeout_seconds=${amd_diagnostics_command_timeout_seconds}
  local remaining_seconds=$((probe_deadline - SECONDS))

  {
    printf '\n$'
    printf ' %q' "$@"
    printf '\n'
  } >> "${log_file}"

  if ! command -v timeout >/dev/null 2>&1; then
    echo "[timeout unavailable; bounded diagnostic command skipped]" \
      >> "${log_file}"
    return 0
  fi
  if [[ "${remaining_seconds}" -le 0 ]]; then
    echo "[diagnostic probe budget exhausted; command skipped]" \
      >> "${log_file}"
    return 0
  fi
  if [[ "${remaining_seconds}" -lt "${command_timeout_seconds}" ]]; then
    command_timeout_seconds=${remaining_seconds}
  fi

  timeout --kill-after=1s "${command_timeout_seconds}s" "$@" \
    >> "${log_file}" 2>&1
  command_status=$?
  if [[ "${command_status}" -ne 0 ]]; then
    echo "[diagnostic command exited with status ${command_status}]" \
      >> "${log_file}"
  fi
  return 0
}

upload_failure_diagnostics_artifact() {
  local upload_root=$1 upload_path=$2 agent_bin=""
  local agent_upload_help=""
  local -a upload_options=()

  if [[ -n "${BUILDKITE_BIN_PATH:-}" \
    && -x "${BUILDKITE_BIN_PATH}/buildkite-agent" ]]; then
    agent_bin="${BUILDKITE_BIN_PATH}/buildkite-agent"
  else
    agent_bin=$(command -v buildkite-agent 2>/dev/null || true)
  fi
  if [[ -z "${agent_bin}" && -x /workspace/buildkite-agent ]]; then
    agent_bin=/workspace/buildkite-agent
  fi
  [[ -n "${agent_bin}" && -n "${BUILDKITE_JOB_ID:-}" ]] || return 1
  command -v timeout >/dev/null 2>&1 || return 1

  # Older AMD runners predate the literal upload options. A restricted path
  # has no glob or delimiter characters, so it is safe with either agent.
  if [[ ! "${upload_path}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)*$ ]]; then
    agent_upload_help=$("${agent_bin}" artifact upload --help 2>&1 || true)
    if [[ "${agent_upload_help}" != *"--literal"* \
      || "${agent_upload_help}" != *"--delimiter"* ]]; then
      return 1
    fi
    upload_options=(--literal --delimiter "")
  fi

  (
    cd "${upload_root}" || exit 1
    BUILDKITE_AGENT_DEBUG=false \
      BUILDKITE_AGENT_DEBUG_HTTP=false \
      BUILDKITE_AGENT_TRACE_HTTP=false \
      BUILDKITE_AGENT_LOG_LEVEL=error \
      timeout --kill-after=2s "${amd_diagnostics_upload_timeout_seconds}s" \
      "${agent_bin}" artifact upload \
        "${upload_options[@]}" "./${upload_path}"
  ) >/dev/null 2>&1
}

append_failure_diagnostic_file() {
  local log_file=$1
  local label=$2
  local path=$3
  local command_status=0

  printf '\n%s:\n' "${label}" >> "${log_file}"
  printf '$ cat %q\n' "${path}" >> "${log_file}"
  if [[ ! -r "${path}" ]]; then
    echo "[file unavailable]" >> "${log_file}"
    return 0
  fi

  cat "${path}" >> "${log_file}" 2>&1
  command_status=$?
  if [[ "${command_status}" -ne 0 ]]; then
    echo "[diagnostic file read exited with status ${command_status}]" \
      >> "${log_file}"
  fi
  return 0
}

decode_mountinfo_path() {
  local value=$1

  value=${value//\\040/ }
  value=${value//\\011/$'\t'}
  value=${value//\\012/$'\n'}
  value=${value//\\134/\\}
  printf '%s\n' "${value}"
}

resolve_current_cgroup_v2_dir() {
  local cgroup_path=""
  local mount_root=""
  local mount_point=""
  local relative_path=""
  local candidate=""

  cgroup_path=$(awk -F: '$2 == "" { print $3; exit }' \
    /proc/self/cgroup 2>/dev/null)
  [[ -n "${cgroup_path}" ]] || return 1

  while IFS=$'\t' read -r mount_root mount_point; do
    mount_root=$(decode_mountinfo_path "${mount_root}")
    mount_point=$(decode_mountinfo_path "${mount_point}")
    if [[ "${cgroup_path}" == "/" ]]; then
      relative_path=""
    elif [[ "${mount_root}" == "/" ]]; then
      relative_path="${cgroup_path}"
    elif [[ "${cgroup_path}" == "${mount_root}" ]]; then
      relative_path=""
    elif [[ "${cgroup_path}" == "${mount_root}/"* ]]; then
      relative_path="/${cgroup_path#"${mount_root}/"}"
    else
      continue
    fi

    candidate="${mount_point%/}${relative_path}"
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done < <(
    awk '
      {
        for (i = 6; i <= NF; i++) {
          if ($i == "-" && $(i + 1) == "cgroup2") {
            print $4 "\t" $5
            break
          }
        }
      }
    ' /proc/self/mountinfo 2>/dev/null
  )
  return 1
}

collect_cgroup_files() {
  local log_file=$1
  local title=$2
  local cgroup_dir=$3
  shift 3
  local cgroup_file=""
  local files_collected=0

  append_failure_diagnostic_section "${log_file}" "${title}"
  if [[ -z "${cgroup_dir}" || ! -d "${cgroup_dir}" ]]; then
    append_failure_diagnostic_note "${log_file}" \
      "resolved_cgroup_path=unavailable"
    return 0
  fi

  append_failure_diagnostic_note "${log_file}" \
    "resolved_cgroup_path=${cgroup_dir}"
  for cgroup_file in "$@"; do
    if [[ -r "${cgroup_dir}/${cgroup_file}" ]]; then
      append_failure_diagnostic_file "${log_file}" "${cgroup_file}" \
        "${cgroup_dir}/${cgroup_file}"
      files_collected=$((files_collected + 1))
    fi
  done
  if [[ "${files_collected}" -eq 0 ]]; then
    append_failure_diagnostic_note "${log_file}" \
      "[no requested cgroup files were readable]"
  fi
}

collect_cgroup_diagnostics() {
  local log_file=$1
  local cgroup_dir=""

  append_failure_diagnostic_section "${log_file}" "Cgroup membership"
  append_failure_diagnostic_file "${log_file}" "current_process" \
    /proc/self/cgroup
  append_failure_diagnostic_file "${log_file}" "pid_namespace_init_process" \
    /proc/1/cgroup

  cgroup_dir=$(resolve_current_cgroup_v2_dir 2>/dev/null || true)
  collect_cgroup_files "${log_file}" \
    "Cgroup v2 current-process resource state" "${cgroup_dir}" \
    cgroup.type cgroup.events \
    memory.current memory.peak memory.max memory.high memory.low memory.min \
    memory.oom.group memory.swap.current memory.swap.max \
    memory.events memory.events.local memory.swap.events memory.stat \
    memory.pressure \
    cpu.max cpu.max.burst cpu.weight cpu.stat cpu.pressure \
    cpuset.cpus cpuset.cpus.effective cpuset.mems cpuset.mems.effective \
    pids.current pids.max pids.events pids.events.local \
    io.stat io.max io.weight io.pressure
  if [[ -n "${cgroup_dir}" && -r "${cgroup_dir}/memory.events" ]]; then
    amd_diagnostics_memory_events_path="${cgroup_dir}/memory.events"
  fi
}

collect_process_diagnostics() {
  local log_file=$1
  local probe_deadline=$2
  local shell_proc_dir="/proc/$$"

  append_failure_diagnostic_section "${log_file}" \
    "Runner process constraints and PID-namespace-visible processes"
  append_failure_diagnostic_file "${log_file}" "diagnostic_shell_limits" \
    "${shell_proc_dir}/limits"
  if command -v awk >/dev/null 2>&1; then
    # shellcheck disable=SC2016  # The expression is evaluated by awk.
    run_failure_diagnostic "${log_file}" "${probe_deadline}" awk '
      $1 ~ /^(Pid:|PPid:|Threads:|VmPeak:|VmSize:|VmHWM:|VmRSS:|RssAnon:|RssFile:|RssShmem:|Cpus_allowed_list:|Mems_allowed_list:|voluntary_ctxt_switches:|nonvoluntary_ctxt_switches:)$/ {
        print
      }
    ' "${shell_proc_dir}/status"
  fi
  if command -v nproc >/dev/null 2>&1; then
    run_failure_diagnostic "${log_file}" "${probe_deadline}" nproc
  fi
  run_failure_diagnostic "${log_file}" "${probe_deadline}" \
    /bin/bash -c 'ulimit -a'
  if command -v ps >/dev/null 2>&1 && command -v head >/dev/null 2>&1; then
    append_failure_diagnostic_note "${log_file}" \
      "process_snapshot=top_100_by_rss"
    run_failure_diagnostic "${log_file}" "${probe_deadline}" \
      /bin/bash -c \
      'ps -eo pid,ppid,stat,etimes,nlwp,pcpu,rss,vsz,comm --sort=-rss | head -n 101'
  fi
  return 0
}

collect_mount_diagnostic() {
  local log_file=$1
  local probe_deadline=$2
  local label=$3
  local path=$4

  printf '\n%s (%s):\n' "${label}" "${path}" >> "${log_file}"
  if [[ ! -e "${path}" ]]; then
    echo "[path unavailable]" >> "${log_file}"
    return 0
  fi
  if command -v findmnt >/dev/null 2>&1; then
    run_failure_diagnostic "${log_file}" "${probe_deadline}" findmnt \
      -n -T "${path}" -o TARGET,FSTYPE,VFS-OPTIONS
  fi
  if command -v df >/dev/null 2>&1; then
    run_failure_diagnostic "${log_file}" "${probe_deadline}" df -h \
      --output=fstype,size,used,avail,pcent,itotal,iused,iavail,ipcent,target \
      -- "${path}"
  fi
  return 0
}

collect_execution_resource_diagnostics() {
  local log_file=$1
  local probe_deadline=$2

  # free and /proc/meminfo generally describe the node from a pod. Cgroups are
  # the live source for this container's resource limits, usage, and events.
  collect_cgroup_diagnostics "${log_file}"
  collect_process_diagnostics "${log_file}" "${probe_deadline}"

  append_failure_diagnostic_section "${log_file}" \
    "Mount-visible filesystem capacity"
  append_failure_diagnostic_note "${log_file}" \
    "These values are filesystem views, not Kubernetes ephemeral-storage quotas."
  collect_mount_diagnostic "${log_file}" "${probe_deadline}" \
    "checkout" "${amd_diagnostics_checkout_root}"
  if is_native_runtime; then
    collect_mount_diagnostic "${log_file}" "${probe_deadline}" \
      "native workspace emptyDir" "${amd_diagnostics_workspace}"
  else
    collect_mount_diagnostic "${log_file}" "${probe_deadline}" \
      "test workspace (if mounted)" "${amd_diagnostics_workspace}"
  fi
  collect_mount_diagnostic "${log_file}" "${probe_deadline}" \
    "temporary directory" "${TMPDIR:-/tmp}"
  if [[ "${TMPDIR:-/tmp}" != "/tmp" ]]; then
    collect_mount_diagnostic "${log_file}" "${probe_deadline}" \
      "system temporary directory" /tmp
  fi
  if is_native_runtime; then
    collect_mount_diagnostic "${log_file}" "${probe_deadline}" \
      "shared-memory emptyDir" /dev/shm
  else
    collect_mount_diagnostic "${log_file}" "${probe_deadline}" \
      "shared-memory mount" /dev/shm
  fi
  if [[ -n "${HF_HOME:-}" ]]; then
    collect_mount_diagnostic "${log_file}" "${probe_deadline}" \
      "Hugging Face cache" "${HF_HOME}"
  fi
  return 0
}

collect_amd_device_nodes() {
  local log_file=$1
  local probe_deadline=$2
  local device=""
  local device_count=0

  for device in /dev/kfd /dev/dri/renderD*; do
    if [[ ! -e "${device}" ]]; then
      continue
    fi
    device_count=$((device_count + 1))
    if command -v stat >/dev/null 2>&1; then
      run_failure_diagnostic "${log_file}" "${probe_deadline}" stat -Lc \
        '%n type=%F mode=%a owner=%u:%g device=%t:%T' -- "${device}"
    else
      append_failure_diagnostic_note "${log_file}" "device=${device}"
    fi
  done
  if [[ "${device_count}" -eq 0 ]]; then
    append_failure_diagnostic_note "${log_file}" \
      "No /dev/kfd or /dev/dri/renderD* device nodes are visible."
  fi
  return 0
}

collect_rocm_failure_diagnostics() {
  local exit_code=$1
  local job_id="${BUILDKITE_JOB_ID:-local}"
  local retry_count="${BUILDKITE_RETRY_COUNT:-0}"
  local parallel_job="${BUILDKITE_PARALLEL_JOB:-0}"
  local diagnostics_relative_path=""
  local diagnostics_path=""
  local diagnostics_parent=""
  local diagnostics_fallback_root=""
  local diagnostics_storage="checkout"
  local diagnostics_artifact_summary=""
  local checkout_real=""
  local diagnostics_parent_real=""
  local exit_signal=""
  local probe_deadline=0
  local runtime="single-node-docker"
  local diagnostics_scope="outer-runner-after-test-container-exit"
  local identity_label="runner"
  local identity_value="${HOSTNAME:-unknown}"
  local k8s_pod="unknown"
  local k8s_namespace="${amd_diagnostics_k8s_namespace}"
  local oom_kill_count=""
  local summary_identity_label="Runner"
  local -a summary_rows=()

  if [[ "${amd_diagnostics_collected}" == "1" ]]; then
    return 0
  fi
  amd_diagnostics_collected=1

  if is_native_runtime; then
    runtime="native-kubernetes"
    diagnostics_scope="current-container-cgroup-and-namespaces"
    if [[ -n "${amd_diagnostics_pod_name}" ]]; then
      identity_label="pod"
      identity_value="${amd_diagnostics_pod_name}"
      k8s_pod="${amd_diagnostics_pod_name}"
    else
      identity_label="container_hostname"
    fi
  elif [[ "${amd_diagnostics_execution_mode}" == "multi-node" ]]; then
    runtime="multi-node-docker"
  fi
  if [[ "${k8s_namespace}" == "unknown" \
    && -r /var/run/secrets/kubernetes.io/serviceaccount/namespace ]]; then
    k8s_namespace=$(
      tr -d '\r\n' \
        < /var/run/secrets/kubernetes.io/serviceaccount/namespace 2>/dev/null
    )
  fi

  job_id="${job_id//[^A-Za-z0-9_.-]/_}"
  retry_count="${retry_count//[^A-Za-z0-9_.-]/_}"
  parallel_job="${parallel_job//[^A-Za-z0-9_.-]/_}"

  if [[ "${exit_code}" -gt 128 && "${exit_code}" -le 192 ]]; then
    exit_signal=$(kill -l "$((exit_code - 128))" 2>/dev/null || true)
  fi

  # Buildkite keeps the supplied artifact path. Restrict the configurable
  # directory to a checkout-relative path so the UI shows a clean artifact key.
  if [[ -z "${amd_diagnostics_dir}" \
    || "${amd_diagnostics_dir}" == /* \
    || "${amd_diagnostics_dir}" == *".."* ]]; then
    echo "WARNING: ignoring unsafe VLLM_CI_DIAGNOSTICS_DIR"
    amd_diagnostics_dir="artifacts/amd-gpu-diagnostics"
  fi
  diagnostics_relative_path="${amd_diagnostics_dir}/${job_id}/diagnostics.log"
  diagnostics_path="${amd_diagnostics_checkout_root}/${diagnostics_relative_path}"
  diagnostics_parent=$(dirname "${diagnostics_path}")
  checkout_real=$(readlink -m "${amd_diagnostics_checkout_root}" 2>/dev/null || true)
  diagnostics_parent_real=$(readlink -m "${diagnostics_parent}" 2>/dev/null || true)
  if [[ -z "${checkout_real}" || -z "${diagnostics_parent_real}" \
    || ("${diagnostics_parent_real}" != "${checkout_real}" \
      && "${diagnostics_parent_real}" != "${checkout_real}/"*) ]] \
    || ! mkdir -p "${diagnostics_parent}" \
    || [[ -L "${diagnostics_path}" ]] \
    || ! (set -o noclobber; : > "${diagnostics_path}") 2>/dev/null; then
    echo "WARNING: unable to create AMD CI diagnostics in the checkout; using temporary storage."
    diagnostics_path=""
    if diagnostics_fallback_root=$(mktemp -d -t \
      vllm-amd-diagnostics.XXXXXX 2>/dev/null); then
      diagnostics_path="${diagnostics_fallback_root}/${diagnostics_relative_path}"
      diagnostics_parent=$(dirname "${diagnostics_path}")
      if (umask 077; mkdir -p "${diagnostics_parent}" \
        && (set -o noclobber; : > "${diagnostics_path}")) 2>/dev/null; then
        diagnostics_storage="temporary-fallback"
      else
        rm -rf -- "${diagnostics_fallback_root}" || true
        diagnostics_fallback_root=""
        diagnostics_path=""
      fi
    fi
  fi
  if [[ -z "${diagnostics_path}" ]]; then
    echo "WARNING: unable to create AMD CI diagnostics at ${diagnostics_relative_path}."
    printf '\nAMD CI failure summary\n'
    printf '%-22s | %s\n' \
      "Field" "Value" \
      "----------------------" "-----" \
      "Exit code" "${exit_code}" \
      "Signal" "${exit_signal:-none}" \
      "Runtime" "${runtime}" \
      "Pod/container" "${identity_value}" \
      "Diagnostics artifact" "unavailable: ${diagnostics_relative_path}"
    return 0
  fi

  {
    echo "AMD CI failure diagnostics"
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_code=${exit_code}"
    echo "exit_signal=${exit_signal:-none}"
    echo "build_url=${BUILDKITE_BUILD_URL:-unknown}"
    echo "job_id=${BUILDKITE_JOB_ID:-unknown}"
    echo "step_key=${BUILDKITE_STEP_KEY:-unknown}"
    echo "step_label=${BUILDKITE_LABEL:-unknown}"
    echo "test_group=${amd_diagnostics_test_group}"
    echo "retry_count=${retry_count}"
    echo "parallel_job=${parallel_job}"
    echo "execution_mode=${amd_diagnostics_execution_mode}"
    echo "runtime=${runtime}"
    echo "diagnostics_scope=${diagnostics_scope}"
    echo "expected_gpu_count=${amd_diagnostics_expected_gpu_count}"
    echo "probe_budget_seconds=${amd_diagnostics_probe_budget_seconds}"
    echo "command_timeout_seconds=${amd_diagnostics_command_timeout_seconds}"
    echo "diagnostics_storage=${diagnostics_storage}"
    echo "agent_name=${BUILDKITE_AGENT_NAME:-unknown}"
    echo "container_hostname=${HOSTNAME:-unknown}"
    echo "k8s_pod=${k8s_pod}"
    echo "k8s_namespace=${k8s_namespace:-unknown}"
    echo "k8s_node=${amd_diagnostics_k8s_node_name:-unknown}"
    echo "kernel=$(uname -srmo 2>/dev/null || echo unknown)"
  } > "${diagnostics_path}"

  probe_deadline=$((SECONDS + amd_diagnostics_probe_budget_seconds))

  # Cgroup, process, and mount state is fast and most useful for explaining OOM,
  # throttling, PID exhaustion, and emptyDir pressure. Capture it before external
  # probes can consume the shared deadline.
  collect_execution_resource_diagnostics \
    "${diagnostics_path}" "${probe_deadline}"

  append_failure_diagnostic_section "${diagnostics_path}" \
    "AMD GPU diagnostics (tool-visible devices)"
  if [[ "${amd_diagnostics_expected_gpu_count}" == "0" ]]; then
    append_failure_diagnostic_note "${diagnostics_path}" \
      "CPU-only job; AMD GPU probes skipped."
  else
    if [[ "${runtime}" == "native-kubernetes" ]]; then
      append_failure_diagnostic_note "${diagnostics_path}" \
        "SMI visibility may be broader than the Kubernetes device allocation."
    else
      append_failure_diagnostic_note "${diagnostics_path}" \
        "Outer-runner SMI visibility may be broader than the test container."
    fi
    collect_amd_device_nodes "${diagnostics_path}" "${probe_deadline}"
  fi
  if [[ "${amd_diagnostics_expected_gpu_count}" != "0" ]] \
    && command -v amd-smi >/dev/null 2>&1; then
    run_failure_diagnostic "${diagnostics_path}" "${probe_deadline}" \
      amd-smi version
    # Bus data identifies a card within the public node without publishing its
    # persistent UUID, serial number, or process list.
    run_failure_diagnostic "${diagnostics_path}" "${probe_deadline}" \
      amd-smi static --bus --gpu all
    run_failure_diagnostic "${diagnostics_path}" "${probe_deadline}" \
      amd-smi metric --ecc --ecc-blocks --pcie --gpu all
    run_failure_diagnostic "${diagnostics_path}" "${probe_deadline}" \
      amd-smi static --ras --gpu all
    run_failure_diagnostic "${diagnostics_path}" "${probe_deadline}" \
      amd-smi metric --power --temperature --usage --mem-usage --gpu all
    run_failure_diagnostic "${diagnostics_path}" "${probe_deadline}" \
      amd-smi xgmi --link-status --gpu all
  elif [[ "${amd_diagnostics_expected_gpu_count}" != "0" ]] \
    && command -v rocm-smi >/dev/null 2>&1; then
    run_failure_diagnostic "${diagnostics_path}" "${probe_deadline}" rocm-smi \
      --showbus --showreplaycount --showrasinfo --showpagesinfo
  elif [[ "${amd_diagnostics_expected_gpu_count}" != "0" ]]; then
    append_failure_diagnostic_note "${diagnostics_path}" \
      "Neither amd-smi nor rocm-smi is available."
  fi

  if [[ "${runtime}" == "native-kubernetes" ]]; then
    append_failure_diagnostic_section "${diagnostics_path}" \
      "Unauthenticated pod network reachability"
  else
    append_failure_diagnostic_section "${diagnostics_path}" \
      "Unauthenticated outer-runner network reachability"
  fi
  if command -v curl >/dev/null 2>&1; then
    run_failure_diagnostic "${diagnostics_path}" "${probe_deadline}" curl -q \
      --silent --location --proto '=https' --proto-redir '=https' \
      --max-redirs 3 --output /dev/null --connect-timeout 2 --max-time 4 \
      --write-out 'target=huggingface http_code=%{http_code} dns_done_s=%{time_namelookup} connect_done_s=%{time_connect} tls_done_s=%{time_appconnect} first_byte_s=%{time_starttransfer} total_s=%{time_total}\n' \
      https://huggingface.co/api/models/gpt2
    run_failure_diagnostic "${diagnostics_path}" "${probe_deadline}" curl -q \
      --silent --location --proto '=https' --proto-redir '=https' \
      --max-redirs 3 --output /dev/null --connect-timeout 2 --max-time 4 \
      --write-out 'target=github_git_smart_http http_code=%{http_code} dns_done_s=%{time_namelookup} connect_done_s=%{time_connect} tls_done_s=%{time_appconnect} first_byte_s=%{time_starttransfer} total_s=%{time_total}\n' \
      'https://github.com/vllm-project/ci-infra.git/info/refs?service=git-upload-pack'
  else
    append_failure_diagnostic_note "${diagnostics_path}" \
      "curl unavailable; reachability probes skipped."
  fi

  if [[ "${runtime}" == "native-kubernetes" \
    && -n "${amd_diagnostics_memory_events_path}" \
    && -r "${amd_diagnostics_memory_events_path}" ]]; then
    oom_kill_count=$(
      awk '$1 == "oom_kill" { print $2; exit }' \
        "${amd_diagnostics_memory_events_path}" 2>/dev/null
    )
    if [[ ! "${oom_kill_count}" =~ ^[0-9]+$ ]]; then
      oom_kill_count=""
    fi
  fi

  case "${identity_label}" in
    pod)
      summary_identity_label="Kubernetes pod"
      ;;
    container_hostname)
      summary_identity_label="Container hostname"
      ;;
  esac

  summary_rows=(
    "Field" "Value"
    "----------------------" "-----"
    "Exit code" "${exit_code}"
    "Signal" "${exit_signal:-none}"
    "Runtime" "${runtime}"
    "${summary_identity_label}" "${identity_value}"
  )
  if [[ -n "${amd_diagnostics_k8s_node_name}" ]]; then
    summary_rows+=("Kubernetes node" "${amd_diagnostics_k8s_node_name}")
  fi
  if [[ -n "${oom_kill_count}" ]]; then
    summary_rows+=("Cgroup OOM kills" "${oom_kill_count}")
  fi
  diagnostics_artifact_summary="${diagnostics_relative_path}"
  if [[ -n "${diagnostics_fallback_root}" ]]; then
    if upload_failure_diagnostics_artifact \
      "${diagnostics_fallback_root}" "${diagnostics_relative_path}"; then
      echo "Uploaded AMD CI diagnostics artifact from temporary storage: ${diagnostics_relative_path}"
      diagnostics_artifact_summary="${diagnostics_relative_path} (direct upload)"
    else
      echo "WARNING: failed to upload temporary AMD CI diagnostics artifact: ${diagnostics_relative_path}"
      echo "--- AMD CI diagnostics (artifact upload failed)"
      cat "${diagnostics_path}" || true
      diagnostics_artifact_summary="job log only: direct upload failed"
    fi
    rm -rf -- "${diagnostics_fallback_root}" || \
      echo "WARNING: unable to remove temporary AMD CI diagnostics storage."
  fi
  summary_rows+=("Diagnostics artifact" "${diagnostics_artifact_summary}")

  printf '\nAMD CI failure summary\n'
  printf '%-22s | %s\n' "${summary_rows[@]}"

  return 0
}

handle_pytest_exit() {
  local exit_code=$1
  if [ "$exit_code" -eq 5 ]; then
    echo "Pytest exit code 5 (no tests collected) - treating as success."
    exit 0
  fi
  exit "$exit_code"
}

###############################################################################
# Pytest marker/keyword re-quoting
#
# When commands are passed through Buildkite -> shell -> $* -> bash -c,
# quotes around multi-word pytest -m/-k expressions get stripped:
#   pytest -v -s -m 'not cpu_test' v1/core
# becomes:
#   pytest -v -s -m not cpu_test v1/core
#
# pytest then interprets "cpu_test" as a file path, not part of the marker.
#
# This function detects unquoted expressions after -m/-k and re-quotes them
# by collecting tokens until a recognizable boundary is reached:
#   - test path (contains '/')
#   - test file (ends with '.py')
#   - another pytest flag (--xxx or -x single-char flags)
#   - command separator (&& || ; |)
#   - environment variable assignment (FOO=bar)
#
# Single-word markers (e.g. -m cpu_test, -m hybrid_model) pass through
# unquoted since they have no spaces and work fine.
#
# Already-quoted expressions (containing literal single quotes) are passed
# through untouched to avoid double-quoting well-formed shell fragments.
#
# NOTE: This ONLY fixes -m/-k flags. It cannot recover arbitrary inner
# double-quotes stripped by the calling shell (see header comment).
# Use VLLM_TEST_COMMANDS to avoid the problem entirely.
###############################################################################
re_quote_pytest_markers() {
  local input="$1"
  local output=""
  local collecting=false
  local marker_buf=""

  # Strip backslash-newline continuations, then flatten remaining newlines
  local flat="${input//$'\\\n'/ }"
  flat="${flat//$'\n'/ }"

  # Disable globbing to prevent *.py etc. from expanding during read -ra
  local restore_glob
  restore_glob="$(shopt -p -o noglob 2>/dev/null || true)"
  set -o noglob
  local -a words
  read -ra words <<< "$flat"
  eval "$restore_glob"

  for word in "${words[@]}"; do
    if $collecting; then
      # If the token we're about to collect already contains a literal
      # single quote, the expression was already quoted upstream.
      # Flush and stop collecting.
      if [[ "$word" == *"'"* ]]; then
        if [[ -n "$marker_buf" ]]; then
          # Should not normally happen (partial buf + quote), flush raw
          output+="${marker_buf} "
          marker_buf=""
        fi
        output+="${word} "
        collecting=false
        continue
      fi

      local is_boundary=false
      case "$word" in
        # Line-continuation artifact
        "\\")
          is_boundary=true ;;
        # Command separators
        "&&"|"||"|";"|"|")
          is_boundary=true ;;
        # Long flags (--ignore, --shard-id, etc.)
        --*)
          is_boundary=true ;;
        # Short flags (-v, -s, -x, etc.) but NOT negative marker tokens
        # like "not" which don't start with "-". Also skip -k/-m which
        # would start a new marker (handled below).
        -[a-zA-Z])
          is_boundary=true ;;
        # Test path (contains /)
        */*)
          is_boundary=true ;;
        # Test file (ends with .py, possibly with ::method)
        *.py|*.py::*)
          is_boundary=true ;;
        # Environment variable assignment preceding a command (FOO=bar)
        *=*)
          # Only treat as boundary if it looks like VAR=value, not
          # pytest filter expressions like num_gpus=2 inside markers
          if [[ "$word" =~ ^[A-Z_][A-Z0-9_]*= ]]; then
            is_boundary=true
          fi
          ;;
      esac

      if $is_boundary; then
        # Strip surrounding double quotes if present (from upstream
        # single-to-double conversion); without this, wrapping below
        # would produce '"expr"' with literal double-quote characters.
        if [[ "$marker_buf" == '"'*'"' ]]; then
          marker_buf="${marker_buf#\"}"
          marker_buf="${marker_buf%\"}"
        fi
        # Flush the collected marker expression
        if [[ "$marker_buf" == *" "* || "$marker_buf" == *"("* ]]; then
          output+="'${marker_buf}' "
        else
          output+="${marker_buf} "
        fi
        collecting=false
        marker_buf=""
        # Check if this boundary word itself starts a new -m/-k
        if [[ "$word" == "-m" || "$word" == "-k" ]]; then
          output+="${word} "
          collecting=true
        # Drop stray backslash tokens silently
        elif [[ "$word" == "\\" ]]; then
          :
        else
          output+="${word} "
        fi
      else
        # Accumulate into marker buffer
        if [[ -n "$marker_buf" ]]; then
          marker_buf+=" ${word}"
        else
          marker_buf="${word}"
        fi
      fi
    elif [[ "$word" == "-m" || "$word" == "-k" ]]; then
      output+="${word} "
      collecting=true
      marker_buf=""
    else
      output+="${word} "
    fi
  done

  # Flush any trailing marker expression (marker at end of command)
  if $collecting && [[ -n "$marker_buf" ]]; then
    # Strip surrounding double quotes (see mid-stream flush comment)
    if [[ "$marker_buf" == '"'*'"' ]]; then
      marker_buf="${marker_buf#\"}"
      marker_buf="${marker_buf%\"}"
    fi
    if [[ "$marker_buf" == *" "* || "$marker_buf" == *"("* ]]; then
      output+="'${marker_buf}'"
    else
      output+="${marker_buf}"
    fi
  fi

  echo "${output% }"
}

# shellcheck disable=SC2317  # Called indirectly by the EXIT trap.
handle_amd_runner_exit() {
  local exit_code=${1:-$?}
  if [[ "${exit_code}" -ne 0 ]]; then
    collect_rocm_failure_diagnostics "${exit_code}"
  fi
  return 0
}

# Catch both test failures and wrapper/setup failures. Runtime-specific cleanup
# traps below replace this trap and call the same handler after cleanup.
trap handle_amd_runner_exit EXIT

###############################################################################
# Main
###############################################################################

if is_native_runtime; then
  echo "--- Native in-pod ROCm CI (AMD_CI_RUNTIME=${AMD_CI_RUNTIME:-unset}, NATIVE_CI=${NATIVE_CI:-unset})"
  artifact_work_dir=""

  # shellcheck disable=SC2317  # Called indirectly by the EXIT trap.
  cleanup_native_workspace() {
    local exit_code=$?
    if [[ -n "${artifact_work_dir}" ]]; then
      rm -rf "${artifact_work_dir}"
    fi
    handle_amd_runner_exit "${exit_code}"
  }
  trap cleanup_native_workspace EXIT

  if [[ -n "${VLLM_TEST_COMMANDS:-}" ]]; then
    commands="${VLLM_TEST_COMMANDS}"
    commands_source="env"
  else
    commands="$*"
    commands_source="argv"
    if [[ -z "$commands" ]]; then
      echo "Error: No test commands provided for native CI." >&2
      exit 1
    fi
  fi

  if [[ "$commands_source" == "argv" ]]; then
    commands=$(re_quote_pytest_markers "$commands")
  fi

  if is_multi_node "$commands"; then
    echo "Native CI does not support multi-node jobs yet."
    exit 1
  fi

  if ! initialize_native_environment; then
    echo "Failed to initialize the native test environment"
    exit 1
  fi
  if [[ "${BUILDKITE_AGENT_META_DATA_QUEUE:-}" == *dpx* \
    && "${VLLM_CI_EXPECTED_GPU_COUNT:-1}" != "0" ]]; then
    check_dpx_gpu_exclusivity || exit 1
  fi
  if [[ "${commands}" == *python_only_compile.sh* ]]; then
    # This no-GPU job validates the ROCm precompiled/editable install path,
    # rather than CPU runtime platform selection.
    VLLM_TARGET_DEVICE=rocm
    export VLLM_TARGET_DEVICE
  fi
  if ! prepare_native_workspace "${commands}"; then
    echo "Failed to prepare native test workspace"
    exit 1
  fi

  export PYTHONPATH="${VLLM_CI_WORKSPACE:-/vllm-workspace}"

  echo "Native test commands: $commands"
  run_native_preflight || exit 1
  echo "--- Test log"
  # Keep AMD CI orchestration variables out of vLLM's runtime environment.
  clear_ci_orchestration_env
  /bin/bash -o pipefail -c "${commands}"
  handle_pytest_exit "$?"
fi

# --- GPU initialization for Docker execution ---
echo "--- ROCm info"
rocminfo

# --- Docker status ---
report_docker_usage

# The per-build test image; agent hooks also read DOCKER_IMAGE_NAME.
image_name="${DOCKER_IMAGE_NAME:-rocm/vllm-ci:build-${BUILDKITE_BUILD_ID:-local}}"
container_name="rocm_${BUILDKITE_COMMIT}_$(tr -dc A-Za-z0-9 < /dev/urandom | head -c 10; echo)"
# The image stages the test workspace under /opt/vllm-ci/workspace.
stage_workspace="mkdir -p /vllm-workspace && cp -a /opt/vllm-ci/workspace/. /vllm-workspace/"

# shellcheck disable=SC2317  # Called indirectly by the EXIT trap.
remove_docker_container() {
  local exit_code=$?
  if docker container inspect "${container_name}" >/dev/null 2>&1; then
    docker rm -f "${container_name}" || true
  fi
  if [[ "${VLLM_CI_REMOVE_TEST_IMAGE:-0}" == "1" ]]; then
    docker image rm -f "${image_name}" || true
  else
    # Keep images by default so later jobs on the same AMD node reuse the
    # shared runtime-ci layers and only pull the per-build vLLM layer.
    echo "Keeping ROCm test image locally: ${image_name}"
  fi
  handle_amd_runner_exit "${exit_code}"
}
trap remove_docker_container EXIT

HF_CACHE="$(realpath ~)/huggingface"
mkdir -p "${HF_CACHE}"
HF_MOUNT="/root/.cache/huggingface"

# Hugging Face Hub defaults to 10s request/download timeouts, while the ROCm
# CI image currently raises downloads to 60s. AMD model-test jobs routinely
# start from a cold or partially-populated shared cache, and the 60s read cap
# has still timed out before pytest reached the vLLM behavior under test.
# Keep the CI default explicit and overridable from the Buildkite environment.
: "${HF_HUB_DOWNLOAD_TIMEOUT:=300}"
: "${HF_HUB_ETAG_TIMEOUT:=60}"

# ---- Command source selection ----
# Prefer VLLM_TEST_COMMANDS (preserves all inner quoting intact).
# Fall back to $* for backward compatibility, but warn that inner
# double-quotes will have been stripped by the calling shell.
if [[ -n "${VLLM_TEST_COMMANDS:-}" ]]; then
  commands="${VLLM_TEST_COMMANDS}"
  commands_source="env"
  echo "Commands sourced from VLLM_TEST_COMMANDS (quoting preserved)"
else
  commands="$*"
  commands_source="argv"
  if [[ -z "$commands" ]]; then
    echo "Error: No test commands provided." >&2
    echo "Usage:" >&2
    echo "  Preferred:  VLLM_TEST_COMMANDS='...' bash $0" >&2
    echo "  Legacy:     bash $0 \"commands here\"" >&2
    exit 1
  fi
  echo "Commands sourced from positional args (legacy mode)"
  echo "WARNING: Inner double-quotes in the command string may have been"
  echo "  stripped by the calling shell. If you see syntax errors, switch to:"
  echo "  export VLLM_TEST_COMMANDS='your commands here'"
  echo "  bash $0"
fi

echo "Raw commands: $commands"

# Only try to repair stripped pytest -m/-k quoting in legacy argv mode.
# VLLM_TEST_COMMANDS preserves inner quoting already, and re-quoting that path
# can corrupt embedded echo strings or otherwise well-formed shell fragments.
if [[ "$commands_source" == "argv" ]]; then
  commands=$(re_quote_pytest_markers "$commands")
  echo "After re-quoting: $commands"
else
  echo "Skipping re-quoting for VLLM_TEST_COMMANDS input"
fi

echo "Final commands: $commands"

if [[ "$commands" == *python_only_compile.sh* ]]; then
  # It needs the same-build wheel and a source checkout; see prepare_native_workspace.
  echo "Error: python_only_compile.sh requires native execution (dind: false)." >&2
  exit 1
fi

echo "--- Pulling container"
docker pull "${image_name}" || exit 1

# Match native CPU jobs even when the container can see AMD devices.
cpu_platform_env=()
if [[ "${VLLM_CI_EXPECTED_GPU_COUNT:-1}" == "0" ]]; then
  cpu_platform_env=(-e "VLLM_TARGET_DEVICE=cpu")
fi

MYPYTHONPATH="/vllm-workspace"

container_job_id="${BUILDKITE_JOB_ID:-${BUILDKITE_PARALLEL_JOB:-0}}"
container_job_id="${container_job_id//[^A-Za-z0-9_.-]/_}"
container_job_id_short="${container_job_id:0:8}"
CONTAINER_TMPDIR="/tmp/vllm-${container_job_id_short}"
CONTAINER_CACHE_ROOT="/tmp/vllm-buildkite-${container_job_id}/cache"
CONTAINER_PREFLIGHT="${stage_workspace} && mkdir -p \"\$TMPDIR\" \"\$TIKTOKEN_RS_CACHE_DIR\" \"\$TORCHINDUCTOR_CACHE_DIR\" \"\$TRITON_CACHE_DIR\" \"\$VLLM_CACHE_ROOT\" \"\$XDG_CACHE_HOME\" && python -c \"import encodings, importlib.metadata as im, importlib.util as iu; [im.version(d) for d in ('transformers', 'torch', 'ray', 'sympy', 'markupsafe', 'vllm')]; missing=[m for m in ('torch.utils.model_zoo', 'transformers.models.nomic_bert', 'ray.dag', 'sympy.physics', 'markupsafe._speedups') if iu.find_spec(m) is None]; assert not missing, missing\""

# Verify GPU access
render_gid=$(getent group render | cut -d: -f3)
if [[ -z "$render_gid" ]]; then
  echo "Error: 'render' group not found. This is required for GPU access." >&2
  exit 1
fi

# --- RDMA device passthrough (conditional) ---
# If the host has RDMA devices, pass them through so tests like
# test_moriio_connector can access ibverbs. On hosts without RDMA
# hardware the tests will gracefully skip via _rdma_available().
RDMA_FLAGS=""
if [ -d /dev/infiniband ]; then
  echo "RDMA devices detected on host, enabling passthrough"
  RDMA_FLAGS="--device /dev/infiniband --cap-add=IPC_LOCK"
else
  echo "No RDMA devices found on host, RDMA tests will be skipped"
fi

# --- Route: multi-node vs single-node ---
clear_ci_orchestration_env
if is_multi_node "$commands"; then
  echo "--- Multi-node job detected"
  DCKR_VER=$(docker --version | sed 's/Docker version \(.*\), build .*/\1/')
  export DCKR_VER

  # Parse the bracket syntax:  prefix ; [node0_cmds] && [node1_cmds]
  #   BASH_REMATCH[1] = prefix (everything before first bracket)
  #   BASH_REMATCH[2] = comma-separated node0 commands
  #   BASH_REMATCH[3] = comma-separated node1 commands
  if [[ "$commands" =~ ^(.*)\[(.*)"] && ["(.*)\]$ ]]; then
    prefix=${BASH_REMATCH[1]//;/}
    echo "PREFIX: ${prefix}"

    export composite_command="(command rocm-smi || true)"
    saved_IFS=$IFS
    IFS=','
    read -ra node0 <<< "${BASH_REMATCH[2]}"
    read -ra node1 <<< "${BASH_REMATCH[3]}"
    IFS=$saved_IFS

    if [[ ${#node0[@]} -ne ${#node1[@]} ]]; then
      echo "Warning: node0 has ${#node0[@]} commands, node1 has ${#node1[@]}. They will be paired by index."
    fi

    for i in "${!node0[@]}"; do
      command_node_0=${node0[i]//\"/}
      command_node_1=${node1[i]//\"/}

      step_cmd="./.buildkite/scripts/run-multi-node-test.sh / 2 2 ${image_name} '${stage_workspace} && cd /vllm-workspace/tests && ${command_node_0}' '${stage_workspace} && cd /vllm-workspace/tests && ${command_node_1}'"
      echo "COMMANDS: ${step_cmd}"
      composite_command="${composite_command} && ${step_cmd}"
    done

    /bin/bash -c "${composite_command}"
    exit_code=$?
    handle_pytest_exit "$exit_code"
  else
    echo "Multi-node job detected but failed to parse bracket command syntax."
    echo "Expected format: prefix ; [node0_cmd1, node0_cmd2] && [node1_cmd1, node1_cmd2]"
    echo "Got: $commands"
    exit 111
  fi
else
  echo "--- Single-node job"
  echo "Render devices: $BUILDKITE_AGENT_META_DATA_RENDER_DEVICES"
  docker_run_terminal_args=(-i)
  if [[ "${ROCM_DOCKER_TTY}" == "1" ]]; then
    docker_run_terminal_args+=(-t)
    echo "Docker interactive stdin: enabled; TTY allocation: enabled"
  else
    echo "Docker interactive stdin: enabled; TTY allocation: disabled"
  fi

  ulimit_core_hard=$(ulimit -H -c)
  if [[ "$ulimit_core_hard" == "unlimited" ]]; then
    # docker run can't pass "unlimited" to --ulimit
    ulimit_core_hard="-1"
  fi
   # Disable core dumps in the ROCm test container unless the ROCm debug agent is enabled
  coredump_flags=(--ulimit "core=0:$ulimit_core_hard")
  if [[ "$commands" == *"ROCm debug agent enabled"* ]]; then
    # Works around https://github.com/rocm/rocm-systems/issues/6206
    coredump_flags=(-e 'HSA_COREDUMP_PATTERN="/tmp/gpucore.%p"')
  else
    echo "ROCm debug agent not enabled, coredumps are disabled in the test container."
  fi

  exit_code=0
  # shellcheck disable=SC2086  # word splitting is intentional: both hold multiple docker flags
  run_docker_with_ci_timeout docker run \
    "${docker_run_terminal_args[@]}" \
    --device /dev/kfd $BUILDKITE_AGENT_META_DATA_RENDER_DEVICES \
    $RDMA_FLAGS \
    --network=host \
    --shm-size=16gb \
    --group-add "$render_gid" \
    --rm \
    "${coredump_flags[@]}" \
    -e HF_TOKEN \
    -e "HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT}" \
    -e "HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT}" \
    -e AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY \
    -e BUILDKITE_PARALLEL_JOB \
    -e BUILDKITE_PARALLEL_JOB_COUNT \
    -e TERM \
    -e FORCE_COLOR \
    -e CLICOLOR_FORCE \
    -e PY_COLORS \
    -e PYTHONFAULTHANDLER \
    -e PYTEST_ADDOPTS \
    -e PYTEST_TIMEOUT \
    -v "${HF_CACHE}:${HF_MOUNT}" \
    -e "HF_HOME=${HF_MOUNT}" \
    -e "PYTHONPATH=${MYPYTHONPATH}" \
    -e "TMPDIR=${CONTAINER_TMPDIR}/tmp" \
    -e "TIKTOKEN_RS_CACHE_DIR=${HF_MOUNT}/tiktoken-rs-cache" \
    -e "TORCHINDUCTOR_CACHE_DIR=${CONTAINER_CACHE_ROOT}/torchinductor" \
    -e "TRITON_CACHE_DIR=${CONTAINER_CACHE_ROOT}/triton" \
    -e "VLLM_CACHE_ROOT=${CONTAINER_CACHE_ROOT}/vllm" \
    -e "XDG_CACHE_HOME=${CONTAINER_CACHE_ROOT}/xdg" \
    -e "PYTORCH_ROCM_ARCH=" \
    "${cpu_platform_env[@]}" \
    --name "${container_name}" \
    "${image_name}" \
    /bin/bash -c "${CONTAINER_PREFLIGHT} && ${commands}" || exit_code=$?
  handle_pytest_exit "${exit_code}"
fi
