#!/bin/bash

# This script runs tests inside the corresponding ROCm docker container.
# It handles both single-node and multi-node test configurations.
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

# Export Python path
export PYTHONPATH=".."

###############################################################################
# Helper Functions
###############################################################################

cleanup_docker() {
  # Get Docker's root directory
  docker_root=$(docker info -f '{{.DockerRootDir}}')
  if [ -z "$docker_root" ]; then
    echo "Failed to determine Docker root directory."
    exit 1
  fi
  echo "Docker root directory: $docker_root"

  docker_root_for_df="$docker_root"
  while [ ! -e "$docker_root_for_df" ] && [ "$docker_root_for_df" != "/" ]; do
    docker_root_for_df=$(dirname "$docker_root_for_df")
  done

  if [ "$docker_root_for_df" != "$docker_root" ]; then
    echo "Docker root path does not exist on host; using $docker_root_for_df for disk usage."
  fi

  disk_usage=$(df -P "$docker_root_for_df" 2>/dev/null | tail -1 | awk '{print $5}' | sed 's/%//')
  if ! [[ "$disk_usage" =~ ^[0-9]+$ ]]; then
    echo "Unable to determine Docker disk usage. Skipping Docker cleanup."
    return
  fi

  threshold=70
  if [ "$disk_usage" -gt "$threshold" ]; then
    echo "Disk usage is above $threshold%. Cleaning up Docker images and volumes..."
    docker image prune -f
    docker volume prune -f && docker system prune --force --filter "until=72h" --all
    echo "Docker images and volumes cleanup completed."
  else
    echo "Disk usage is below $threshold%. No cleanup needed."
  fi
}

cleanup_network() {
  local max_nodes=${NUM_NODES:-2}
  for node in $(seq 0 $((max_nodes - 1))); do
    if docker ps -a -q -f name="node${node}" | grep -q .; then
      docker stop "node${node}" || true
    fi
  done
  if docker network ls | grep -q docker-net; then
    docker network rm docker-net || true
  fi
}

assigned_rocm_cards() {
  local render_devices="${BUILDKITE_AGENT_META_DATA_RENDER_DEVICES:-}"
  if [[ -z "$render_devices" ]]; then
    return
  fi

  python3 - "$render_devices" <<'PY'
import json
import os
import re
import subprocess
import sys

render_devices = sorted(set(re.findall(r"/dev/dri/(renderD[0-9]+)", sys.argv[1])))
if not render_devices:
    sys.exit(0)

render_unique_ids = set()
for render_dev in render_devices:
    unique_id_path = f"/sys/class/drm/{render_dev}/device/unique_id"
    try:
        with open(unique_id_path, encoding="utf-8") as f:
            unique_id = f.read().strip().lower()
    except OSError:
        continue
    if unique_id:
        render_unique_ids.add("0x" + unique_id.lstrip("0x"))

try:
    result = subprocess.run(
        ["rocm-smi", "--showuniqueid", "--json"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    cards = json.loads(result.stdout)
except Exception:
    sys.exit(0)

for card, values in sorted(cards.items()):
    unique_id = str(values.get("Unique ID", "")).lower()
    if unique_id in render_unique_ids:
        print(card)
PY
}

assigned_gpu_vram_usage() {
  local cards="$1"
  if [[ -z "$cards" ]]; then
    return 1
  fi

  python3 - "$cards" <<'PY'
import json
import subprocess
import sys

cards = set(sys.argv[1].split())
try:
    result = subprocess.run(
        ["rocm-smi", "--showmemuse", "--json"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    mem = json.loads(result.stdout)
except Exception:
    sys.exit(1)

usages = []
for card in sorted(cards):
    value = mem.get(card, {}).get("GPU Memory Allocated (VRAM%)")
    try:
        usages.append(int(str(value).strip()))
    except (TypeError, ValueError):
        pass

if not usages:
    sys.exit(1)

print(max(usages))
PY
}

assigned_gpu_pids() {
  local cards="$1"
  if [[ -z "$cards" ]]; then
    return 1
  fi

  python3 - "$cards" <<'PY'
import json
import os
import re
import subprocess
import sys

card_ids = {
    card.removeprefix("card")
    for card in sys.argv[1].split()
    if card.removeprefix("card").isdigit()
}
if not card_ids:
    sys.exit(0)

current_pids = {os.getpid(), os.getppid()}
direct_matches: list[int] = []
busy_pids: list[int] = []
seen: set[int] = set()


def record_pid(pid: int, proc_gpus: str, vram_used: int) -> None:
    if pid in current_pids or pid in seen:
        return
    if vram_used <= 0:
        return

    seen.add(pid)
    busy_pids.append(pid)
    proc_cards = set(re.findall(r"[0-9]+", proc_gpus))
    if proc_cards & card_ids:
        direct_matches.append(pid)


try:
    result = subprocess.run(
        ["rocm-smi", "--showpids", "--json"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    data = json.loads(result.stdout)
except Exception:
    data = {}

for key, raw_value in data.get("system", {}).items():
    match = re.fullmatch(r"PID([0-9]+)", str(key))
    if match is None:
        continue
    fields = [field.strip() for field in str(raw_value).split(",")]
    if len(fields) < 3:
        continue
    try:
        vram_used = int(fields[2])
    except ValueError:
        vram_used = 0
    record_pid(int(match.group(1)), fields[1], vram_used)

# Some ROCm-SMI builds return "WARNING: No JSON data to report" for
# `--showpids --json`. Parse the plain text table as a fallback so the idle
# gate can still clean stale KFD processes before starting a test container.
if not seen:
    try:
        result = subprocess.run(
            ["rocm-smi", "--showpids"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        result = None

    if result is not None:
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) < 5 or not fields[0].isdigit():
                continue
            try:
                vram_used = int(fields[3])
            except ValueError:
                continue
            record_pid(int(fields[0]), fields[2], vram_used)

if direct_matches:
    print(*direct_matches, sep="\n")
    sys.exit(0)

# Some ROCm-SMI versions report the process table GPU column using a KFD node
# id that does not match the JSON card key. For a one-card job with exactly one
# busy KFD process, that process is still the contaminated assigned device.
if len(card_ids) == 1 and len(busy_pids) == 1:
    print(busy_pids[0])
PY
}

cleanup_assigned_gpu_processes() {
  local cards="$1"
  local pids
  pids="$(assigned_gpu_pids "$cards" | tr '\n' ' ' | sed 's/[[:space:]]*$//' || true)"
  if [[ -z "$pids" ]]; then
    echo "No KFD processes with VRAM allocations found on assigned ROCm cards."
    return 1
  fi

  echo "Attempting to terminate KFD process(es) on assigned ROCm cards: ${pids}"
  # shellcheck disable=SC2086
  kill -TERM $pids 2>/dev/null || true
  sleep "${VLLM_CI_GPU_IDLE_TERM_GRACE_SECONDS:-10}"

  local alive=()
  for pid in $pids; do
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid")
    fi
  done

  if [[ ${#alive[@]} -gt 0 ]]; then
    echo "KFD process(es) still alive after SIGTERM, sending SIGKILL: ${alive[*]}"
    kill -KILL "${alive[@]}" 2>/dev/null || true
  fi
}

wait_for_assigned_gpus_idle() {
  local cards
  cards="$(assigned_rocm_cards | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
  if [[ -z "$cards" ]]; then
    if [[ -n "${BUILDKITE_AGENT_META_DATA_RENDER_DEVICES:-}" ]]; then
      echo "Could not map assigned render devices to ROCm cards."
      echo "Render devices: ${BUILDKITE_AGENT_META_DATA_RENDER_DEVICES}"
      return 1
    fi
    echo "No assigned render-device metadata; skipping GPU idle wait."
    return
  fi

  local threshold="${VLLM_CI_GPU_IDLE_VRAM_THRESHOLD:-10}"
  local timeout="${VLLM_CI_GPU_IDLE_WAIT_SECONDS:-900}"
  local interval="${VLLM_CI_GPU_IDLE_POLL_SECONDS:-15}"
  local start
  start=$(date +%s)

  echo "Waiting for assigned ROCm cards to be idle: ${cards} (VRAM <= ${threshold}%)."
  while true; do
    local usage
    usage="$(assigned_gpu_vram_usage "$cards" || true)"
    if [[ "$usage" =~ ^[0-9]+$ ]] && [ "$usage" -le "$threshold" ]; then
      echo "Assigned ROCm cards are idle enough: max VRAM ${usage}%."
      return
    fi

    local now elapsed
    now=$(date +%s)
    elapsed=$((now - start))
    if [ "$elapsed" -ge "$timeout" ]; then
      echo "Assigned ROCm cards did not become idle within ${timeout}s; max VRAM ${usage:-unknown}%."
      rocm-smi --showmemuse || true
      rocm-smi --showpids || true

      if [[ "${VLLM_CI_GPU_IDLE_KILL_BUSY_PROCS:-1}" == "1" ]]; then
        cleanup_assigned_gpu_processes "$cards"
        usage="$(assigned_gpu_vram_usage "$cards" || true)"
        if [[ "$usage" =~ ^[0-9]+$ ]] && [ "$usage" -le "$threshold" ]; then
          echo "Assigned ROCm cards became idle after KFD process cleanup: max VRAM ${usage}%."
          return
        fi
        echo "Assigned ROCm cards are still busy after cleanup: max VRAM ${usage:-unknown}%."
        rocm-smi --showmemuse || true
        rocm-smi --showpids || true
      fi
      return 1
    fi

    echo "Assigned ROCm cards still busy: max VRAM ${usage:-unknown}% (${elapsed}s/${timeout}s)."
    sleep "$interval"
  done
}

is_multi_node() {
  local cmds="$1"
  # Primary signal: NUM_NODES environment variable set by the pipeline
  if [[ "${NUM_NODES:-1}" -gt 1 ]]; then
    return 0
  fi
  # Fallback: detect the bracket syntax structurally
  # Pattern: [...] && [...] (per-node command arrays)
  if [[ "$cmds" =~ \[.*\].*\&\&.*\[.*\] ]]; then
    return 0
  fi
  return 1
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
# through untouched to avoid double-quoting values injected by
# apply_rocm_test_overrides.
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

###############################################################################
# ROCm-specific pytest command rewrites
#
# These apply ignore flags and environment overrides for tests that are not
# yet supported or behave differently on ROCm hardware. Kept as a single
# function so new exclusions are easy to add in one place.
###############################################################################

apply_rocm_test_overrides() {
  local cmds="$1"

  # --- Model registry filter ---
  if [[ $cmds == *"pytest -v -s models/test_registry.py"* ]]; then
    cmds=${cmds//"pytest -v -s models/test_registry.py"/"pytest -v -s models/test_registry.py -k 'not BambaForCausalLM and not GritLM and not Mamba2ForCausalLM and not Zamba2ForCausalLM'"}
  fi

  # --- LoRA: disable custom paged attention ---
  if [[ $cmds == *"pytest -v -s lora"* ]]; then
    cmds=${cmds//"pytest -v -s lora"/"pytest -v -s lora"}
  fi

  # --- Kernel ignores ---
  if [[ $cmds == *" kernels/core"* ]]; then
    cmds="${cmds} \
    --ignore=kernels/core/test_fused_quant_layernorm.py \
    --ignore=kernels/core/test_permute_cols.py"
  fi

  if [[ $cmds == *" kernels/attention"* ]]; then
    cmds="${cmds} \
    --ignore=kernels/attention/test_attention_selector.py \
    --ignore=kernels/attention/test_encoder_decoder_attn.py \
    --ignore=kernels/attention/test_flash_attn.py \
    --ignore=kernels/attention/test_flashinfer.py \
    --ignore=kernels/attention/test_prefix_prefill.py \
    --ignore=kernels/attention/test_cascade_flash_attn.py \
    --ignore=kernels/attention/test_mha_attn.py \
    --ignore=kernels/attention/test_lightning_attn.py \
    --ignore=kernels/attention/test_attention.py"
  fi

  if [[ $cmds == *" kernels/quantization"* ]]; then
    cmds="${cmds} \
    --ignore=kernels/quantization/test_int8_quant.py \
    --ignore=kernels/quantization/test_machete_mm.py \
    --ignore=kernels/quantization/test_block_fp8.py \
    --ignore=kernels/quantization/test_block_int8.py \
    --ignore=kernels/quantization/test_marlin_gemm.py \
    --ignore=kernels/quantization/test_cutlass_scaled_mm.py \
    --ignore=kernels/quantization/test_int8_kernel.py"
  fi

  if [[ $cmds == *" kernels/mamba"* ]]; then
    cmds="${cmds} \
    --ignore=kernels/mamba/test_mamba_mixer2.py \
    --ignore=kernels/mamba/test_causal_conv1d.py \
    --ignore=kernels/mamba/test_mamba_ssm_ssd.py"
  fi

  if [[ $cmds == *" kernels/moe"* ]]; then
    cmds="${cmds} \
    --ignore=kernels/moe/test_moe.py \
    --ignore=kernels/moe/test_cutlass_moe.py"
  fi

  # --- Entrypoint ignores ---
  if [[ $cmds == *" entrypoints/openai "* ]]; then
    cmds=${cmds//" entrypoints/openai "/" entrypoints/openai \
    --ignore=entrypoints/openai/chat_completion/test_audio.py \
    --ignore=entrypoints/openai/completion/test_shutdown.py \
    --ignore=entrypoints/openai/test_completion.py \
    --ignore=entrypoints/openai/models/test_models.py \
    --ignore=entrypoints/openai/test_return_tokens_as_ids.py \
    --ignore=entrypoints/openai/chat_completion/test_root_path.py \
    --ignore=entrypoints/openai/completion/test_prompt_validation.py "}
  fi

  if [[ $cmds == *" entrypoints/serve"* ]]; then
    cmds="${cmds} \
    --ignore=entrypoints/serve/lora/test_lora_adapters.py"
  fi

  if [[ $cmds == *" entrypoints/llm "* ]]; then
    cmds=${cmds//" entrypoints/llm "/" entrypoints/llm \
    --ignore=entrypoints/llm/test_chat.py \
    --ignore=entrypoints/llm/test_accuracy.py \
    --ignore=entrypoints/llm/test_init.py \
    --ignore=entrypoints/llm/test_prompt_validation.py "}
  fi

  # Clean up escaped newlines from --ignore appends
  cmds=$(echo "$cmds" | sed 's/ \\ / /g')

  echo "$cmds"
}

###############################################################################
# Main
###############################################################################

# --- GPU initialization ---
echo "--- ROCm info"
rocminfo

# --- Docker housekeeping ---
cleanup_docker

# --- Pull test image ---
echo "--- Pulling container"
image_name="rocm/vllm-ci:${BUILDKITE_COMMIT}"
container_name="rocm_${BUILDKITE_COMMIT}_$(tr -dc A-Za-z0-9 < /dev/urandom | head -c 10; echo)"
docker pull "${image_name}"

remove_docker_container() {
  docker rm -f "${container_name}" || docker image rm -f "${image_name}" || true
}
trap remove_docker_container EXIT

# --- Prepare commands ---
echo "--- Running container"

HF_CACHE="$(realpath ~)/huggingface"
mkdir -p "${HF_CACHE}"
HF_MOUNT="/root/.cache/huggingface"

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

commands=$(apply_rocm_test_overrides "$commands")
echo "Final commands: $commands"

MYPYTHONPATH=".."

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

if ! wait_for_assigned_gpus_idle; then
  echo "Assigned ROCm cards are still busy; refusing to start tests on contaminated GPUs."
  exit 1
fi

# --- Route: multi-node vs single-node ---
if is_multi_node "$commands"; then
  echo "--- Multi-node job detected"
  export DCKR_VER=$(docker --version | sed 's/Docker version \(.*\), build .*/\1/')

  # Parse the bracket syntax:  prefix ; [node0_cmds] && [node1_cmds]
  #   BASH_REMATCH[1] = prefix (everything before first bracket)
  #   BASH_REMATCH[2] = comma-separated node0 commands
  #   BASH_REMATCH[3] = comma-separated node1 commands
  if [[ "$commands" =~ ^(.*)\[(.*)"] && ["(.*)\]$ ]]; then
    prefix=$(echo "${BASH_REMATCH[1]}" | sed 's/;//g')
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
      command_node_0=$(echo "${node0[i]}" | sed 's/\"//g')
      command_node_1=$(echo "${node1[i]}" | sed 's/\"//g')

      step_cmd="./.buildkite/scripts/run-multi-node-test.sh /vllm-workspace/tests 2 2 ${image_name} '${command_node_0}' '${command_node_1}'"
      echo "COMMANDS: ${step_cmd}"
      composite_command="${composite_command} && ${step_cmd}"
    done

    /bin/bash -c "${composite_command}"
    exit_code=$?
    cleanup_network
    handle_pytest_exit "$exit_code"
  else
    echo "Multi-node job detected but failed to parse bracket command syntax."
    echo "Expected format: prefix ; [node0_cmd1, node0_cmd2] && [node1_cmd1, node1_cmd2]"
    echo "Got: $commands"
    cleanup_network
    exit 111
  fi
else
  echo "--- Single-node job"
  echo "Render devices: $BUILDKITE_AGENT_META_DATA_RENDER_DEVICES"

  # Let the EXIT trap own container cleanup. With docker run --rm, an external
  # cleanup race can make docker report "No such container" after tests pass.
  docker run \
    --device /dev/kfd $BUILDKITE_AGENT_META_DATA_RENDER_DEVICES \
    $RDMA_FLAGS \
    --network=host \
    --shm-size=16gb \
    --group-add "$render_gid" \
    -e HF_TOKEN \
    -e AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY \
    -e BUILDKITE_PARALLEL_JOB \
    -e BUILDKITE_PARALLEL_JOB_COUNT \
    -v "${HF_CACHE}:${HF_MOUNT}" \
    -e "HF_HOME=${HF_MOUNT}" \
    -e "PYTHONPATH=${MYPYTHONPATH}" \
    -e "PYTORCH_ROCM_ARCH=" \
    --name "${container_name}" \
    "${image_name}" \
    /bin/bash -c "${commands}"

  exit_code=$?
  handle_pytest_exit "$exit_code"
fi
