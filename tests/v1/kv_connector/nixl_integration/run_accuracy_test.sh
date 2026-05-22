#!/bin/bash
set -xe

# Parse command line arguments
KV_BUFFER_DEVICE="cuda"  # Default to cuda
ATTENTION_BACKEND=""  # Default to empty (use vllm default)
CROSS_LAYERS_BLOCKS="False"
ENABLE_HMA_VAR=""  # Default to empty (HMA disabled by default for kv connector)
# Check for ENABLE_HMA_FLAG environment variable
if [[ -n "${ENABLE_HMA_FLAG:-}" ]]; then
  ENABLE_HMA_VAR="--no-disable-hybrid-kv-cache-manager"
fi

while [[ $# -gt 0 ]]; do
  case $1 in
    --kv_buffer_device)
      KV_BUFFER_DEVICE="$2"
      shift 2
      ;;
    --attention-backend)
      ATTENTION_BACKEND="$2"
      shift 2
      ;;
    --enable-cross-layers)
      CROSS_LAYERS_BLOCKS="True"
      shift 1
      ;;
    *)
      echo "Unknown option $1"
      echo "Usage: $0 [--kv_buffer_device <cuda|cpu>] [--attention-backend <backend>]"
      exit 1
      ;;
  esac
done

echo "Running accuracy tests with kv_buffer_device=$KV_BUFFER_DEVICE"
if [[ -n "$ATTENTION_BACKEND" ]]; then
  echo "Using attention backend: $ATTENTION_BACKEND"
fi
if [[ -n "$ENABLE_HMA_VAR" ]]; then
  echo "HMA (Hybrid KV Cache Manager) enabled"
fi
if [[ -n "$VLLM_SERVE_EXTRA_ARGS" ]]; then
  echo "vLLM serve extra args: $VLLM_SERVE_EXTRA_ARGS"
fi

DECODER_KV_LAYOUT=${DECODER_KV_LAYOUT:-"HND"} # Default to HND, optional NHD
if [[ "$DECODER_KV_LAYOUT" == "NHD" ]]; then
  KV_CONFIG_HETERO_LAYOUT=',"enable_permute_local_kv":"True"'
else
  KV_CONFIG_HETERO_LAYOUT=''
fi

if [[ "$CROSS_LAYERS_BLOCKS" == "True" ]]; then
  KV_EXTRA_CONFIG=',"kv_connector_extra_config":{"enable_cross_layers_blocks": "True"}'
else
  KV_EXTRA_CONFIG=''
fi

# Build the kv-transfer-config once
if [[ "$KV_BUFFER_DEVICE" == "cuda" ]]; then
  KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both"'${KV_CONFIG_HETERO_LAYOUT}${KV_EXTRA_CONFIG}'}'
else
  KV_CONFIG="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"$KV_BUFFER_DEVICE\""${KV_CONFIG_HETERO_LAYOUT}${KV_EXTRA_CONFIG}"}"
fi

# Models to run
MODEL_NAMES=${MODEL_NAMES:-}
if [[ -n "$MODEL_NAMES" ]]; then
  MODELS=("$MODEL_NAMES")
else
  MODELS=(
      "Qwen/Qwen3-0.6B"
  )
fi

# Number of prefill and decode instances to create
NUM_PREFILL_INSTANCES=${NUM_PREFILL_INSTANCES:-1} # Default to 1
NUM_DECODE_INSTANCES=${NUM_DECODE_INSTANCES:-1}   # Default to 1
PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:-1}
DECODER_TP_SIZE=${DECODER_TP_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.2}
PREFILL_BLOCK_SIZE=${PREFILL_BLOCK_SIZE:-128}
DECODE_BLOCK_SIZE=${DECODE_BLOCK_SIZE:-128}
PREFILL_INTERNAL_PORT_BASE=${PREFILL_INTERNAL_PORT_BASE:-30000}
DECODE_INTERNAL_PORT_BASE=${DECODE_INTERNAL_PORT_BASE:-31000}
# Comma-separated extra args for vllm serve (e.g. --max-model-len,2048)
VLLM_SERVE_EXTRA_ARGS=${VLLM_SERVE_EXTRA_ARGS:-}

# Resolve the repository root from the script location instead of `.git`.
# The ROCm CI image copies `/vllm-workspace` without the Git metadata, so
# `git rev-parse --show-toplevel` is not reliable at runtime.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GIT_ROOT="${GIT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd -P)}"

SMI_BIN=$(which nvidia-smi || which rocm-smi || echo "")

device_visibility_env() {
  local gpu_ids="$1"
  if [[ "$SMI_BIN" == *"rocm"* ]]; then
    # Do not set ROCR_VISIBLE_DEVICES together with HIP/CUDA here. ROCr
    # applies its filter below the HIP runtime, so setting both to non-zero
    # physical ids double-filters the device list and can hide the target GPUs.
    echo "-u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=$gpu_ids CUDA_VISIBLE_DEVICES=$gpu_ids"
  else
    echo "CUDA_VISIBLE_DEVICES=$gpu_ids"
  fi
}

SERVER_PIDS=()
PROXY_PID=""

# Trap the SIGINT signal (triggered by Ctrl+C)
trap cleanup_instances SIGINT SIGTERM EXIT

# Waits for vLLM to start.
wait_for_server() {
  local port=$1
  local server_pid=${2:-}
  local server_name=${3:-server}
  local endpoint=${4:-/v1/completions}
  local deadline=${5:-1200}
  local elapsed=0
  while [ $elapsed -lt $deadline ]; do
    if [[ -n "$server_pid" ]] && ! ps -p "$server_pid" > /dev/null 2>&1; then
      local status=0
      wait "$server_pid" || status=$?
      echo "FAIL: ${server_name} process ${server_pid} exited with status ${status} before port ${port} became ready"
      return 1
    fi
    if curl -s "localhost:${port}${endpoint}" > /dev/null; then
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  echo "FAIL: ${server_name} on port ${port} did not start within ${deadline}s"
  return 1
}

# Function to clean up previous instances
cleanup_instances() {
  echo "Cleaning up any running vLLM instances..."
  local pids=()
  if [[ -n "$PROXY_PID" ]]; then
    pids+=("$PROXY_PID")
  fi
  if [[ ${#SERVER_PIDS[@]} -gt 0 ]]; then
    pids+=("${SERVER_PIDS[@]}")
  fi

  for pid in "${pids[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "${pids[@]}"; do
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  done
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done

  pkill -f "vllm serve" || true
  pkill -f "toy_proxy_server.py.*--port 8192" || true
  sleep 2
  SERVER_PIDS=()
  PROXY_PID=""
}

get_num_gpus() {
  if [[ "$SMI_BIN" == *"nvidia"* ]]; then
    $SMI_BIN --query-gpu=name --format=csv,noheader | wc -l
  elif [[ "$SMI_BIN" == *"rocm"* ]]; then
    $SMI_BIN -l | grep -c GPU
  else
    # works for non-cuda platforms,
    # assuming at least 1 device and
    # let system to decide which card to use
    echo "1"
  fi
}

# Function to run tests for a specific model
run_tests_for_model() {
  local model_name=$1
  echo "================================"
  echo "Testing model: $model_name"
  echo "================================"

  # Arrays to store all hosts and ports
  PREFILL_HOSTS=()
  PREFILL_PORTS=()
  PREFILL_PIDS=()
  DECODE_HOSTS=()
  DECODE_PORTS=()
  DECODE_PIDS=()

  # Start prefill instances
  for i in $(seq 0 $((NUM_PREFILL_INSTANCES-1))); do
    # Calculate GPU ID - we'll distribute across available GPUs
    GPU_ID=$((i % $(get_num_gpus)))
    NEXT_GPU=${GPU_ID}
    # If PREFILLER_TP_SIZE is more than 1
    for (( j=1; j < PREFILLER_TP_SIZE; j++ )); do
      NEXT_GPU=$(((GPU_ID + j) % $(get_num_gpus)))
      GPU_ID="${GPU_ID},${NEXT_GPU}"
    done

    # Calculate port number (base port + instance number)
    PORT=$((8100 + i))
    # Calculate side channel port. Avoid clash with with TP workers.
    SIDE_CHANNEL_PORT=$((5559 + i))
    INTERNAL_PORT_BASE=$((PREFILL_INTERNAL_PORT_BASE + i * 100))
    GPU_ENV="$(device_visibility_env "$GPU_ID")"

    echo "Starting prefill instance $i on GPU $GPU_ID, port $PORT"

    # Build the command with or without model-specific args
    BASE_CMD="env $GPU_ENV \
    VLLM_PORT=$INTERNAL_PORT_BASE \
    VLLM_KV_CACHE_LAYOUT='HND' \
    UCX_NET_DEVICES=all \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT \
    vllm serve $model_name \
    --port $PORT \
    --enforce-eager \
    --block-size ${PREFILL_BLOCK_SIZE} \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --tensor-parallel-size $PREFILLER_TP_SIZE \
    --kv-transfer-config '$KV_CONFIG'"
    if [[ -n "$VLLM_SERVE_EXTRA_ARGS" ]]; then
      IFS=',' read -r -a extra_args <<< "$VLLM_SERVE_EXTRA_ARGS"
      for arg in "${extra_args[@]}"; do
        BASE_CMD="${BASE_CMD} $arg"
      done
    fi

    # Add attention backend config if specified
    if [[ -n "$ATTENTION_BACKEND" ]]; then
      BASE_CMD="${BASE_CMD} --attention-backend=$ATTENTION_BACKEND"
    fi

    # Add HMA flag if specified
    if [[ -n "$ENABLE_HMA_VAR" ]]; then
      BASE_CMD="${BASE_CMD} $ENABLE_HMA_VAR"
    fi
    
    FULL_CMD="$BASE_CMD"
    setsid bash -c "$FULL_CMD" &
    SERVER_PID="$!"
    SERVER_PIDS+=("$SERVER_PID")
    PREFILL_PIDS+=("$SERVER_PID")

    # Store host and port for proxy configuration
    PREFILL_HOSTS+=("localhost")
    PREFILL_PORTS+=("$PORT")
  done

  # Start decode instances
  for i in $(seq 0 $((NUM_DECODE_INSTANCES-1))); do
    # Calculate GPU ID - we'll distribute across available GPUs, starting from after prefill GPUs
    GPU_ID=$(((i + NEXT_GPU + 1) % $(get_num_gpus)))
    # If DECODER_TP_SIZE is more than 1
    for (( j=1; j < DECODER_TP_SIZE; j++ )); do
      NEXT_GPU=$(((GPU_ID + j) % $(get_num_gpus)))
      GPU_ID="${GPU_ID},${NEXT_GPU}"
    done
    # Calculate port number (base port + instance number)
    PORT=$((8200 + i))
    # Calculate side channel port
    SIDE_CHANNEL_PORT=$((5659 + i * $DECODER_TP_SIZE))
    INTERNAL_PORT_BASE=$((DECODE_INTERNAL_PORT_BASE + i * 100))
    GPU_ENV="$(device_visibility_env "$GPU_ID")"

    echo "Starting decode instance $i on GPU $GPU_ID, port $PORT"

    # Build the command with or without model-specific args
    BASE_CMD="env $GPU_ENV \
    VLLM_PORT=$INTERNAL_PORT_BASE \
    VLLM_KV_CACHE_LAYOUT=$DECODER_KV_LAYOUT \
    UCX_NET_DEVICES=all \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT \
    vllm serve $model_name \
    --port $PORT \
    --enforce-eager \
    --block-size ${DECODE_BLOCK_SIZE} \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --kv-transfer-config '$KV_CONFIG'"
    if [[ -n "$VLLM_SERVE_EXTRA_ARGS" ]]; then
      IFS=',' read -r -a extra_args <<< "$VLLM_SERVE_EXTRA_ARGS"
      for arg in "${extra_args[@]}"; do
        BASE_CMD="${BASE_CMD} $arg"
      done
    fi

    # Add attention backend config if specified
    if [[ -n "$ATTENTION_BACKEND" ]]; then
      BASE_CMD="${BASE_CMD} --attention-backend=$ATTENTION_BACKEND"
    fi

    # Add HMA flag if specified
    if [[ -n "$ENABLE_HMA_VAR" ]]; then
      BASE_CMD="${BASE_CMD} $ENABLE_HMA_VAR"
    fi

  # DP-EP attention mode
  if [[ -z "$DP_EP" ]]; then
    BASE_CMD="${BASE_CMD} --tensor-parallel-size $DECODER_TP_SIZE"
  else
    echo "DP-EP Attention enabled, deploying with dp=DECODER_TP_SIZE and tp=1"
    BASE_CMD="${BASE_CMD} --data-parallel-size $DECODER_TP_SIZE \
    --tensor-parallel-size 1 --enable-expert-parallel"
  fi

    FULL_CMD="$BASE_CMD"

    setsid bash -c "$FULL_CMD" &
    SERVER_PID="$!"
    SERVER_PIDS+=("$SERVER_PID")
    DECODE_PIDS+=("$SERVER_PID")

    # Store host and port for proxy configuration
    DECODE_HOSTS+=("localhost")
    DECODE_PORTS+=("$PORT")
  done

  # Wait for all instances to start
  for idx in "${!PREFILL_PORTS[@]}"; do
    PORT="${PREFILL_PORTS[$idx]}"
    PID="${PREFILL_PIDS[$idx]}"
    echo "Waiting for prefill instance on port $PORT to start..."
    wait_for_server "$PORT" "$PID" "prefill"
  done

  for idx in "${!DECODE_PORTS[@]}"; do
    PORT="${DECODE_PORTS[$idx]}"
    PID="${DECODE_PIDS[$idx]}"
    echo "Waiting for decode instance on port $PORT to start..."
    wait_for_server "$PORT" "$PID" "decode"
  done

  # Build the command for the proxy server with all the hosts and ports
  PROXY_CMD="python3 ${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py --port 8192"

  # Add all prefill hosts and ports
  PROXY_CMD+=" --prefiller-hosts ${PREFILL_HOSTS[*]}"
  PROXY_CMD+=" --prefiller-ports ${PREFILL_PORTS[*]}"

  # Add all decode hosts and ports
  PROXY_CMD+=" --decoder-hosts ${DECODE_HOSTS[*]}"
  PROXY_CMD+=" --decoder-ports ${DECODE_PORTS[*]}"

  # Start the proxy server
  echo "Starting proxy server with command: $PROXY_CMD"
  setsid bash -c "$PROXY_CMD" &
  PROXY_PID="$!"

  # Wait for the proxy to start
  sleep 5

  # Run lm eval for this model
  echo "Running tests for $model_name"
  TEST_MODEL=$model_name python3 -m pytest -s -x "${GIT_ROOT}"/tests/v1/kv_connector/nixl_integration/test_accuracy.py

  # Clean up before running next model
  cleanup_instances
  sleep 3
}

# Run tests for each model
for model in "${MODELS[@]}"; do
  run_tests_for_model "$model"
done

echo "All tests completed!"
