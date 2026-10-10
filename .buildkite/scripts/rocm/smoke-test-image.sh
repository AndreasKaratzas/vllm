#!/usr/bin/env bash
# Fast structural smoke test for the full ROCm CI image.

set -euo pipefail

run_smoke_checks() {
    local required_dir=""

    local workspace=/opt/vllm-ci/workspace
    local vllm_dir=""

    for required_dir in \
        "${workspace}/tests" \
        "${workspace}/.buildkite" \
        "${workspace}/requirements"; do
        if [[ ! -d "${required_dir}" ]]; then
            echo "Missing directory: ${required_dir}" >&2
            return 1
        fi
    done
    if [[ ! -s /opt/vllm-ci/commit.txt || ! -s /opt/vllm-ci/wheel-filename.txt ]]; then
        echo "Missing /opt/vllm-ci build metadata" >&2
        return 1
    fi
    vllm_dir=$(PYTHONDONTWRITEBYTECODE=1 python3 -c \
        'import importlib.util as u; print(u.find_spec("vllm").submodule_search_locations[0])')
    if [[ ! -x "${vllm_dir}/vllm-rs" ]]; then
        echo "Missing executable: ${vllm_dir}/vllm-rs" >&2
        return 1
    fi

    command -v python3
    command -v uv
    command -v pytest

    if ! command -v amd-smi >/dev/null 2>&1 \
        && ! command -v rocminfo >/dev/null 2>&1; then
        echo "No ROCm CLI found in image" >&2
        return 1
    fi

    PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import torch
import vllm

print(torch.__version__)
print(vllm.__version__)
PY

    echo "AMD image smoke OK"
}

if [[ "${1:-}" == "--inside" ]]; then
    run_smoke_checks
    exit
fi
if (($#)); then
    echo "Usage: $0 [--inside]" >&2
    exit 2
fi

smoke_marker="./build/rocm-smoke-export/vllm-smoke-ok"
expected_smoke_id="${BUILDKITE_BUILD_ID:-local}"
if [[ -f "${smoke_marker}" \
    && ( -z "${VLLM_CI_SMOKE_IMAGE:-}" \
        || "${VLLM_CI_SMOKE_IMAGE}" == "${IMAGE_TAG:-}" ) ]]; then
    actual_smoke_id="$(< "${smoke_marker}")"
    if [[ "${actual_smoke_id}" != "${expected_smoke_id}" ]]; then
        echo "ROCm smoke marker belongs to ${actual_smoke_id}, not ${expected_smoke_id}" \
            >&2
        exit 1
    fi
    echo "AMD image smoke OK (verified inside BuildKit)"
    rm -f -- "${smoke_marker}"
    rmdir -- "$(dirname "${smoke_marker}")" 2>/dev/null || true
    exit
fi

image_ref="${VLLM_CI_SMOKE_IMAGE:-${IMAGE_TAG:-rocm/vllm-ci:${BUILDKITE_COMMIT:?BUILDKITE_COMMIT is required}}}"

docker run --rm -i --network=none --entrypoint /bin/bash "${image_ref}" \
    -s -- --inside < "${BASH_SOURCE[0]}"
