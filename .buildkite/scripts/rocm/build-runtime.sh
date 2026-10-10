#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Ensure the ROCm runtime images for this commit exist, building only what is
# missing, and hand their digests to the per-commit image build.
#
#   docker/Dockerfile.rocm_base
#     wheel_* / rocm_runtime_libs / rdma_core_tree   one artifact per source dependency
#     runtime      SDK, torch, wheelhouse, requirements: the deployment base and
#                  the environment the vLLM wheel is compiled in
#     runtime-ci   runtime plus test dependencies (its lower layers are runtime's)
#
# Every image is tagged by the content key of its Dockerfile stage closure
# (stage_key.py), so a tag is reused by every build with the same inputs and
# never rebuilt. Builds are reproducible (SOURCE_DATE_EPOCH, rewrite-timestamp),
# so even a rebuild republishes identical layers.
#
# Trust: only main-branch builds of vllm-project/vllm write and read the
# trusted tags; every other build writes "pr-" tags and may read trusted ones.
#
# Env: ROCM_CI_IMAGE_REPO (default rocm/vllm-ci), PYTORCH_ROCM_ARCH,
#      ROCM_BUILDER_NAME (default vllm-rocm), ROCM_BUILDKIT_IMAGE.
# Meta-data out: rocm-runtime-image=<repo>:runtime-<key>@sha256:...
#                rocm-runtime-ci-image=<repo>:runtime-ci-<key>@sha256:...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="${ROCM_BASE_DOCKERFILE:-docker/Dockerfile.rocm_base}"
REPO="${ROCM_CI_IMAGE_REPO:-rocm/vllm-ci}"
BUILDER="${ROCM_BUILDER_NAME:-vllm-rocm}"
ARTIFACT_STAGES=(
    wheel_triton wheel_fa wheel_aiter wheel_mori wheel_nixl
    wheel_deepep wheel_torchcodec wheel_lmcache wheel_fastsafetensors
    rocm_runtime_libs rdma_core_tree
)
OUTPUT_ATTRS="oci-mediatypes=true,compression=zstd,compression-level=3,force-compression=true,rewrite-timestamp=true"

SOURCE_DATE_EPOCH=$(sed -nE 's/^ARG SOURCE_DATE_EPOCH=([0-9]+)$/\1/p' "${DOCKERFILE}" | head -1)
BUILD_ARGS=(
    --build-arg "PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx90a;gfx942;gfx950}"
    --build-arg "SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:?SOURCE_DATE_EPOCH missing in ${DOCKERFILE}}"
)

is_trusted_build() {
    [[ "${BUILDKITE:-false}" == "true" \
        && "${BUILDKITE_PULL_REQUEST:-false}" == "false" \
        && "${BUILDKITE_BRANCH:-}" == "main" \
        && "${BUILDKITE_REPO:-}" =~ [:/]vllm-project/vllm(\.git)?$ ]]
}

if is_trusted_build; then
    WRITE_PREFIX=""
    READ_PREFIXES=("")
else
    WRITE_PREFIX="pr-"
    READ_PREFIXES=("" "pr-")
fi

remote_digest() {
    docker buildx imagetools inspect "$1" --format '{{json .Manifest}}' 2>/dev/null \
        | python3 -c 'import json, sys; print(json.load(sys.stdin)["digest"])'
}

setup_builder() {
    if ! docker buildx inspect "${BUILDER}" >/dev/null 2>&1; then
        docker buildx create --name "${BUILDER}" --driver docker-container \
            --driver-opt "image=${ROCM_BUILDKIT_IMAGE:-moby/buildkit:v0.33.1}" \
            --buildkitd-config "${SCRIPT_DIR}/buildkitd.toml" >/dev/null
    fi
    docker buildx inspect --builder "${BUILDER}" --bootstrap | grep -E '^(Name|BuildKit version):'
}

stage_key() {
    python3 "${SCRIPT_DIR}/stage_key.py" "${DOCKERFILE}" "$1" "${BUILD_ARGS[@]}" "${@:2}"
}

# ensure_image <stage> <tag-name> [extra build flags...] -> prints ref@digest
ensure_image() {
    local stage="$1" name="$2" key prefix ref digest
    shift 2
    key=$(stage_key "${stage}" "${KEY_CONTEXTS[@]}" | cut -c1-24)
    for prefix in "${READ_PREFIXES[@]}"; do
        ref="${REPO}:${prefix}${name}-${key}"
        if digest=$(remote_digest "${ref}"); then
            echo "reuse ${ref}@${digest}" >&2
            echo "${ref}@${digest}"
            return 0
        fi
    done
    ref="${REPO}:${WRITE_PREFIX}${name}-${key}"
    echo "--- :docker: Building ${stage} -> ${ref}" >&2
    docker buildx build --builder "${BUILDER}" --provenance=false --sbom=false \
        --progress "${BUILDKIT_PROGRESS:-plain}" \
        -f "${DOCKERFILE}" --target "${stage}" "${BUILD_ARGS[@]}" "$@" \
        --label "vllm.rocm.stage=${stage}" --label "vllm.rocm.key=${key}" \
        --output "type=registry,name=${ref},${OUTPUT_ATTRS}" . >&2
    digest=$(remote_digest "${ref}")
    echo "${ref}@${digest}"
}

main() {
    local stage ref runtime_ref runtime_ci_ref
    local -a contexts=()
    KEY_CONTEXTS=()

    bash "${SCRIPT_DIR}/prune-agent.sh" || true
    setup_builder

    # Artifacts first: each is keyed by its own build closure.
    for stage in "${ARTIFACT_STAGES[@]}"; do
        ref=$(ensure_image "${stage}" "dep-${stage}")
        contexts+=(--build-context "${stage}=docker-image://${ref}")
        KEY_CONTEXTS+=(--context "${stage}=${ref#*@}")
    done

    # The runtimes are keyed by the artifact digests plus their own inputs and
    # built from those artifacts, never from the build_* stages. runtime-ci
    # shares every runtime layer, so publishing both costs one manifest.
    runtime_ref=$(ensure_image runtime runtime "${contexts[@]}")
    runtime_ci_ref=$(ensure_image runtime-ci runtime-ci "${contexts[@]}")
    echo "ROCm runtime:    ${runtime_ref}"
    echo "ROCm CI runtime: ${runtime_ci_ref}"
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent meta-data set rocm-runtime-image "${runtime_ref}"
        buildkite-agent meta-data set rocm-runtime-ci-image "${runtime_ci_ref}"
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
