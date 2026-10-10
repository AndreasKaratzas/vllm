#!/usr/bin/env bash
# Build the per-commit ROCm CI image: the runtime-ci image selected by
# build-runtime.sh plus one vLLM layer, pushed as IMAGE_TAG. GPU jobs pull it
# directly. The wheel is compiled in the deployment runtime, so CI tests the
# same wheel a release built from that runtime ships, and test-only
# dependency changes do not invalidate the native compile cache.

set -euo pipefail

metadata_get() {
    local key="$1"
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent meta-data get "${key}" 2>/dev/null || true
    fi
}

load_digest_handoff() {
    local metadata_key="$1"
    local env_name="$2"
    local description="$3"
    local image_ref=""

    image_ref="$(metadata_get "${metadata_key}")"
    if [[ -z "${image_ref}" ]]; then
        return 1
    fi
    if [[ ! "${image_ref}" =~ @sha256:[0-9a-f]{64}$ ]]; then
        echo "${description} is not digest-pinned: ${image_ref}" >&2
        return 1
    fi

    printf -v "${env_name}" '%s' "${image_ref}"
    export "${env_name?}"
    echo "Using ${description}: ${image_ref}"
}

main() {
    # This job always builds the checked-out commit. Some externally generated
    # pipeline templates still inject remote-fetch settings; do not let those
    # settings make the source identity commit-specific or bypass local edits.
    export REMOTE_VLLM=0
    unset VLLM_BRANCH

    local handoff
    for handoff in "rocm-runtime-image BASE_IMAGE" "rocm-runtime-ci-image CI_BASE_IMAGE"; do
        # shellcheck disable=SC2086
        if ! load_digest_handoff ${handoff} "ROCm ${handoff%% *} handoff"; then
            if [[ "${BUILDKITE:-false}" == "true" ]]; then
                echo "Required ${handoff%% *} metadata is missing or invalid" >&2
                return 1
            fi
            echo "No ${handoff%% *} metadata found; using the local default"
        fi
    done

    bash .buildkite/scripts/ci-bake-rocm.sh test-rocm-ci-thin
}

main "$@"
