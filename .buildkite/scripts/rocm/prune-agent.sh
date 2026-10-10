#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Keep Docker and BuildKit disk use on ROCm CI agents bounded. Run before each
# build or test step. Safe on shared hosts: it only removes stopped containers,
# dangling images, CI-repo images no running container uses, and the cache of
# the CI builders, and it only evicts images when free disk is low.
#
# Image "created" times are pinned to SOURCE_DATE_EPOCH, so age filters such
# as `docker image prune --filter until=` would match every CI image. Eviction
# is therefore driven by free space, not by age.
#
# Usage: prune-agent.sh [--dry-run]

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi
MIN_FREE_PCT="${ROCM_AGENT_MIN_FREE_PCT:-25}"
CI_IMAGE_REPOS_RE="${ROCM_AGENT_CI_IMAGE_REPOS_RE:-^(docker\.io/)?rocm/vllm-(ci|dev|ci-artifact|ci-cache)(:|@)}"
read -r -a BUILDERS <<< "${ROCM_AGENT_BUILDERS:-vllm-builder vllm-builder-cache vllm-rocm-base-builder}"

run() {
    echo "+ $*"
    if ((DRY_RUN == 0)); then
        "$@" || echo "warning: '$*' failed" >&2
    fi
}

docker_root() {
    docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker
}

free_pct() {
    local used
    used=$(df --output=pcent "$(docker_root)" | tail -1 | tr -dc '0-9')
    echo $((100 - used))
}

disk_low() {
    (($(free_pct) < MIN_FREE_PCT))
}

echo "--- :broom: Agent disk: $(free_pct)% free (threshold ${MIN_FREE_PCT}%)"
docker system df || true

run docker container prune --force --filter until=24h
run docker image prune --force

for builder in "${BUILDERS[@]}"; do
    docker buildx inspect "${builder}" >/dev/null 2>&1 || continue
    if disk_low; then
        # The builder's GC policy (buildkitd.toml) bounds the cache continuously;
        # this only reacts when the disk is low anyway.
        run docker buildx prune --builder "${builder}" --force \
            --filter "type==exec.cachemount" --filter "until=72h"
        run docker buildx prune --builder "${builder}" --force --filter "until=168h"
    fi
done

if disk_low; then
    in_use=$(docker ps --format '{{.Image}}' | sort -u)
    while read -r ref; do
        [[ -n "${ref}" ]] || continue
        if grep -qxF -- "${ref}" <<< "${in_use}"; then
            continue
        fi
        run docker image rm "${ref}"
    done < <(docker image ls --format '{{.Repository}}:{{.Tag}}' \
        | grep -E "${CI_IMAGE_REPOS_RE}" | grep -v ':<none>$' || true)
fi

echo "Agent disk after pruning: $(free_pct)% free"
