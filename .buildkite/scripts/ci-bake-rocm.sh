#!/bin/bash
# ci-bake-rocm.sh - Docker buildx bake wrapper for ROCm CI builds.
#
# Builds the per-commit vLLM layer on top of the ROCm runtime image selected by
# rocm/build-runtime.sh. The csrc and Rust stages use content-keyed registry
# caches; the test image tag stays build-scoped.
#
# Usage:
#   ci-bake-rocm.sh [TARGET]
#
# Set BAKE_PRINT_ONLY=1 to stop after docker buildx bake --print.

set -euo pipefail

DEFAULT_REPO_SLUG="vllm-project/vllm"
DEFAULT_CI_HCL_SOURCE="docker/ci-rocm.hcl"
# ROCm CI forces REMOTE_VLLM=0, so content identity covers only the selected
# local-source stages rather than unreachable remote-fetch alternatives.
DEFAULT_ROCM_CSRC_CONTENT_FILES=".dockerignore requirements/common.txt requirements/rocm.txt pyproject.toml setup.py CMakeLists.txt cmake csrc vllm/envs.py vllm/__init__.py tools/build_rust.py"
DEFAULT_ROCM_CSRC_DOCKERFILE_STAGES="base fetch_vllm_0 fetch_vllm build_vllm_dependencies rocm-triton-kernels csrc-build"
DEFAULT_ROCM_RUST_CONTENT_FILES=".dockerignore .git_archival.txt pyproject.toml requirements/build/rust.txt rust/Cargo.lock rust/Cargo.toml rust/proto rust/src rust-toolchain.toml tools/build_rust.py tools/build_rust.sh"
DEFAULT_ROCM_RUST_DOCKERFILE_STAGES="base fetch_vllm_0 fetch_vllm vllm-version rust_toolchain_input_0 rust-toolchain-input rust_input_0 rust-input rust-toolchain rust-build"
# Docker's 128-character tag limit minus the longest cache prefix
# ("csrc-rocm-branch-" and "rust-rocm-branch-", both 17 characters).
ROCM_CACHE_BRANCH_TAG_MAX_LEN=111
CACHE_WRITE_SCOPE=""

TARGET=""
CI_HCL_SOURCE="${CI_HCL_SOURCE:-}"
CI_HCL_PATH=""
CSRC_CACHE_OVERRIDE_PATH=""
ROCM_ARG_OVERRIDE_PATH=""
BUILD_CONTEXT_OVERRIDE_PATH=""
SCRIPT_TMP_DIR=""
BAKE_CONFIG_FILE=""
ROCM_BUILD_CONTEXT_ROOT=""
ROCM_BUILD_CONTEXT_INDEX=""
ROCM_BUILD_CONTEXT_COMMIT=""
BAKE_FILES=()
BAKE_ALLOW_ARGS=()
BAKE_TARGETS=()

cleanup() {
    if [[ -n "${SCRIPT_TMP_DIR}" && -d "${SCRIPT_TMP_DIR}" ]]; then
        rm -rf "${SCRIPT_TMP_DIR}"
    fi
}
trap cleanup EXIT

clean_docker_tag() {
    local input="$1"
    echo "${input}" | sed 's/[^a-zA-Z0-9._-]/_/g' | cut -c1-128
}

is_url_like() {
    local value="${1:-}"
    [[ "${value}" =~ ^[a-zA-Z][a-zA-Z0-9+.-]*:// || "${value}" == git@*:* ]]
}

is_full_git_sha() {
    local value="${1:-}"
    [[ "${value}" =~ ^[0-9a-fA-F]{40}$ ]]
}

select_cache_branch_name() {
    local candidate=""
    local var=""

    for var in \
        ROCM_CACHE_BRANCH_NAME \
        BUILDKITE_PULL_REQUEST_HEAD_BRANCH \
        BUILDKITE_HEAD_BRANCH \
        BUILDKITE_BRANCH \
        VLLM_BRANCH; do
        candidate="${!var:-}"
        [[ -n "${candidate}" ]] || continue
        is_url_like "${candidate}" && continue
        is_full_git_sha "${candidate}" && continue
        printf '%s\n' "${candidate}"
        return 0
    done
}

cache_scope_suffix() {
    local arch_hash=""
    arch_hash=$(printf '%s' "${PYTORCH_ROCM_ARCH:-default}" | sha256sum | cut -c1-12)
    printf 'arch-%s\n' "${arch_hash}"
}

compose_cache_branch_tag() {
    local repo_slug="$1"
    local branch="$2"
    local suffix=""
    local prefix=""
    local max_prefix_len=0
    local max_tag_len="${ROCM_CACHE_BRANCH_TAG_MAX_LEN}"

    suffix="$(cache_scope_suffix)"
    prefix="$(clean_docker_tag "${repo_slug}")-$(clean_docker_tag "${branch}")"
    max_prefix_len=$((max_tag_len - ${#suffix} - 1))
    if (( max_prefix_len < 1 )); then
        max_prefix_len=1
    fi
    printf '%s-%s\n' "${prefix:0:${max_prefix_len}}" "${suffix}"
}

parse_repo_slug() {
    local repo_url="${1:-}"
    local repo_slug=""

    if [[ -z "${repo_url}" ]]; then
        printf '%s\n' "${DEFAULT_REPO_SLUG}"
        return 0
    fi

    repo_slug=$(echo "${repo_url}" | sed -E 's#(git@|https?://)([^/:]+)[:/]([^/]+/[^/.]+)(\.git)?$#\3#')
    if [[ "${repo_slug}" != */* ]]; then
        repo_slug="${DEFAULT_REPO_SLUG}"
    fi
    printf '%s\n' "${repo_slug}"
}

normalize_repo_slug() {
    local repo_slug="${1:-}"

    repo_slug="${repo_slug%/}"
    repo_slug="${repo_slug%.git}"
    repo_slug="${repo_slug#https://github.com/}"
    repo_slug="${repo_slug#http://github.com/}"
    repo_slug="${repo_slug#ssh://git@github.com/}"
    repo_slug="${repo_slug#git@github.com:}"
    repo_slug="${repo_slug#github.com/}"
    printf '%s\n' "${repo_slug}"
}

is_trusted_ci_cache_writer() {
    local actual_repo=""

    [[ "${BUILDKITE:-false}" == "true" ]] || return 1
    [[ "${BUILDKITE_PULL_REQUEST:-false}" == "false" ]] || return 1
    [[ "${BUILDKITE_BRANCH:-}" == "main" ]] || return 1
    actual_repo=$(normalize_repo_slug "${BUILDKITE_REPO:-}")
    [[ "${actual_repo}" == "${DEFAULT_REPO_SLUG}" ]]
}

# Main writes canonical content cache refs; other builds read them and write
# into a namespace scoped to their source repository.
configure_cache_write_scope() {
    local identity=""

    if is_trusted_ci_cache_writer; then
        CACHE_WRITE_SCOPE=""
        echo "Trusted main build: publishing canonical content cache refs"
        return 0
    fi
    identity=$(printf '%s\n' "${BUILDKITE_PULL_REQUEST_REPO:-${BUILDKITE_REPO:-local}}" \
        | sha256sum | cut -c1-12)
    CACHE_WRITE_SCOPE="preview-${identity}"
    echo "Non-canonical cache writes use source scope: ${CACHE_WRITE_SCOPE}"
}

get_buildkite_repo_slug() {
    parse_repo_slug "${BUILDKITE_PULL_REQUEST_REPO:-${BUILDKITE_REPO:-}}"
}

get_buildkite_target_repo_slug() {
    parse_repo_slug "${BUILDKITE_REPO:-}"
}

get_buildkite_target_repo_url() {
    local repo_url="${BUILDKITE_REPO:-}"

    if [[ -n "${repo_url}" ]] && is_url_like "${repo_url}"; then
        printf '%s\n' "${repo_url}"
        return 0
    fi

    printf 'https://github.com/%s.git\n' "${DEFAULT_REPO_SLUG}"
}

git_fetch_with_timeout() {
    local timeout_secs="${ROCM_CACHE_GIT_FETCH_TIMEOUT:-60}"
    local -a fetch_command=(git fetch --no-auto-maintenance)

    # Detached maintenance can race a later shallow fetch on .git/shallow.
    if command -v timeout >/dev/null 2>&1; then
        timeout "${timeout_secs}s" "${fetch_command[@]}" "$@"
    else
        "${fetch_command[@]}" "$@"
    fi
}

git_fetch_for_cache() {
    git_fetch_with_timeout "$@" 2>/dev/null
}

list_content_files() {
    # Hash the checkout inputs Docker can intentionally consume, not ignored
    # compiler/test debris left behind on a reused worker.
    if [[ -n "${ROCM_BUILD_CONTEXT_ROOT:-}" ]]; then
        GIT_INDEX_FILE="${ROCM_BUILD_CONTEXT_INDEX}" \
            git ls-files -z --cached -- "$1" | LC_ALL=C sort -z
    else
        git ls-files -z --cached --others --exclude-standard -- "$1" \
            | LC_ALL=C sort -z
    fi
}

content_regular_file() {
    local file="$1"
    local physical_file=""

    physical_file="${file}"
    if [[ -n "${ROCM_BUILD_CONTEXT_ROOT:-}" && "${file}" != /* ]]; then
        physical_file="${ROCM_BUILD_CONTEXT_ROOT}/${file}"
    fi
    [[ -f "${physical_file}" ]]
}

hash_content_file() {
    local file="$1"
    local physical_file=""
    local checksum=""
    local file_mode=""

    physical_file="${file}"
    if [[ -n "${ROCM_BUILD_CONTEXT_ROOT:-}" && "${file}" != /* ]]; then
        physical_file="${ROCM_BUILD_CONTEXT_ROOT}/${file}"
    fi
    if [[ -L "${physical_file}" ]]; then
        printf 'symlink:%s\ntarget:' "${file}"
        readlink -n -- "${physical_file}" || return $?
        printf '\n'
        return
    fi
    if [[ ! -f "${physical_file}" ]]; then
        printf 'missing:%s\n' "${file}"
        return
    fi
    file_mode=$(stat -c '%a' "${physical_file}") || return $?
    printf 'file:%s\nmode:%s\n' "${file}" "${file_mode}"
    checksum=$(sha256sum < "${physical_file}") || return $?
    checksum="${checksum%% *}"
    printf '%s  %s\n' "${checksum}" "${file}"
}

hash_content_directory() {
    if ! list_content_files "$1" | while IFS= read -r -d '' file; do
        hash_content_file "${file}" || exit $?
    done; then
        echo "Failed to hash content under $1" >&2
        return 1
    fi
}

compute_content_hash() {
    local path=""
    local physical_path=""

    for path in "$@"; do
        physical_path="${path}"
        if [[ -n "${ROCM_BUILD_CONTEXT_ROOT:-}" && "${path}" != /* ]]; then
            physical_path="${ROCM_BUILD_CONTEXT_ROOT}/${path}"
        fi
        if [[ -L "${physical_path}" || -f "${physical_path}" ]]; then
            hash_content_file "${path}" || return $?
        elif [[ -d "${physical_path}" ]]; then
            hash_content_directory "${path}" || return $?
        else
            printf 'missing:%s\n' "${path}"
        fi
    done | sha256sum | cut -d' ' -f1
}

validate_ci_build_context_source() {
    local source_root="$1"
    local context_commit=""
    local source_head=""
    local entry=""
    local metadata=""
    local mode=""
    local stage=""
    local path=""
    local worktree_diff_status=0
    local staged_diff_status=0
    local index_modes_file="${SCRIPT_TMP_DIR}/git-index-modes"
    local info_attributes=""

    if ! context_commit=$(git -C "${source_root}" rev-parse \
        --verify "${BUILDKITE_COMMIT:-HEAD}^{commit}"); then
        echo "Failed to resolve the CI Docker context revision" >&2
        return 1
    fi
    if ! source_head=$(git -C "${source_root}" rev-parse --verify HEAD); then
        echo "Failed to resolve the checked-out CI revision" >&2
        return 1
    fi
    if [[ "${context_commit}" != "${source_head}" ]]; then
        echo "BUILDKITE_COMMIT does not match the checked-out CI revision" >&2
        return 1
    fi
    ROCM_BUILD_CONTEXT_COMMIT="${context_commit}"
    # Shared AMD workspaces may present tracked files with inflated executable
    # bits. Context modes come from the pinned Git tree, so ignore only that
    # filesystem drift while continuing to reject content changes.
    git -C "${source_root}" -c core.fileMode=false diff \
        --quiet --no-ext-diff --ignore-submodules=none -- \
        || worktree_diff_status=$?
    if (( worktree_diff_status != 0 )); then
        if (( worktree_diff_status > 1 )); then
            printf 'Failed to inspect tracked CI worktree changes (git diff exited %s)\n' \
                "${worktree_diff_status}" >&2
            return 1
        fi
        echo "Tracked worktree changes cannot be omitted from the CI Docker context" \
            >&2
        echo "Tracked worktree diff (first 50 entries):" >&2
        git -C "${source_root}" -c core.fileMode=false diff \
            --no-ext-diff --ignore-submodules=none --name-status -- \
            | sed -n '1,50p' >&2 || true
        git -C "${source_root}" -c core.fileMode=false diff \
            --no-ext-diff --ignore-submodules=none --summary -- \
            | sed -n '1,50p' >&2 || true
        return 1
    fi
    git -C "${source_root}" diff --cached \
        --quiet --no-ext-diff --ignore-submodules=none HEAD -- \
        || staged_diff_status=$?
    if (( staged_diff_status != 0 )); then
        if (( staged_diff_status > 1 )); then
            printf 'Failed to inspect staged CI changes (git diff exited %s)\n' \
                "${staged_diff_status}" >&2
            return 1
        fi
        echo "Staged changes cannot be omitted from the CI Docker context" >&2
        echo "Staged diff (first 50 entries):" >&2
        git -C "${source_root}" diff --cached \
            --no-ext-diff --ignore-submodules=none --name-status HEAD -- \
            | sed -n '1,50p' >&2 || true
        git -C "${source_root}" diff --cached \
            --no-ext-diff --ignore-submodules=none --summary HEAD -- \
            | sed -n '1,50p' >&2 || true
        return 1
    fi
    # Untracked and ignored worker outputs are intentionally absent: the
    # pinned Git tree, rather than mutable checkout contents, is the contract.
    if ! info_attributes=$(git -C "${source_root}" \
        rev-parse --path-format=absolute --git-path info/attributes); then
        echo "Failed to locate repository-local Git attributes" >&2
        return 1
    fi
    if [[ -s "${info_attributes}" ]]; then
        echo "Repository-local Git attributes cannot define the CI Docker context" \
            >&2
        return 1
    fi

    ROCM_BUILD_CONTEXT_INDEX="${SCRIPT_TMP_DIR}/docker-context.index"
    if ! GIT_INDEX_FILE="${ROCM_BUILD_CONTEXT_INDEX}" \
        git -C "${source_root}" -c core.splitIndex=false \
            read-tree "${context_commit}^{tree}"; then
        echo "Failed to create the CI Docker context index" >&2
        return 1
    fi
    if ! GIT_INDEX_FILE="${ROCM_BUILD_CONTEXT_INDEX}" \
        git -C "${source_root}" ls-files --stage -z \
            > "${index_modes_file}"; then
        echo "Failed to read the Git index for the CI Docker context" >&2
        return 1
    fi
    while IFS= read -r -d '' entry; do
        metadata="${entry%%$'\t'*}"
        mode="${metadata%% *}"
        stage="${metadata##* }"
        path="${entry#*$'\t'}"
        if [[ "${stage}" != "0" ]]; then
            echo "Unmerged Git entry cannot be used as Docker context: ${path}" >&2
            return 1
        fi
        case "${mode}" in
            100644|100755|120000)
                ;;
            160000)
                echo "Git submodule cannot be materialized in the CI Docker context: ${path}" \
                    >&2
                return 1
                ;;
            *)
                echo "Unsupported Git mode ${mode} for ${path}" >&2
                return 1
                ;;
        esac
    done < "${index_modes_file}"
}

describe_ci_revision() {
    git -C "$1" describe --tags --long --abbrev=10 \
        --match 'v[0-9]*' "$2" 2>/dev/null
}

write_ci_git_archival_metadata() {
    local source_root="$1"
    local context_root="$2"
    local commit="${ROCM_BUILD_CONTEXT_COMMIT}"
    local commit_date=""
    local describe=""
    local is_shallow="false"

    commit_date=$(git -C "${source_root}" show -s --format=%cI "${commit}") \
        || return $?
    is_shallow=$(git -C "${source_root}" rev-parse --is-shallow-repository) \
        || return $?
    if git -C "${source_root}" remote get-url origin >/dev/null 2>&1; then
        # AMD agents use shallow, no-tag clones. Reach a release tag so one
        # commit cannot acquire different versions from different depths.
        # Versioning only needs tags and commits, not historical source trees.
        echo "Synchronizing version tags for the canonical CI Docker context"
        (cd "${source_root}" && git_fetch_with_timeout --quiet \
            --filter=tree:0 --prune origin '+refs/tags/*:refs/tags/*') \
            || return $?
        if [[ "${is_shallow}" == "true" ]] \
            && ! describe_ci_revision "${source_root}" "${commit}" >/dev/null; then
            echo "Deepening history to reach a version tag"
            (cd "${source_root}" && git_fetch_with_timeout --quiet \
                --filter=tree:0 --no-tags --deepen=1000 origin "${commit}") \
                || return $?
            is_shallow=$(git -C "${source_root}" \
                rev-parse --is-shallow-repository) || return $?
            if [[ "${is_shallow}" == "true" ]] \
                && ! describe_ci_revision "${source_root}" "${commit}" >/dev/null; then
                echo "No version tag within 1,000 commits; fetching full history"
                (cd "${source_root}" && git_fetch_with_timeout --quiet \
                    --filter=tree:0 --no-tags --unshallow origin "${commit}") \
                    || return $?
            fi
        fi
    fi
    if ! describe=$(describe_ci_revision "${source_root}" "${commit}"); then
        echo "No numeric version tag is reachable from the CI revision" >&2
        return 1
    fi
    if [[ -e "${context_root}/.git_archival.txt" \
        || -L "${context_root}/.git_archival.txt" ]]; then
        echo "The source tree already contains .git_archival.txt" >&2
        return 1
    fi
    printf 'node: %s\nnode-date: %s\ndescribe-name: %s\n' \
        "${commit}" "${commit_date}" "${describe}" \
        > "${context_root}/.git_archival.txt" \
        && chmod 0644 -- "${context_root}/.git_archival.txt"
}

write_build_context_override() {
    local escaped_context=""

    [[ -n "${ROCM_BUILD_CONTEXT_ROOT:-}" ]] || return 0
    escaped_context=$(hcl_escape_string "${ROCM_BUILD_CONTEXT_ROOT}") || return $?
    if ! {
        printf 'target "_common-rocm" {\n'
        printf '  context = "%s"\n' "${escaped_context}"
        printf '}\n'
    } > "${BUILD_CONTEXT_OVERRIDE_PATH}"; then
        echo "Failed to write the CI Docker context override" >&2
        return 1
    fi
    BAKE_FILES+=(-f "${BUILD_CONTEXT_OVERRIDE_PATH}")
}

prepare_ci_build_context() {
    local source_root=""
    local context_root=""

    [[ "${BUILDKITE:-false}" == "true" ]] || return 0
    [[ "${REMOTE_VLLM:-0}" == "0" ]] || return 0

    # BuildKit includes file modes in cache keys. Export the pinned revision to
    # an owned context instead of changing modes in the shared checkout.
    if ! source_root=$(git rev-parse --show-toplevel); then
        echo "Failed to locate the CI source checkout" >&2
        return 1
    fi
    validate_ci_build_context_source "${source_root}" || return $?
    context_root="${SCRIPT_TMP_DIR}/docker-context"
    if ! mkdir -m 0700 -- "${context_root}"; then
        echo "Failed to create the owned CI Docker context" >&2
        return 1
    fi
    if ! (
        umask 0022
        unset GIT_LFS_SKIP_SMUDGE
        export GIT_ATTR_NOSYSTEM=1
        export GIT_INDEX_FILE="${ROCM_BUILD_CONTEXT_INDEX}"
        git -C "${source_root}" \
            -c core.attributesFile=/dev/null \
            -c core.autocrlf=false \
            -c core.eol=lf \
            -c core.symlinks=true \
            --work-tree="${context_root}" \
            checkout-index --all
    ); then
        echo "Failed to materialize the CI Docker context from the Git index" >&2
        return 1
    fi

    # setuptools-scm understands Git's stable archive format, so wheels and
    # Rust artifacts retain their exact version without copying Git history.
    write_ci_git_archival_metadata "${source_root}" "${context_root}" \
        || return $?
    if [[ -e "${context_root}/.git" ]]; then
        echo "Canonical CI Docker context unexpectedly contains .git" >&2
        return 1
    fi
    ROCM_BUILD_CONTEXT_ROOT="${context_root}"
    BAKE_ALLOW_ARGS+=(--allow "fs.read=${ROCM_BUILD_CONTEXT_ROOT}")
    echo "Using canonical CI Docker context: ${ROCM_BUILD_CONTEXT_ROOT}"
}

hash_dockerfile_stages() {
    local dockerfile="$1"
    local stages="$2"
    local physical_dockerfile=""

    physical_dockerfile="${dockerfile}"
    if [[ -n "${ROCM_BUILD_CONTEXT_ROOT:-}" && "${dockerfile}" != /* ]]; then
        physical_dockerfile="${ROCM_BUILD_CONTEXT_ROOT}/${dockerfile}"
    fi

    awk -v wanted_stages="${stages}" '
        BEGIN {
            split(wanted_stages, stage_list, /[[:space:]]+/)
            for (idx in stage_list) {
                if (stage_list[idx] != "") {
                    wanted[stage_list[idx]] = 1
                }
            }
            emit = 0
        }
        toupper($1) == "FROM" {
            stage = ""
            for (idx = 1; idx <= NF; idx++) {
                if (tolower($idx) == "as" && idx < NF) {
                    stage = tolower($(idx + 1))
                }
            }
            emit = (stage in wanted)
        }
        emit {
            print
        }
    ' "${physical_dockerfile}"
}

discover_dockerfile_stage_args() {
    local dockerfile="$1"
    local stages="$2"
    local physical_dockerfile=""

    physical_dockerfile="${dockerfile}"
    if [[ -n "${ROCM_BUILD_CONTEXT_ROOT:-}" && "${dockerfile}" != /* ]]; then
        physical_dockerfile="${ROCM_BUILD_CONTEXT_ROOT}/${dockerfile}"
    fi
    [[ -f "${physical_dockerfile}" ]] || return 0

    awk -v wanted_stages="${stages}" '
        function add_arg(name) {
            if (name != "" && !(name in seen)) {
                seen[name] = 1
                args[++arg_count] = name
            }
        }
        BEGIN {
            split(wanted_stages, stage_list, /[[:space:]]+/)
            for (idx in stage_list) {
                if (stage_list[idx] != "") {
                    wanted[stage_list[idx]] = 1
                }
            }
            emit = 1
        }
        {
            line = $0
            if (toupper($1) == "FROM") {
                stage = ""
                for (idx = 1; idx <= NF; idx++) {
                    if (tolower($idx) == "as" && idx < NF) {
                        stage = tolower($(idx + 1))
                    }
                }
                emit = (stage in wanted)
            }
            if (emit) {
                lines[++line_count] = line
            }
        }
        END {
            for (idx = 1; idx <= line_count; idx++) {
                line = lines[idx]
                arg_name = line
                sub(/^[[:space:]]*[Aa][Rr][Gg][[:space:]]+/, "", arg_name)
                if (arg_name != line) {
                    sub(/[=[:space:]].*/, "", arg_name)
                    if (arg_name ~ /^[A-Za-z_][A-Za-z0-9_]*$/) {
                        add_arg(arg_name)
                    }
                }
            }

            for (idx = 1; idx <= line_count; idx++) {
                line = lines[idx]
                for (arg_idx = 1; arg_idx <= arg_count; arg_idx++) {
                    name = args[arg_idx]
                    if (line ~ "\\$\\{" name "([}:][^}]*)?\\}" \
                        || line ~ "\\$" name "([^A-Za-z0-9_]|$)") {
                        used[name] = 1
                    }
                }
            }

            for (arg_idx = 1; arg_idx <= arg_count; arg_idx++) {
                name = args[arg_idx]
                if (used[name]) {
                    print name
                }
            }
        }
    ' "${physical_dockerfile}"
}

get_content_arg_names() {
    local dockerfile="$1"
    local stages="$2"
    local explicit_args="${3:-}"

    if [[ -n "${explicit_args}" ]]; then
        tr ' ' '\n' <<< "${explicit_args}"
    else
        discover_dockerfile_stage_args "${dockerfile}" "${stages}"
    fi | awk 'NF && !seen[$0]++'
}

extract_dockerfile_arg_default() {
    local dockerfile="$1"
    local arg_name="$2"
    local physical_dockerfile=""

    physical_dockerfile="${dockerfile}"
    if [[ -n "${ROCM_BUILD_CONTEXT_ROOT:-}" && "${dockerfile}" != /* ]]; then
        physical_dockerfile="${ROCM_BUILD_CONTEXT_ROOT}/${dockerfile}"
    fi
    sed -n -E "s/^[[:space:]]*[Aa][Rr][Gg][[:space:]]+${arg_name}=\"?([^\"[:space:]]+)\"?.*/\\1/p" \
        "${physical_dockerfile}" | head -1
}

resolve_image_digest() {
    local image_ref="$1"
    local attempts="${ROCM_IMAGE_DIGEST_ATTEMPTS:-4}"
    local delay_secs="${ROCM_IMAGE_DIGEST_RETRY_DELAY:-2}"
    local attempt=0
    local digest=""
    local output=""
    local status=0

    if [[ "${image_ref}" =~ @(sha256:[0-9a-f]{64})$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return
    fi
    if [[ ! "${attempts}" =~ ^[1-9][0-9]*$ \
        || ! "${delay_secs}" =~ ^[0-9]+$ ]]; then
        echo "Invalid image digest retry configuration" >&2
        return 1
    fi

    for ((attempt = 1; attempt <= attempts; attempt++)); do
        status=0
        output=$(docker buildx imagetools inspect "${image_ref}" 2>&1) || status=$?
        digest=$(awk '$1 == "Digest:" { print $2; exit }' <<< "${output}")
        if ((status == 0)) && [[ "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
            printf '%s\n' "${digest}"
            return
        fi
        if ((attempt < attempts)); then
            printf \
                'Image digest lookup failed for %s (%d/%d, status %d); retrying\n' \
                "${image_ref}" "${attempt}" "${attempts}" "${status}" >&2
            sleep "${delay_secs}"
        fi
    done

    printf 'Failed to resolve digest for %s (status %d)\n%s\n' \
        "${image_ref}" "${status}" "${output:-<no output>}" >&2
    return 1
}

resolve_dockerfile_arg_value() {
    local dockerfile="$1"
    local arg_name="$2"
    local env_name="${arg_name}"
    local value=""

    case "${arg_name}" in
        ARG_PYTORCH_ROCM_ARCH)
            env_name="PYTORCH_ROCM_ARCH"
            ;;
        max_jobs)
            env_name="CI_MAX_JOBS"
            ;;
    esac

    value="${!env_name:-}"
    if [[ -z "${value}" && "${env_name}" != "${arg_name}" ]]; then
        value="${!arg_name:-}"
    fi
    if [[ -z "${value}" ]] && content_regular_file "${dockerfile}"; then
        value=$(extract_dockerfile_arg_default "${dockerfile}" "${arg_name}")
    fi

    printf '%s\n' "${value}"
}

hash_dockerfile_arg_values() {
    local dockerfile="$1"
    local arg_name=""
    local arg_value=""
    local digest=""
    shift || true

    for arg_name in "$@"; do
        [[ -n "${arg_name}" ]] || continue
        arg_value=$(resolve_dockerfile_arg_value "${dockerfile}" "${arg_name}")
        if [[ "${arg_name}" == "BASE_IMAGE" && -n "${arg_value}" ]]; then
            if ! digest=$(resolve_image_digest "${arg_value}"); then
                echo "Failed to resolve digest for BASE_IMAGE=${arg_value}" >&2
                return 1
            fi
            printf 'arg:%s.digest=%s\n' "${arg_name}" "${digest}"
        else
            printf 'arg:%s=%s\n' "${arg_name}" "${arg_value:-<empty>}"
        fi
    done
}

pin_base_image() {
    local dockerfile="${CI_BASE_DOCKERFILE}"
    local base_image=""
    local digest=""

    base_image=$(resolve_dockerfile_arg_value "${dockerfile}" "BASE_IMAGE")
    [[ -n "${base_image}" ]] || return 0
    if ! digest=$(resolve_image_digest "${base_image}"); then
        echo "Error: could not resolve base image digest for ${base_image}" >&2
        echo "Refusing to compute content-addressed cache keys from a mutable tag." >&2
        return 1
    fi
    BASE_IMAGE="${base_image%@*}@${digest}"
    export BASE_IMAGE
    echo "Pinned base image for this build: ${BASE_IMAGE}"
}

is_commit_image_target() {
    [[ -n "${IMAGE_TAG:-}" && -n "${BUILDKITE_COMMIT:-}" ]]
}

# Targets whose local outputs (wheel, smoke marker) later steps consume, so an
# already-pushed image must not short-circuit them.
has_local_outputs() {
    should_export_rocm_smoke || [[ "${TARGET}" == *"export-wheel"* ]]
}

should_export_rocm_smoke() {
    [[ "${TARGET}" == "test-rocm-ci-thin" || "${TARGET}" == "smoke-test-rocm-ci" ]]
}

verify_rocm_smoke_export() {
    local marker="./build/rocm-smoke-export/vllm-smoke-ok"
    local expected_smoke_id="${BUILDKITE_BUILD_ID:-local}"
    local actual_smoke_id=""

    should_export_rocm_smoke || return 0
    if [[ ! -f "${marker}" ]]; then
        echo "ROCm BuildKit smoke marker is missing: ${marker}" >&2
        return 1
    fi
    actual_smoke_id="$(< "${marker}")"
    if [[ "${actual_smoke_id}" != "${expected_smoke_id}" ]]; then
        echo "ROCm BuildKit smoke marker belongs to ${actual_smoke_id}, not ${expected_smoke_id}" \
            >&2
        return 1
    fi
}

registry_ref_exists_with_retry() {
    local image_ref="$1"
    local attempts="${ROCM_REGISTRY_PROBE_ATTEMPTS:-2}"
    local delay_secs="${ROCM_REGISTRY_PROBE_RETRY_DELAY:-1}"
    local output=""
    local status=0
    local attempt=0

    if [[ ! "${attempts}" =~ ^[1-9][0-9]*$ \
        || ! "${delay_secs}" =~ ^[0-9]+$ ]]; then
        echo "Invalid registry probe retry configuration" >&2
        return 1
    fi

    for ((attempt = 1; attempt <= attempts; attempt++)); do
        status=0
        output=$(docker buildx imagetools inspect "${image_ref}" 2>&1) \
            || status=$?
        if ((status == 0)); then
            return 0
        fi
        # A definitive registry miss is normal for a new content key. Retry
        # transport/rate-limit failures, but do not add latency to a real 404.
        if grep -Eiq \
            'manifest unknown|no such manifest|name unknown|not found|does not exist' \
            <<< "${output}"; then
            return 1
        fi
        if ((attempt < attempts)); then
            echo "Registry probe failed for ${image_ref} (${attempt}/${attempts}); retrying" >&2
            sleep "${delay_secs}"
        fi
    done

    echo "Registry probe failed after ${attempts} attempts: ${image_ref}" >&2
    return 1
}

use_existing_builder() {
    echo "Using existing builder: ${BUILDER_NAME}"
    docker buildx use "${BUILDER_NAME}"
    docker buildx inspect --bootstrap
}

buildx_driver() {
    local builder="${1:-}"

    if [[ -n "${builder}" ]]; then
        docker buildx inspect "${builder}" 2>/dev/null
    else
        docker buildx inspect 2>/dev/null
    fi | awk -F': *' '$1 == "Driver" { print $2; exit }'
}

builder_supports_registry_cache() {
    local driver="$1"

    [[ -n "${driver}" && "${driver}" != "docker" ]]
}

create_and_bootstrap_builder() {
    local driver="$1"
    local endpoint="${2:-}"

    echo "Creating builder '${BUILDER_NAME}' with ${driver} driver"
    if [[ -n "${endpoint}" ]]; then
        docker buildx create \
            --name "${BUILDER_NAME}" \
            --driver "${driver}" \
            --use \
            "${endpoint}"
    else
        # Pinned BuildKit (reproducible exports) with a disk-bounded GC policy.
        docker buildx create --name "${BUILDER_NAME}" --driver "${driver}" --use \
            --driver-opt "image=${ROCM_BUILDKIT_IMAGE:-moby/buildkit:v0.33.1}" \
            --buildkitd-config "$(dirname "${BASH_SOURCE[0]}")/rocm/buildkitd.toml"
    fi
    docker buildx inspect --bootstrap
}

init_config() {
    ROCM_BASE_DOCKERFILE="${ROCM_BASE_DOCKERFILE:-docker/Dockerfile.rocm_base}"
    CI_BASE_DOCKERFILE="docker/Dockerfile.rocm"
    export ROCM_BASE_DOCKERFILE CI_BASE_DOCKERFILE

    TARGET="${1:-test-rocm-ci-thin}"
    BAKE_TARGETS=("${TARGET}")
    CI_HCL_SOURCE="${CI_HCL_SOURCE:-${CI_HCL_FILE:-${DEFAULT_CI_HCL_SOURCE}}}"
    VLLM_BAKE_FILE="${VLLM_BAKE_FILE:-docker/docker-bake-rocm.hcl}"
    BUILDER_NAME="${BUILDER_NAME:-vllm-builder}"
    BUILDKIT_SOCKET="${BUILDKIT_SOCKET:-/run/buildkit/buildkitd.sock}"
    PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx90a;gfx942;gfx950}"
    # One epoch per base lineage, owned by the base Dockerfile.
    SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(sed -nE \
        's/^ARG SOURCE_DATE_EPOCH=([0-9]+)$/\1/p' "${ROCM_BASE_DOCKERFILE}" | head -1)}"
    if [[ ! "${SOURCE_DATE_EPOCH}" =~ ^[0-9]+$ ]]; then
        echo "SOURCE_DATE_EPOCH must be set in the ROCm base Dockerfile" >&2
        return 1
    fi
    export SOURCE_DATE_EPOCH PYTORCH_ROCM_ARCH

    SCRIPT_TMP_DIR=$(mktemp -d -t ci-bake-rocm.XXXXXX)
    CI_HCL_PATH="${SCRIPT_TMP_DIR}/ci.hcl"
    CSRC_CACHE_OVERRIDE_PATH="${SCRIPT_TMP_DIR}/rocm-csrc-cache-override.hcl"
    ROCM_ARG_OVERRIDE_PATH="${SCRIPT_TMP_DIR}/rocm-arg-override.hcl"
    BUILD_CONTEXT_OVERRIDE_PATH="${SCRIPT_TMP_DIR}/build-context-override.hcl"
    BAKE_CONFIG_FILE="bake-config-build-${BUILDKITE_BUILD_NUMBER:-local}.json"
}

print_header() {
    echo "--- :docker: Setting up Docker buildx bake"
    echo "Target: ${TARGET}"
    echo "CI HCL source: ${CI_HCL_SOURCE}"
    echo "vLLM bake file: ${VLLM_BAKE_FILE}"
    if is_commit_image_target; then
        echo "Build mode: build-scoped commit image"
    else
        echo "Build mode: generic"
    fi
    if [[ "${USE_SCCACHE:-0}" == "1" ]]; then
        echo "Compiler cache: sccache enabled"
    fi
}

validate_inputs() {
    if [[ ! -f "${VLLM_BAKE_FILE}" ]]; then
        echo "Error: vLLM bake file not found at ${VLLM_BAKE_FILE}"
        echo "Make sure you're running from the vLLM repository root"
        exit 1
    fi

    if [[ -n "${CI_HCL_SOURCE:-}" ]] && is_url_like "${CI_HCL_SOURCE}"; then
        echo "Error: remote CI HCL sources are not supported: ${CI_HCL_SOURCE}"
        echo "Use the vLLM-owned docker/ci-rocm.hcl or set CI_HCL_SOURCE to a local file."
        exit 1
    fi

    if [[ -n "${CI_HCL_SOURCE:-}" && ! -f "${CI_HCL_SOURCE}" ]]; then
        echo "Error: CI HCL file not found at ${CI_HCL_SOURCE}"
        echo "Set CI_HCL_SOURCE to a local file if you need an override."
        exit 1
    fi
}

load_ci_hcl() {
    echo "--- :page_facing_up: Loading ci.hcl"
    cp "${CI_HCL_SOURCE}" "${CI_HCL_PATH}"
    echo "Copied ${CI_HCL_SOURCE} to ${CI_HCL_PATH}"
}

init_bake_files() {
    BAKE_FILES=(-f "${VLLM_BAKE_FILE}" -f "${CI_HCL_PATH}")
}

setup_builder() {
    echo "--- :buildkite: Setting up buildx builder"

    local setup_mode="${ROCM_SETUP_BUILDX_BUILDER:-auto}"
    local current_driver=""
    local named_driver=""

    if [[ "${setup_mode}" == "0" || "${setup_mode}" == "false" ]]; then
        echo "Using current Docker buildx builder"
        echo "ROCM_SETUP_BUILDX_BUILDER=${setup_mode}; cache exporters may fail if the driver is docker"
        docker buildx inspect --bootstrap
        echo "Active builder:"
        docker buildx ls | grep -E '^\*|^NAME' || docker buildx ls
        return 0
    fi

    current_driver=$(buildx_driver || true)
    if [[ "${setup_mode}" != "1" ]] && builder_supports_registry_cache "${current_driver}"; then
        echo "Using current Docker buildx builder with ${current_driver} driver"
        docker buildx inspect --bootstrap
        echo "Active builder:"
        docker buildx ls | grep -E '^\*|^NAME' || docker buildx ls
        return 0
    fi

    if [[ "${setup_mode}" != "1" ]]; then
        echo "Current buildx driver '${current_driver:-unknown}' cannot export registry caches"
        echo "Creating or using a cache-capable builder: ${BUILDER_NAME}"
    fi

    if docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
        named_driver=$(buildx_driver "${BUILDER_NAME}" || true)
        if ! builder_supports_registry_cache "${named_driver}"; then
            echo "Builder '${BUILDER_NAME}' uses ${named_driver:-unknown} driver; using ${BUILDER_NAME}-cache instead"
            BUILDER_NAME="${BUILDER_NAME}-cache"
        fi
    fi

    if [[ -S "${BUILDKIT_SOCKET}" ]]; then
        echo "Found local buildkitd socket at ${BUILDKIT_SOCKET}"
        echo "Using remote driver to connect to buildkitd"

        if docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
            use_existing_builder
        else
            create_and_bootstrap_builder remote "unix://${BUILDKIT_SOCKET}"
        fi
    elif docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
        use_existing_builder
    else
        echo "No local buildkitd found, using docker-container driver"
        create_and_bootstrap_builder docker-container
    fi

    echo "Active builder:"
    docker buildx ls | grep -E '^\*|^NAME' || docker buildx ls
}

# rewrite-timestamp needs BuildKit >= 0.14 to apply to pushed images. Older
# builders silently ignore it and every rebuild gets new layer digests.
require_reproducible_builder() {
    local version=""
    version=$(docker buildx inspect --bootstrap 2>/dev/null \
        | sed -nE 's/^BuildKit version:[[:space:]]+v?([0-9]+\.[0-9]+).*/\1/p' | head -1)
    if [[ -z "${version}" ]]; then
        echo "Could not determine the BuildKit version of the active builder" >&2
        return 1
    fi
    if (( ${version%%.*} == 0 && ${version#*.} < 14 )); then
        echo "BuildKit ${version} cannot produce reproducible ROCm images; need >= 0.14" >&2
        return 1
    fi
    echo "BuildKit ${version}: reproducible exports enabled (SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH})"
}

validate_cache_branch_tag() {
    local name="$1"
    local value="$2"

    if [[ -n "${value}" \
        && (! "${value}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ \
            || ${#value} -gt ${ROCM_CACHE_BRANCH_TAG_MAX_LEN}) ]]; then
        echo "Invalid ${name}; expected a Docker tag component of at most ${ROCM_CACHE_BRANCH_TAG_MAX_LEN} characters" >&2
        return 1
    fi
}

prepare_git_cache_metadata() {
    local cache_branch_name=""
    local cache_base_branch="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}"
    local target_repo_slug=""
    local target_repo_url=""
    local merge_base_ref=""

    if [[ -z "${PARENT_COMMIT:-}" || -z "${VLLM_MERGE_BASE_COMMIT:-}" ]] \
        && git rev-parse --is-shallow-repository 2>/dev/null | grep -q "true"; then
        echo "Shallow clone detected - deepening for cache key computation"
        git_fetch_for_cache --filter=tree:0 --no-tags --deepen=1 \
            origin "$(git rev-parse HEAD)" || true
    fi

    if [[ -z "${PARENT_COMMIT:-}" ]]; then
        PARENT_COMMIT=$(git rev-parse HEAD~1 2>/dev/null || echo "")
        if [[ -n "${PARENT_COMMIT}" ]]; then
            export PARENT_COMMIT
            echo "Computed parent commit for cache fallback: ${PARENT_COMMIT}"
        else
            echo "Could not determine parent commit"
        fi
    else
        echo "Using provided PARENT_COMMIT: ${PARENT_COMMIT}"
    fi

    if [[ -z "${ROCM_CACHE_BRANCH_TAG:-}" ]]; then
        cache_branch_name=$(select_cache_branch_name)
        if [[ -z "${cache_branch_name}" && "${BUILDKITE_PULL_REQUEST:-false}" != "false" ]]; then
            cache_branch_name="pr-${BUILDKITE_PULL_REQUEST}"
            echo "Using pull request number for ROCm branch cache tag: ${cache_branch_name}"
        fi
    fi

    if [[ -z "${ROCM_CACHE_BRANCH_TAG:-}" && -n "${cache_branch_name}" ]]; then
        ROCM_CACHE_BRANCH_TAG=$(
            compose_cache_branch_tag "$(get_buildkite_repo_slug)" "${cache_branch_name}"
        )
        export ROCM_CACHE_BRANCH_TAG
        echo "Computed ROCm branch cache tag: ${ROCM_CACHE_BRANCH_TAG} (from ${cache_branch_name})"
    elif [[ -n "${ROCM_CACHE_BRANCH_TAG:-}" ]]; then
        echo "Using provided ROCM_CACHE_BRANCH_TAG: ${ROCM_CACHE_BRANCH_TAG}"
    elif [[ -n "${BUILDKITE_BRANCH:-}" ]]; then
        echo "Skipping ROCm branch cache tag: no usable branch name found"
        echo "  BUILDKITE_BRANCH=${BUILDKITE_BRANCH}"
    fi

    if [[ -z "${ROCM_CACHE_UPSTREAM_BRANCH_TAG:-}" \
          && -n "${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-}" \
          && "${BUILDKITE_PULL_REQUEST:-false}" != "false" ]]; then
        target_repo_slug=$(get_buildkite_target_repo_slug)
        ROCM_CACHE_UPSTREAM_BRANCH_TAG=$(
            compose_cache_branch_tag "${target_repo_slug}" "${BUILDKITE_PULL_REQUEST_BASE_BRANCH}"
        )
        export ROCM_CACHE_UPSTREAM_BRANCH_TAG
        echo "Computed ROCm upstream branch cache tag: ${ROCM_CACHE_UPSTREAM_BRANCH_TAG}"
    elif [[ -n "${ROCM_CACHE_UPSTREAM_BRANCH_TAG:-}" ]]; then
        echo "Using provided ROCM_CACHE_UPSTREAM_BRANCH_TAG: ${ROCM_CACHE_UPSTREAM_BRANCH_TAG}"
    fi

    validate_cache_branch_tag \
        ROCM_CACHE_BRANCH_TAG "${ROCM_CACHE_BRANCH_TAG:-}"
    validate_cache_branch_tag \
        ROCM_CACHE_UPSTREAM_BRANCH_TAG "${ROCM_CACHE_UPSTREAM_BRANCH_TAG:-}"

    if [[ -z "${VLLM_MERGE_BASE_COMMIT:-}" ]]; then
        target_repo_url=$(get_buildkite_target_repo_url)
        merge_base_ref="refs/remotes/vllm-cache-upstream/${cache_base_branch}"
        git_fetch_for_cache --no-tags --depth=200 "${target_repo_url}" \
            "+refs/heads/${cache_base_branch}:${merge_base_ref}" 2>/dev/null || true
        VLLM_MERGE_BASE_COMMIT=$(git merge-base HEAD "${merge_base_ref}" 2>/dev/null || echo "")
        if [[ -z "${VLLM_MERGE_BASE_COMMIT}" ]]; then
            git_fetch_for_cache --no-tags --deepen=1000 "${target_repo_url}" \
                "+refs/heads/${cache_base_branch}:${merge_base_ref}" 2>/dev/null || true
            VLLM_MERGE_BASE_COMMIT=$(git merge-base HEAD "${merge_base_ref}" 2>/dev/null || echo "")
        fi
        if [[ -n "${VLLM_MERGE_BASE_COMMIT}" ]]; then
            export VLLM_MERGE_BASE_COMMIT
            echo "Computed merge base commit for cache fallback: ${VLLM_MERGE_BASE_COMMIT}"
        else
            echo "Could not determine merge base with ${cache_base_branch}"
        fi
    else
        echo "Using provided VLLM_MERGE_BASE_COMMIT: ${VLLM_MERGE_BASE_COMMIT}"
    fi
}

uses_rocm_csrc_cache() {
    case "${TARGET}" in
        csrc-rocm-ci \
            | test-rocm-ci \
            | test-rocm-ci-thin \
            | export-wheel-rocm \
            | smoke-test-rocm-ci)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

uses_rocm_rust_cache() {
    case "${TARGET}" in
        rust-rocm-ci \
            | test-rocm-ci \
            | test-rocm-ci-thin \
            | export-wheel-rocm \
            | smoke-test-rocm-ci)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

compute_rocm_csrc_content_hash() {
    local bake_dir=""
    local dockerfile_rocm=""
    local content_files="${ROCM_CSRC_CONTENT_FILES:-${DEFAULT_ROCM_CSRC_CONTENT_FILES}}"
    local stages="${ROCM_CSRC_DOCKERFILE_STAGES:-${DEFAULT_ROCM_CSRC_DOCKERFILE_STAGES}}"
    local -a content_paths=()
    local -a content_args=()

    bake_dir=$(dirname "${VLLM_BAKE_FILE}")
    dockerfile_rocm="${CI_BASE_DOCKERFILE:-${bake_dir}/Dockerfile.rocm}"
    read -r -a content_paths <<< "${content_files}"
    mapfile -t content_args < <(
        get_content_arg_names "${dockerfile_rocm}" "${stages}" "${ROCM_CSRC_CONTENT_ARGS:-}"
    )

    {
        printf 'csrc-input-files-hash:%s\n' "$(compute_content_hash "${content_paths[@]}")"
        printf 'dockerfile:%s\n' "${dockerfile_rocm}"
        printf 'resolved-build-args:\n'
        hash_dockerfile_arg_values "${dockerfile_rocm}" "${content_args[@]}"
        printf 'dockerfile-stages:%s\n' "${stages}"
        if content_regular_file "${dockerfile_rocm}"; then
            hash_dockerfile_stages "${dockerfile_rocm}" "${stages}"
        else
            printf 'missing:%s\n' "${dockerfile_rocm}"
        fi
    } | sha256sum | cut -d' ' -f1
}

compute_rocm_csrc_content_hash_if_needed() {
    local cache_repo="${DOCKERHUB_CACHE_REPO:-rocm/vllm-ci-cache}"
    local write_scope="${CACHE_WRITE_SCOPE}"

    if [[ "${ROCM_CSRC_CONTENT_CACHE:-1}" == "0" ]] || ! uses_rocm_csrc_cache; then
        return 0
    fi

    ROCM_CSRC_CONTENT_HASH=$(compute_rocm_csrc_content_hash)
    ROCM_CSRC_TRUSTED_CONTENT_CACHE_REF="${cache_repo}:csrc-rocm-input-${ROCM_CSRC_CONTENT_HASH}"
    ROCM_CSRC_CONTENT_CACHE_REF="${ROCM_CSRC_TRUSTED_CONTENT_CACHE_REF}"
    if [[ -n "${write_scope}" ]]; then
        ROCM_CSRC_CONTENT_CACHE_REF="${ROCM_CSRC_CONTENT_CACHE_REF}-${write_scope}"
    fi
    export ROCM_CSRC_CONTENT_HASH
    export ROCM_CSRC_TRUSTED_CONTENT_CACHE_REF
    export ROCM_CSRC_CONTENT_CACHE_REF
    echo "ROCm csrc content cache ref: ${ROCM_CSRC_CONTENT_CACHE_REF}"
}

compute_rocm_rust_content_hash() {
    local bake_dir=""
    local dockerfile_rocm=""
    local content_files="${ROCM_RUST_CONTENT_FILES:-${DEFAULT_ROCM_RUST_CONTENT_FILES}}"
    local stages="${ROCM_RUST_DOCKERFILE_STAGES:-${DEFAULT_ROCM_RUST_DOCKERFILE_STAGES}}"
    local -a content_paths=()
    local -a content_args=()

    bake_dir=$(dirname "${VLLM_BAKE_FILE}")
    dockerfile_rocm="${CI_BASE_DOCKERFILE:-${bake_dir}/Dockerfile.rocm}"
    read -r -a content_paths <<< "${content_files}"
    mapfile -t content_args < <(
        get_content_arg_names \
            "${dockerfile_rocm}" "${stages}" "${ROCM_RUST_CONTENT_ARGS:-}"
    )

    {
        printf 'rust-input-files-hash:%s\n' "$(compute_content_hash "${content_paths[@]}")"
        printf 'dockerfile:%s\n' "${dockerfile_rocm}"
        printf 'resolved-build-args:\n'
        hash_dockerfile_arg_values "${dockerfile_rocm}" "${content_args[@]}"
        printf 'dockerfile-stages:%s\n' "${stages}"
        if content_regular_file "${dockerfile_rocm}"; then
            hash_dockerfile_stages "${dockerfile_rocm}" "${stages}"
        else
            printf 'missing:%s\n' "${dockerfile_rocm}"
        fi
    } | sha256sum | cut -d' ' -f1
}

compute_rocm_rust_content_hash_if_needed() {
    local cache_repo="${DOCKERHUB_CACHE_REPO:-rocm/vllm-ci-cache}"
    local write_scope="${CACHE_WRITE_SCOPE}"

    if [[ "${ROCM_RUST_CONTENT_CACHE:-1}" == "0" ]] || ! uses_rocm_rust_cache; then
        return 0
    fi

    ROCM_RUST_CONTENT_HASH=$(compute_rocm_rust_content_hash)
    ROCM_RUST_TRUSTED_CONTENT_CACHE_REF="${cache_repo}:rust-rocm-input-${ROCM_RUST_CONTENT_HASH}"
    ROCM_RUST_CONTENT_CACHE_REF="${ROCM_RUST_TRUSTED_CONTENT_CACHE_REF}"
    if [[ -n "${write_scope}" ]]; then
        ROCM_RUST_CONTENT_CACHE_REF="${ROCM_RUST_CONTENT_CACHE_REF}-${write_scope}"
    fi
    export ROCM_RUST_CONTENT_HASH
    export ROCM_RUST_TRUSTED_CONTENT_CACHE_REF
    export ROCM_RUST_CONTENT_CACHE_REF
    echo "ROCm Rust content cache ref: ${ROCM_RUST_CONTENT_CACHE_REF}"
}

write_hcl_string_list_entries() {
    local indent="$1"
    local value=""
    shift

    for value in "$@"; do
        value="${value//\\/\\\\}"
        value="${value//\"/\\\"}"
        printf '%s"%s",\n' "${indent}" "${value}"
    done
}

hcl_escape_string() {
    local value="$1"

    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    printf '%s' "${value}"
}

write_hcl_string_list() {
    local indent="$1"
    shift

    printf '%s[\n' "${indent}"
    write_hcl_string_list_entries "${indent}  " "$@"
    printf '%s]\n' "${indent}"
}

write_rocm_build_arg_override() {
    local bake_dir=""
    local dockerfile_rocm=""
    local -a arg_names=()
    local arg_name=""
    local arg_value=""

    bake_dir=$(dirname "${VLLM_BAKE_FILE}")
    dockerfile_rocm="${CI_BASE_DOCKERFILE:-${bake_dir}/Dockerfile.rocm}"
    mapfile -t arg_names < <(
        {
            get_content_arg_names \
                "${dockerfile_rocm}" \
                "${ROCM_CSRC_DOCKERFILE_STAGES:-${DEFAULT_ROCM_CSRC_DOCKERFILE_STAGES}}" \
                "${ROCM_CSRC_CONTENT_ARGS:-}"
            get_content_arg_names \
                "${dockerfile_rocm}" \
                "${ROCM_RUST_DOCKERFILE_STAGES:-${DEFAULT_ROCM_RUST_DOCKERFILE_STAGES}}" \
                "${ROCM_RUST_CONTENT_ARGS:-}"
        } | awk 'NF && !seen[$0]++'
    )

    {
        cat <<EOF
target "_common-rocm" {
  args = {
EOF
        for arg_name in "${arg_names[@]}"; do
            [[ -n "${arg_name}" ]] || continue
            arg_value=$(resolve_dockerfile_arg_value "${dockerfile_rocm}" "${arg_name}")
            [[ -n "${arg_value}" ]] || continue
            printf '    %s = "%s"\n' "${arg_name}" "$(hcl_escape_string "${arg_value}")"
        done
        cat <<EOF
  }
}
EOF
    } > "${ROCM_ARG_OVERRIDE_PATH}"

    BAKE_FILES+=(-f "${ROCM_ARG_OVERRIDE_PATH}")
    echo "Appended resolved ROCm Docker ARG override"
}

write_hcl_string_list_attr() {
    local indent="$1"
    local attr="$2"
    shift 2

    printf '%s%s = [\n' "${indent}" "${attr}"
    write_hcl_string_list_entries "${indent}  " "$@"
    printf '%s]\n' "${indent}"
}

validate_cache_export_mode() {
    local mode="$1"
    local env_name="$2"

    case "${mode}" in
        min|max)
            ;;
        *)
            echo "Error: ${env_name} must be one of: min, max"
            exit 1
            ;;
    esac
}

validate_content_cache_export_mode() {
    local mode="$1"
    local env_name="$2"

    case "${mode}" in
        missing|always|never)
            ;;
        *)
            echo "Error: ${env_name} must be one of: missing, always, never"
            exit 1
            ;;
    esac
}

should_export_content_cache_ref() {
    local cache_ref="$1"
    local cache_name="$2"
    local trusted_ref="${3:-${cache_ref}}"
    local mode="${ROCM_CONTENT_CACHE_EXPORT_MODE:-missing}"

    case "${mode}" in
        always)
            echo "${cache_name} content cache export mode is always"
            return 0
            ;;
        never)
            echo "${cache_name} content cache export mode is never"
            return 1
            ;;
        missing|"")
            if registry_ref_exists_with_retry "${trusted_ref}"; then
                echo "${cache_name} trusted content cache is visible: ${trusted_ref}"
                return 1
            fi
            if [[ "${cache_ref}" != "${trusted_ref}" ]] \
                && registry_ref_exists_with_retry "${cache_ref}"; then
                echo "${cache_name} scoped content cache is visible: ${cache_ref}"
                return 1
            fi
            echo "${cache_name} content cache is missing: ${cache_ref}"
            return 0
            ;;
        *)
            echo "Error: ROCM_CONTENT_CACHE_EXPORT_MODE must be one of: missing, always, never"
            exit 1
            ;;
    esac
}

write_rocm_cache_override() {
    local cache_repo="${DOCKERHUB_CACHE_REPO:-rocm/vllm-ci-cache}"
    local content_cache_export_mode="${ROCM_CONTENT_CACHE_EXPORT_MODE:-missing}"
    local csrc_cache_to_mode="${ROCM_CSRC_CACHE_TO_MODE:-max}"
    local rust_cache_to_mode="${ROCM_RUST_CACHE_TO_MODE:-max}"
    local rocm_cache_to_mode="${ROCM_FINAL_CACHE_TO_MODE:-min}"
    local -a csrc_content_cache_from=()
    local -a rust_content_cache_from=()
    local -a combined_content_cache_from=()
    local -a csrc_cache_to=()
    local -a rust_cache_to=()
    local -a rocm_cache_to=()
    local -a export_wheel_cache_to=()
    local export_csrc_cache=1
    local export_rust_cache=1

    if ! uses_rocm_csrc_cache && ! uses_rocm_rust_cache; then
        return 0
    fi

    validate_content_cache_export_mode \
        "${content_cache_export_mode}" \
        "ROCM_CONTENT_CACHE_EXPORT_MODE"
    validate_cache_export_mode "${csrc_cache_to_mode}" "ROCM_CSRC_CACHE_TO_MODE"
    validate_cache_export_mode "${rust_cache_to_mode}" "ROCM_RUST_CACHE_TO_MODE"
    validate_cache_export_mode "${rocm_cache_to_mode}" "ROCM_FINAL_CACHE_TO_MODE"
    echo "ROCm content cache export mode: ${content_cache_export_mode}"
    echo "ROCm csrc cache export mode: ${csrc_cache_to_mode}"
    echo "ROCm Rust fallback cache export mode: ${rust_cache_to_mode}"
    echo "ROCm final image cache export mode: ${rocm_cache_to_mode}"

    if [[ -n "${ROCM_CSRC_CONTENT_CACHE_REF:-}" ]]; then
        csrc_content_cache_from+=(
            "type=registry,ref=${ROCM_CSRC_TRUSTED_CONTENT_CACHE_REF}"
        )
        if [[ "${ROCM_CSRC_CONTENT_CACHE_REF}" != \
            "${ROCM_CSRC_TRUSTED_CONTENT_CACHE_REF}" ]]; then
            csrc_content_cache_from+=(
                "type=registry,ref=${ROCM_CSRC_CONTENT_CACHE_REF}"
            )
        fi
        if should_export_content_cache_ref \
            "${ROCM_CSRC_CONTENT_CACHE_REF}" "ROCm csrc" \
            "${ROCM_CSRC_TRUSTED_CONTENT_CACHE_REF}"; then
            csrc_cache_to+=(
                "type=registry,ref=${ROCM_CSRC_CONTENT_CACHE_REF},mode=${csrc_cache_to_mode},ignore-error=true"
            )
        else
            export_csrc_cache=0
        fi
    fi

    if [[ -n "${ROCM_RUST_CONTENT_CACHE_REF:-}" ]]; then
        rust_content_cache_from+=(
            "type=registry,ref=${ROCM_RUST_TRUSTED_CONTENT_CACHE_REF}"
        )
        if [[ "${ROCM_RUST_CONTENT_CACHE_REF}" != \
            "${ROCM_RUST_TRUSTED_CONTENT_CACHE_REF}" ]]; then
            rust_content_cache_from+=(
                "type=registry,ref=${ROCM_RUST_CONTENT_CACHE_REF}"
            )
        fi
        # Legacy commit/branch exports are only needed while the exact-input
        # content ref is absent. The exact ref itself is refreshed below.
        if ! should_export_content_cache_ref \
            "${ROCM_RUST_CONTENT_CACHE_REF}" "ROCm Rust" \
            "${ROCM_RUST_TRUSTED_CONTENT_CACHE_REF}"; then
            export_rust_cache=0
        fi
        if [[ "${content_cache_export_mode}" != "never" ]]; then
            # Refresh the exact-input ref in the original solve regardless of
            # which local or remote cache supplied the Rust result.
            rust_cache_to+=(
                "type=registry,ref=${ROCM_RUST_CONTENT_CACHE_REF},mode=min,ignore-error=true"
            )
            echo "ROCm Rust exact-input cache will be refreshed (mode=min): ${ROCM_RUST_CONTENT_CACHE_REF}"
        fi
    fi

    combined_content_cache_from=("${csrc_content_cache_from[@]}" "${rust_content_cache_from[@]}")

    # Docker Hub cache exports are best-effort. A cache-only target failure can
    # otherwise cancel the sibling image target before its manifest is pushed.
    if [[ -n "${BUILDKITE_COMMIT:-}" ]]; then
        if [[ ${export_csrc_cache} -eq 1 ]]; then
            csrc_cache_to+=(
                "type=registry,ref=${cache_repo}:csrc-rocm-${BUILDKITE_COMMIT},mode=${csrc_cache_to_mode},ignore-error=true"
            )
        fi
        if [[ ${export_rust_cache} -eq 1 ]]; then
            rust_cache_to+=(
                "type=registry,ref=${cache_repo}:rust-rocm-${BUILDKITE_COMMIT},mode=${rust_cache_to_mode},ignore-error=true"
            )
        fi
        rocm_cache_to+=(
            "type=registry,ref=${cache_repo}:rocm-${BUILDKITE_COMMIT},mode=${rocm_cache_to_mode},ignore-error=true"
        )
    fi

    if [[ -n "${ROCM_CACHE_BRANCH_TAG:-}" ]]; then
        if [[ ${export_csrc_cache} -eq 1 ]]; then
            csrc_cache_to+=(
                "type=registry,ref=${cache_repo}:csrc-rocm-branch-${ROCM_CACHE_BRANCH_TAG},mode=${csrc_cache_to_mode},ignore-error=true"
            )
        fi
        if [[ ${export_rust_cache} -eq 1 ]]; then
            rust_cache_to+=(
                "type=registry,ref=${cache_repo}:rust-rocm-branch-${ROCM_CACHE_BRANCH_TAG},mode=${rust_cache_to_mode},ignore-error=true"
            )
        fi
        rocm_cache_to+=(
            "type=registry,ref=${cache_repo}:rocm-branch-${ROCM_CACHE_BRANCH_TAG},mode=${rocm_cache_to_mode},ignore-error=true"
        )
    fi

    # Standalone image/wheel targets reach rust-build but not the cache-only
    # exporter unless it is requested explicitly.
    if ((${#rust_cache_to[@]} > 0)); then
        case "${TARGET}" in
            test-rocm-ci|export-wheel-rocm|smoke-test-rocm-ci)
                BAKE_TARGETS=("rust-rocm-ci" "${BAKE_TARGETS[@]}")
                ;;
        esac
    fi

    # test-rocm-ci exports the same final-image refs in the thin group; a
    # second exporter to one ref only races it.
    if [[ "${TARGET}" == "test-rocm-ci-thin" ]]; then
        export_wheel_cache_to=()
    else
        export_wheel_cache_to=("${rocm_cache_to[@]}")
    fi

    {
        cat <<EOF
target "csrc-rocm-ci" {
  cache-from = concat(
    get_cache_from_rocm_csrc(),
EOF
        write_hcl_string_list "    " "${csrc_content_cache_from[@]}"
        cat <<EOF
  )
EOF
        write_hcl_string_list_attr "  " "cache-to" "${csrc_cache_to[@]}"
        cat <<EOF
}

target "rust-rocm-ci" {
  cache-from = concat(
    get_cache_from_rocm_rust(),
EOF
        write_hcl_string_list "    " "${rust_content_cache_from[@]}"
        cat <<EOF
  )
EOF
        write_hcl_string_list_attr "  " "cache-to" "${rust_cache_to[@]}"
        cat <<EOF
}

target "test-rocm-ci" {
  cache-from = concat(
    get_cache_from_rocm(),
EOF
        write_hcl_string_list "    " "${combined_content_cache_from[@]}"
        cat <<EOF
  )
EOF
        write_hcl_string_list_attr "  " "cache-to" "${rocm_cache_to[@]}"
        cat <<EOF
}

target "smoke-test-rocm-ci" {
  cache-from = concat(
    get_cache_from_rocm(),
EOF
        write_hcl_string_list "    " "${combined_content_cache_from[@]}"
        cat <<EOF
  )
}

target "export-wheel-rocm" {
  cache-from = concat(
    get_cache_from_rocm(),
EOF
        write_hcl_string_list "    " "${combined_content_cache_from[@]}"
        cat <<EOF
  )
EOF
        write_hcl_string_list_attr "  " "cache-to" "${export_wheel_cache_to[@]}"
        cat <<EOF
}
EOF
    } > "${CSRC_CACHE_OVERRIDE_PATH}"

    BAKE_FILES+=(-f "${CSRC_CACHE_OVERRIDE_PATH}")
    echo "Appended ROCm cache override with non-fatal registry exports"
}

print_bake_config() {
    echo "--- :page_facing_up: Resolved bake configuration"
    docker buildx bake "${BAKE_ALLOW_ARGS[@]}" \
        "${BAKE_FILES[@]}" --print "${BAKE_TARGETS[@]}" | tee "${BAKE_CONFIG_FILE}"

    if command -v buildkite-agent >/dev/null 2>&1 && [[ -n "${BUILDKITE_BUILD_NUMBER:-}" ]]; then
        buildkite-agent artifact upload "${BAKE_CONFIG_FILE}" || true
        echo "Uploaded ${BAKE_CONFIG_FILE} as Buildkite artifact"
    else
        echo "Saved bake config to ${BAKE_CONFIG_FILE} (not in Buildkite, skipping upload)"
    fi
}

run_bake() {
    echo "--- :docker: Building ${TARGET}"
    docker buildx bake \
        "${BAKE_ALLOW_ARGS[@]}" \
        "${BAKE_FILES[@]}" \
        --progress "${BUILDKIT_PROGRESS:-plain}" \
        "${BAKE_TARGETS[@]}"
    echo "--- :white_check_mark: Build complete"
}

# Thin images already contain the installed wheel and test workspace; only the
# python-only compile job needs the .whl itself.
upload_thin_wheel_artifact() {
    local artifact_dir="artifacts/vllm-rocm-wheel"
    local -a wheels=()

    [[ "${TARGET}" == "test-rocm-ci-thin" ]] || return 0
    mapfile -t wheels < <(find ./wheel-export -maxdepth 1 -type f -name '*.whl' -print)
    if ((${#wheels[@]} != 1)); then
        echo "Expected exactly one exported vLLM wheel; found ${#wheels[@]}" >&2
        return 1
    fi
    rm -rf "${artifact_dir}" && mkdir -p "${artifact_dir}" || return 1
    cp "${wheels[0]}" "${artifact_dir}/" || return 1
    (cd "${artifact_dir}" && sha256sum -- *.whl > "$(basename "${wheels[0]}").sha256") || return 1
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent artifact upload "${artifact_dir}/*" || return 1
    else
        echo "Not in Buildkite; wheel left in ${artifact_dir}"
    fi
}

main() {
    init_config "$@"
    configure_cache_write_scope
    print_header
    validate_inputs
    load_ci_hcl
    init_bake_files
    pin_base_image
    setup_builder
    require_reproducible_builder
    prepare_git_cache_metadata
    # prepare_git_cache_metadata may deepen a shallow checkout. Derive archival
    # version metadata only after that lookup sees the available tag history.
    prepare_ci_build_context
    write_rocm_build_arg_override
    compute_rocm_csrc_content_hash_if_needed
    compute_rocm_rust_content_hash_if_needed
    write_rocm_cache_override
    # Keep the context override last so every bake target uses the owned tree.
    write_build_context_override
    print_bake_config
    if [[ "${BAKE_PRINT_ONLY:-0}" == "1" ]]; then
        echo "BAKE_PRINT_ONLY=1 set; skipping build"
        return 0
    fi
    if has_local_outputs; then
        # Outputs, not caches: never let a failed or retried build reuse them.
        rm -rf ./wheel-export ./build/rocm-smoke-export
    fi
    run_bake
    verify_rocm_smoke_export
    upload_thin_wheel_artifact
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
