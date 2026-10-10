# ci-rocm.hcl - CI-specific configuration for vLLM ROCm Docker builds
#
# This file lives in the vLLM repo at docker/ci-rocm.hcl so ROCm Docker
# build mechanics can evolve with Dockerfile.rocm and docker-bake-rocm.hcl.
# Used with: docker buildx bake -f docker/docker-bake-rocm.hcl -f docker/ci-rocm.hcl test-rocm-ci-thin
#
# Registry cache: Docker Hub (rocm/vllm-ci-cache) is used exclusively.
# AMD build agents already have Docker Hub credentials (they push the test
# image to rocm/vllm-ci), so no additional credential setup is required.
# ROCm CI uses Docker Hub for BuildKit layer cache by default. A separate
# compiler cache can be enabled with USE_SCCACHE=1 when AMD provides a shared
# S3-compatible cache endpoint.

# CI metadata

variable "BUILDKITE_COMMIT" {
  default = ""
}

variable "BUILDKITE_BUILD_NUMBER" {
  default = ""
}

variable "BUILDKITE_BUILD_ID" {
  default = ""
}

variable "PARENT_COMMIT" {
  default = ""
}

# Merge-base of HEAD with main - provides a more stable cache fallback than
# parent commit for long-lived PRs. Mirrors the VLLM_MERGE_BASE_COMMIT
# pattern used in the shared ci.hcl file. Auto-computed by ci-bake-rocm.sh
# when unset.
variable "VLLM_MERGE_BASE_COMMIT" {
  default = ""
}

# Bridge to vLLM's COMMIT variable for OCI labels
variable "COMMIT" {
  default = BUILDKITE_COMMIT
}

# Image tags (set by CI)

variable "IMAGE_TAG" {
  default = ""
}

variable "IMAGE_TAG_LATEST" {
  default = ""
}

# Constant per base lineage (read from docker/Dockerfile.rocm_base by the CI
# scripts). With rewrite-timestamp=true it makes rebuilds of unchanged steps
# byte-identical, so registries and nodes dedupe them.
variable "SOURCE_DATE_EPOCH" {
  default = ""
}

# zstd decompresses ~1.8x faster than gzip on pull+unpack and exports ~8x
# faster (measured on containerd 2.1); same attributes as
# .buildkite/image_build/zstd.hcl. No force-compression here: it would make
# BuildKit fetch and recompress inherited parent layers.
variable "ROCM_IMAGE_OUTPUT_ATTRS" {
  default = "oci-mediatypes=true,compression=zstd,compression-level=3,rewrite-timestamp=true"
}

# ROCm-specific GPU architecture targets

variable "PYTORCH_ROCM_ARCH" {
  default = "gfx90a;gfx942;gfx950"
}

# Runtime image the test stage adds one vLLM layer to (Dockerfile.rocm_base
# target runtime-ci). CI sets it to the digest from build-runtime.sh.
variable "CI_BASE_IMAGE" {
  default = "rocm/vllm-dev:ci_base"
}

# Leave CI_MAX_JOBS empty so the Dockerfile falls back to $(nproc) and uses
# the full builder parallelism. Operators can still override this per build.
variable "CI_MAX_JOBS" {
  default = ""
}

# Docker Hub registry cache for AMD builds.
#
# A separate repo (rocm/vllm-ci-cache) is used for BuildKit layer cache.
# Final-image cache exports use mode=min to reduce the volume of data pushed.
# Source-scoped csrc cache exports default to mode=max so fresh workers can
# recover more of the native build graph when ROCm extension inputs change.
# NOTE: mode=min still includes all layers referenced by the final image
# manifest, including inherited base layers (~7.25GB ROCm runtime).
# Docker Hub auto-creates the repo on first push.
#
# Final-image cache stays commit-scoped. Branch-to-branch reuse for the test
# image comes from importing the parent and merge-base commit cache refs.
#
# The source-scoped native cache is exported both per-commit and per-branch so
# ROCm extension rebuilds are shareable within the same commit reruns and across
# consecutive commits on the same branch without depending on a single global
# latest tag.

variable "DOCKERHUB_CACHE_REPO" {
  default = "rocm/vllm-ci-cache"
}

variable "ROCM_CACHE_BRANCH_TAG" {
  default = ""
}

variable "ROCM_CACHE_UPSTREAM_BRANCH_TAG" {
  default = ""
}

variable "ROCM_CSRC_CACHE_TO_MODE" {
  default = "max"
}

variable "ROCM_RUST_CACHE_TO_MODE" {
  default = "max"
}

variable "ROCM_FINAL_CACHE_TO_MODE" {
  default = "min"
}

# Functions

function "get_cache_from_rocm" {
  params = []
  result = compact([
    # Exact commit hit - fastest cache on re-runs of the same commit
    BUILDKITE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rocm-${BUILDKITE_COMMIT}" : "",
    # Parent commit - useful cache for incremental changes
    PARENT_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rocm-${PARENT_COMMIT}" : "",
    # Merge-base with main - stable fallback for long-lived or rebased PRs;
    # maps to a real main-branch commit whose cache layers are likely warm
    VLLM_MERGE_BASE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rocm-${VLLM_MERGE_BASE_COMMIT}" : "",
    # Import the source-scoped native build cache as well so builds whose
    # Python/package layers changed can still reuse compiled ROCm objects.
    BUILDKITE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-${BUILDKITE_COMMIT}" : "",
    PARENT_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-${PARENT_COMMIT}" : "",
    VLLM_MERGE_BASE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-${VLLM_MERGE_BASE_COMMIT}" : "",
    ROCM_CACHE_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-branch-${ROCM_CACHE_BRANCH_TAG}" : "",
    ROCM_CACHE_UPSTREAM_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-branch-${ROCM_CACHE_UPSTREAM_BRANCH_TAG}" : "",
    # Import the source-scoped Rust frontend cache so non-Rust changes do not
    # force a fresh cargo release build.
    BUILDKITE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-${BUILDKITE_COMMIT}" : "",
    PARENT_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-${PARENT_COMMIT}" : "",
    VLLM_MERGE_BASE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-${VLLM_MERGE_BASE_COMMIT}" : "",
    ROCM_CACHE_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-branch-${ROCM_CACHE_BRANCH_TAG}" : "",
    ROCM_CACHE_UPSTREAM_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-branch-${ROCM_CACHE_UPSTREAM_BRANCH_TAG}" : "",
    # Branch-scoped full image cache - fallback when parent-commit cache is evicted
    ROCM_CACHE_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rocm-branch-${ROCM_CACHE_BRANCH_TAG}" : "",
    ROCM_CACHE_UPSTREAM_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rocm-branch-${ROCM_CACHE_UPSTREAM_BRANCH_TAG}" : "",
  ])
}

function "get_cache_to_rocm" {
  params = []
  result = compact([
    # Commit-scoped cache for exact re-runs.
    BUILDKITE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rocm-${BUILDKITE_COMMIT},mode=${ROCM_FINAL_CACHE_TO_MODE},compression=zstd" : "",
    # Branch-scoped cache so later commits on the same branch can reuse the full
    # image layers when the parent-commit cache is evicted. Unlike the old
    # rocm-latest tag (which caused duplicate exporter 400s), this is per-branch.
    ROCM_CACHE_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rocm-branch-${ROCM_CACHE_BRANCH_TAG},mode=${ROCM_FINAL_CACHE_TO_MODE},compression=zstd" : "",
  ])
}

function "get_cache_from_rocm_csrc" {
  params = []
  result = compact([
    BUILDKITE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-${BUILDKITE_COMMIT}" : "",
    PARENT_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-${PARENT_COMMIT}" : "",
    VLLM_MERGE_BASE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-${VLLM_MERGE_BASE_COMMIT}" : "",
    ROCM_CACHE_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-branch-${ROCM_CACHE_BRANCH_TAG}" : "",
    ROCM_CACHE_UPSTREAM_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-branch-${ROCM_CACHE_UPSTREAM_BRANCH_TAG}" : "",
  ])
}

function "get_cache_to_rocm_csrc" {
  params = []
  result = compact([
    # Export the exact-commit native cache for same-commit reruns.
    BUILDKITE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-${BUILDKITE_COMMIT},mode=${ROCM_CSRC_CACHE_TO_MODE},compression=zstd" : "",
    # Export the branch-scoped native cache so later commits on the same branch
    # can reuse compiled ROCm objects even when the exact parent cache is absent.
    ROCM_CACHE_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-branch-${ROCM_CACHE_BRANCH_TAG},mode=${ROCM_CSRC_CACHE_TO_MODE},compression=zstd" : "",
  ])
}

function "get_cache_from_rocm_rust" {
  params = []
  result = compact([
    BUILDKITE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-${BUILDKITE_COMMIT}" : "",
    PARENT_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-${PARENT_COMMIT}" : "",
    VLLM_MERGE_BASE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-${VLLM_MERGE_BASE_COMMIT}" : "",
    ROCM_CACHE_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-branch-${ROCM_CACHE_BRANCH_TAG}" : "",
    ROCM_CACHE_UPSTREAM_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-branch-${ROCM_CACHE_UPSTREAM_BRANCH_TAG}" : "",
  ])
}

function "get_cache_to_rocm_rust" {
  params = []
  result = compact([
    # Export exact-commit and branch-scoped Rust caches. A content-addressed
    # cache ref is appended by ci-bake-rocm.sh when that wrapper is used.
    BUILDKITE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-${BUILDKITE_COMMIT},mode=${ROCM_RUST_CACHE_TO_MODE},compression=zstd" : "",
    ROCM_CACHE_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-branch-${ROCM_CACHE_BRANCH_TAG},mode=${ROCM_RUST_CACHE_TO_MODE},compression=zstd" : "",
  ])
}

# CI targets

target "_ci-rocm" {
  annotations = [
    "manifest:vllm.buildkite.build_number=${BUILDKITE_BUILD_NUMBER}",
    "manifest:vllm.buildkite.build_id=${BUILDKITE_BUILD_ID}",
  ]
  args = {
    ARG_PYTORCH_ROCM_ARCH = PYTORCH_ROCM_ARCH
    CI_BASE_IMAGE         = CI_BASE_IMAGE
    ROCM_SMOKE_ID         = BUILDKITE_BUILD_ID
    VLLM_CI_COMMIT        = BUILDKITE_COMMIT
    max_jobs              = CI_MAX_JOBS
    SOURCE_DATE_EPOCH     = SOURCE_DATE_EPOCH
  }
}

target "test-rocm-ci" {
  inherits   = ["_common-rocm", "_ci-rocm", "_labels"]
  target     = "test"
  cache-from = get_cache_from_rocm()
  cache-to   = get_cache_to_rocm()
  tags = compact([
    IMAGE_TAG,
    IMAGE_TAG_LATEST,
  ])
  output = ["type=registry,${ROCM_IMAGE_OUTPUT_ATTRS}"]
}

# Validate the test image in the shared BuildKit graph and export only the
# success marker. This avoids pulling the multi-GB image into the host daemon.
target "smoke-test-rocm-ci" {
  inherits   = ["_common-rocm", "_ci-rocm"]
  target     = "export_test_smoke"
  cache-from = get_cache_from_rocm()
  output     = ["type=local,dest=./build/rocm-smoke-export"]
}

# Cache-only target for the source-scoped ROCm native build stage.
# This persists the csrc-build stage in the registry cache even though the
# final test image only consumes it indirectly while packaging the wheel.
target "csrc-rocm-ci" {
  inherits   = ["_common-rocm", "_ci-rocm"]
  target     = "csrc-build"
  cache-from = get_cache_from_rocm_csrc()
  cache-to   = get_cache_to_rocm_csrc()
  output     = ["type=cacheonly"]
}

# Cache-only target for the Rust frontend build stage. Final-image cache
# exports use mode=min and do not reliably persist intermediate cargo layers,
# so Rust gets its own source-scoped cache target.
target "rust-rocm-ci" {
  inherits   = ["_common-rocm", "_ci-rocm"]
  target     = "rust-build"
  cache-from = get_cache_from_rocm_rust()
  cache-to   = get_cache_to_rocm_rust()
  output     = ["type=cacheonly"]
}

# Keep wheel export on the same CI graph as the test image build so the
# shared build_vllm/export_vllm stages resolve identically within one bake
# invocation. Without this, export-wheel-rocm uses the plain local target
# args while test-rocm-ci uses CI-only args, which can lead to separate
# cache lineages and inconsistent export_vllm results.
target "export-wheel-rocm" {
  inherits   = ["_common-rocm", "_ci-rocm"]
  target     = "export_vllm"
  cache-from = get_cache_from_rocm()
  cache-to   = get_cache_to_rocm()
  output     = ["type=local,dest=./wheel-export"]
}

# Per-commit test image as a single layer on the runtime image (COPY --link),
# so the builder never pulls the runtime and the push is only the vLLM layer.
# GPU jobs pull this image directly. The smoke check runs in the same BuildKit
# graph, which already holds the runtime as the csrc-build base.
group "test-rocm-ci-thin" {
  targets = ["rust-rocm-ci", "csrc-rocm-ci", "test-rocm-ci", "export-wheel-rocm", "smoke-test-rocm-ci"]
}
