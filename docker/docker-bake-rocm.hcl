# docker-bake-rocm.hcl - vLLM ROCm Docker build configuration
#
# This file lives in the vLLM repo at docker/docker-bake-rocm.hcl
# Equivalent of docker-bake.hcl for ROCm builds.
#
# Usage:
#   docker buildx bake -f docker/docker-bake-rocm.hcl              # Build test (default)
#   docker buildx bake -f docker/docker-bake-rocm.hcl final-rocm   # Build final image
#   docker buildx bake -f docker/docker-bake-rocm.hcl --print      # Show resolved config
#
# CI usage (with the vLLM-owned CI overlay):
#   docker buildx bake -f docker/docker-bake-rocm.hcl -f docker/ci-rocm.hcl test-rocm-ci-thin

variable "MAX_JOBS" {
  # Empty string lets the Dockerfile fall back to $(nproc) via
  # MAX_JOBS="${MAX_JOBS:-$(nproc)}" in each RUN step, which uses all
  # available cores on whatever machine the build runs on.
  # Override with --set '*.args.max_jobs=8' for local builds on small machines.
  default = ""
}

variable "PYTORCH_ROCM_ARCH" {
  default = "gfx90a;gfx942;gfx950"
}

variable "COMMIT" {
  default = ""
}

variable "CI_BASE_DOCKERFILE" {
  default = "docker/Dockerfile.rocm"
}

# REMOTE_VLLM=0: use local source via Docker build context (ONBUILD COPY ./ vllm/)
# REMOTE_VLLM=1: clone from GitHub at VLLM_BRANCH (standalone builds without local source)
variable "REMOTE_VLLM" {
  default = "0"
}

variable "VLLM_BRANCH" {
  default = "main"
}

# BASE_IMAGE: the runtime image the vLLM wheel is compiled in (and release
# images start from). CI sets it to the digest selected by build-runtime.sh.
variable "BASE_IMAGE" {
  default = "rocm/vllm-dev:base"
}

# CI_BASE_IMAGE: the image the test stage adds the vLLM layer to
# (Dockerfile.rocm_base target runtime-ci). CI sets it to the same digest as
# BASE_IMAGE.
variable "CI_BASE_IMAGE" {
  default = "rocm/vllm-dev:ci_base"
}

group "default" {
  targets = ["test-rocm"]
}

target "_common-rocm" {
  dockerfile = CI_BASE_DOCKERFILE
  context    = "."
  args = {
    max_jobs                        = MAX_JOBS
    ARG_PYTORCH_ROCM_ARCH           = PYTORCH_ROCM_ARCH
    REMOTE_VLLM                     = REMOTE_VLLM
    VLLM_BRANCH                     = VLLM_BRANCH
    CI_BASE_IMAGE                   = CI_BASE_IMAGE
    BASE_IMAGE                      = BASE_IMAGE
  }
}

target "_labels" {
  labels = {
    "org.opencontainers.image.source"      = "https://github.com/vllm-project/vllm"
    "org.opencontainers.image.vendor"      = "vLLM"
    "org.opencontainers.image.title"       = "vLLM ROCm"
    "org.opencontainers.image.description" = "vLLM: A high-throughput and memory-efficient inference and serving engine for LLMs (ROCm)"
    "org.opencontainers.image.licenses"    = "Apache-2.0"
    "org.opencontainers.image.revision"    = COMMIT
  }
  annotations = [
    "manifest:org.opencontainers.image.revision=${COMMIT}",
  ]
}

target "test-rocm" {
  inherits = ["_common-rocm", "_labels"]
  target   = "test"
  tags     = ["rocm/vllm:test"]
  output   = ["type=docker"]
}

# Wheel export target - extracts the built vLLM wheel + test workspace
# to local disk. CI uploads the wheel for the python-only compile job.
#
# Usage:
#   docker buildx bake -f docker/docker-bake-rocm.hcl export-wheel-rocm
#   # Creates ./wheel-export/*.whl, ./wheel-export/requirements/, etc.
#
# After a full bake build, BuildKit cache makes this nearly instant.
target "export-wheel-rocm" {
  inherits = ["_common-rocm"]
  target   = "export_vllm"
  output   = ["type=local,dest=./wheel-export"]
}

target "final-rocm" {
  inherits = ["_common-rocm", "_labels"]
  target   = "final"
  tags     = ["rocm/vllm:latest"]
  output   = ["type=docker"]
}
