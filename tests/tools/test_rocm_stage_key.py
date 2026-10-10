# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Content keys decide when ROCm CI rebuilds its wheelhouse and runtime images.

A key that misses an input ships a stale image; a key that over-reaches turns
an unrelated edit into a multi-hour dependency rebuild.
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[2] / ".buildkite" / "scripts" / "rocm" / "stage_key.py"
_spec = importlib.util.spec_from_file_location("stage_key", _SCRIPT)
stage_key = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage_key)

DOCKERFILE = """\
ARG BASE_IMAGE=ubuntu:22.04
ARG AITER_BRANCH=v1
ARG FA_BRANCH=a
FROM ${BASE_IMAGE} AS base
ARG PYTORCH_ROCM_ARCH
RUN echo sdk ${PYTORCH_ROCM_ARCH}

FROM base AS build_aiter
ARG AITER_BRANCH
RUN echo aiter ${AITER_BRANCH}

FROM base AS build_fa
ARG FA_BRANCH
RUN echo fa \\
    ${FA_BRANCH}

FROM scratch AS wheel_aiter
COPY --from=build_aiter /app/install/ /

FROM scratch AS wheel_fa
COPY --from=build_fa /app/install/ /

FROM base AS runtime
RUN --mount=type=bind,from=wheel_aiter,target=/w true
RUN --mount=type=bind,from=wheel_fa,target=/w true
COPY requirements/rocm.txt /tmp/
"""


@pytest.fixture
def key(tmp_path):
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements" / "rocm.txt").write_text("numpy\n")
    dockerfile = tmp_path / "Dockerfile"

    def compute(target, text=DOCKERFILE, build_args=None, contexts=None):
        dockerfile.write_text(text)
        lines = stage_key.material(
            str(dockerfile), target, build_args or {}, contexts or {}, str(tmp_path)
        )
        return "\n".join(lines)

    compute.root = tmp_path
    return compute


def test_pin_bump_rebuilds_only_that_dependency(key):
    bumped = DOCKERFILE.replace("AITER_BRANCH=v1", "AITER_BRANCH=v2")
    assert key("wheel_aiter") != key("wheel_aiter", bumped)
    assert key("wheel_fa") == key("wheel_fa", bumped)
    assert key("runtime") != key("runtime", bumped)


def test_build_arg_override_and_arch_are_inputs(key):
    assert key("wheel_fa") != key("wheel_fa", build_args={"FA_BRANCH": "b"})
    assert key("wheel_fa") != key(
        "wheel_fa", build_args={"PYTORCH_ROCM_ARCH": "gfx942"}
    )


def test_context_files_are_inputs_but_not_for_unrelated_stages(key):
    before_runtime, before_fa = key("runtime"), key("wheel_fa")
    (key.root / "requirements" / "rocm.txt").write_text("numpy==2\n")
    assert key("runtime") != before_runtime
    assert key("wheel_fa") == before_fa


def test_named_context_replaces_the_stage_subtree(key):
    with_artifacts = {"wheel_aiter": "k1", "wheel_fa": "k2"}
    bumped = DOCKERFILE.replace("AITER_BRANCH=v1", "AITER_BRANCH=v2")
    # With the artifacts supplied, only their keys matter, not how they build.
    assert key("runtime", contexts=with_artifacts) == key(
        "runtime", bumped, contexts=with_artifacts
    )
    assert key("runtime", contexts=with_artifacts) != key(
        "runtime", contexts={**with_artifacts, "wheel_fa": "k3"}
    )


def test_comments_do_not_change_keys(key):
    commented = DOCKERFILE.replace(
        "FROM base AS runtime", "# deployment base\nFROM base AS runtime"
    )
    assert key("runtime") == key("runtime", commented)
