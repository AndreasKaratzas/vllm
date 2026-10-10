# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / ".buildkite" / "scripts" / "docker-build-metadata-args.sh"
ROCM_CI_BAKE = REPO_ROOT / ".buildkite" / "scripts" / "ci-bake-rocm.sh"
ROCM_IMAGE_SMOKE = REPO_ROOT / ".buildkite" / "scripts" / "rocm" / "smoke-test-image.sh"
TORCHCODEC_INSTALL = REPO_ROOT / "tools" / "install_torchcodec_rocm.sh"


def run_helper(
    *args: str,
    env: dict[str, str] | None = None,
    path: str | None = None,
) -> list[str]:
    helper_env = {"PATH": path or os.environ["PATH"]}
    if env:
        helper_env.update(env)
    result = subprocess.run(
        ["bash", str(HELPER), *args],
        check=True,
        env=helper_env,
        stdout=subprocess.PIPE,
        text=True,
    )
    return shlex.split(result.stdout)


def option_values(args: list[str], option: str) -> list[str]:
    return [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == option]


def build_args(args: list[str]) -> dict[str, str]:
    values = {}
    for value in option_values(args, "--build-arg"):
        key, arg_value = value.split("=", 1)
        values[key] = arg_value
    return values


def test_release_metadata_args_prefer_pipeline_id() -> None:
    args = run_helper(
        "cu130-ubuntu2404",
        env={
            "BUILDKITE": "1",
            "BUILDKITE_COMMIT": "abc123",
            "BUILDKITE_PIPELINE_ID": "pipe-uuid",
            "BUILDKITE_PIPELINE_SLUG": "release",
            "BUILDKITE_BUILD_URL": "https://buildkite.example/vllm/builds/1",
            "RELEASE_VERSION": "v0.20.0",
        },
    )

    assert build_args(args) == {
        "VLLM_BUILD_COMMIT": "abc123",
        "VLLM_BUILD_PIPELINE": "pipe-uuid",
        "VLLM_BUILD_URL": "https://buildkite.example/vllm/builds/1",
        "VLLM_IMAGE_TAG": "vllm/vllm-openai:v0.20.0-cu130-ubuntu2404",
    }
    expected_tag = (
        "public.ecr.aws/q9t5s3a7/vllm-release-repo:"
        f"abc123-{os.uname().machine}-cu130-ubuntu2404"
    )
    assert option_values(args, "--tag") == [expected_tag]


def test_nightly_metadata_args_fall_back_to_pipeline_slug() -> None:
    args = run_helper(
        "ubuntu2404",
        env={
            "BUILDKITE": "1",
            "BUILDKITE_COMMIT": "def456",
            "BUILDKITE_PIPELINE_SLUG": "release",
            "BUILDKITE_BUILD_URL": "https://buildkite.example/vllm/builds/2",
            "NIGHTLY": "1",
        },
    )

    assert build_args(args) == {
        "VLLM_BUILD_COMMIT": "def456",
        "VLLM_BUILD_PIPELINE": "release",
        "VLLM_BUILD_URL": "https://buildkite.example/vllm/builds/2",
        "VLLM_IMAGE_TAG": "vllm/vllm-openai:nightly-def456-ubuntu2404",
    }
    expected_tag = (
        "public.ecr.aws/q9t5s3a7/vllm-release-repo:"
        f"def456-{os.uname().machine}-ubuntu2404"
    )
    assert option_values(args, "--tag") == [expected_tag]


def test_local_metadata_args_use_local_overrides() -> None:
    args = run_helper(
        env={
            "VLLM_IMAGE_TAG": "local/test:dev",
            "VLLM_BUILD_COMMIT": "localsha",
            "VLLM_BUILD_PIPELINE": "local-pipeline",
            "VLLM_BUILD_URL": "https://buildkite.example/local",
        },
    )

    assert build_args(args) == {
        "VLLM_BUILD_COMMIT": "localsha",
        "VLLM_BUILD_PIPELINE": "local-pipeline",
        "VLLM_BUILD_URL": "https://buildkite.example/local",
        "VLLM_IMAGE_TAG": "local/test:dev",
    }
    assert option_values(args, "--tag") == ["local/test:dev"]


def test_release_version_lookup_failure_falls_back_to_commit(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    buildkite_agent = fake_bin / "buildkite-agent"
    buildkite_agent.write_text("#!/bin/sh\nexit 1\n")
    buildkite_agent.chmod(0o755)

    args = run_helper(
        "cu129",
        env={
            "BUILDKITE": "1",
            "BUILDKITE_COMMIT": "fallback123",
            "BUILDKITE_PIPELINE_SLUG": "release",
        },
        path=f"{fake_bin}:{os.environ['PATH']}",
    )

    assert build_args(args)["VLLM_IMAGE_TAG"] == ("vllm/vllm-openai:vfallback123-cu129")


def test_vllm_openai_image_embeds_metadata_contract() -> None:
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text()

    for expected in (
        "ARG VLLM_BUILD_COMMIT",
        "ARG VLLM_BUILD_PIPELINE",
        "ARG VLLM_BUILD_URL",
        "ARG VLLM_IMAGE_TAG",
        "VLLM_BUILD_COMMIT=${VLLM_BUILD_COMMIT:-unknown}",
        "VLLM_BUILD_PIPELINE=${VLLM_BUILD_PIPELINE:-local}",
        "VLLM_BUILD_URL=${VLLM_BUILD_URL:-}",
        "VLLM_IMAGE_TAG=${VLLM_IMAGE_TAG:-local/vllm-openai:dev}",
        'ai.vllm.build.commit="${VLLM_BUILD_COMMIT}"',
        'ai.vllm.build.pipeline="${VLLM_BUILD_PIPELINE}"',
        'ai.vllm.build.url="${VLLM_BUILD_URL}"',
        'ai.vllm.image.tag="${VLLM_IMAGE_TAG}"',
    ):
        assert expected in dockerfile


def test_rust_build_cache_excludes_git_metadata() -> None:
    import torch

    from vllm.platforms import current_platform

    dockerfile_names = ["Dockerfile", "Dockerfile.cpu"]
    # CPU jobs can reuse ROCm artifacts, which omit the XPU Dockerfile.
    if not current_platform.is_rocm() and (
        not current_platform.is_cpu() or torch.version.hip is None
    ):
        dockerfile_names.append("Dockerfile.xpu")
    for name in dockerfile_names:
        dockerfile = (REPO_ROOT / "docker" / name).read_text()
        cached_stage, exact_version_stage = dockerfile.split(
            "FROM rust-build-cache AS rust-build", maxsplit=1
        )
        exact_version_stage = exact_version_stage.split("\nFROM ", maxsplit=1)[0]
        cached_run = cached_stage.rsplit("RUN ", maxsplit=1)[1]

        assert 'SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0+docker.cache"' in cached_run
        assert "source=.git,target=.git" not in cached_run
        assert "source=.git,target=.git" in exact_version_stage
        assert 'SETUPTOOLS_SCM_PRETEND_METADATA="{dirty=false}"' in exact_version_stage
        assert "bash tools/build_rust.sh" in exact_version_stage


def test_rocm_ci_base_bake_embeds_content_hash_label() -> None:
    bake_file = (REPO_ROOT / "docker" / "docker-bake-rocm.hcl").read_text()

    for expected in (
        'variable "CI_BASE_CONTENT_HASH"',
        'target "ci-base-rocm"',
        'target   = "ci_base"',
        '"vllm.ci_base.content_hash" = CI_BASE_CONTENT_HASH',
    ):
        assert expected in bake_file


def test_rocm_ci_base_metadata_inputs_cover_ci_base_files() -> None:
    ci_bake = ROCM_CI_BAKE.read_text()

    for expected in (
        "requirements/common.txt",
        "requirements/rocm.txt",
        "requirements/test/rocm.txt",
        "docker/Dockerfile.rocm",
    ):
        assert expected in ci_bake


def test_torchcodec_cache_keeps_wheel_tags_and_invalidates_build_abi(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    scripts = {
        "python3": """#!/bin/bash
if [[ "$*" == *"torchcodec.decoders"* ]]; then
    test -f "$FAKE_INSTALLED"
elif [[ "$*" == *"pybind11.get_cmake_dir"* ]]; then
    echo /mock/pybind11
else
    cat >/dev/null
    echo "$FAKE_ABI"
fi
""",
        "pkg-config": """#!/bin/bash
if [ "$1" = --modversion ]; then echo 60.1; fi
""",
        "git": """#!/bin/bash
if [ "$1" = init ]; then mkdir -p "${@: -1}"; fi
""",
        "pip": """#!/bin/bash
if [ "$1" = wheel ]; then
    mkdir -p "${@: -1}"
    touch "${@: -1}/$FAKE_WHEEL_NAME"
    echo build >> "$FAKE_BUILDS"
elif [[ "${@: -1}" == *.whl ]]; then
    test "$(basename "${@: -1}")" = "$FAKE_WHEEL_NAME" || exit 1
    touch "$FAKE_INSTALLED"
fi
""",
    }
    for name, content in scripts.items():
        executable = fake_bin / name
        executable.write_text(content)
        executable.chmod(0o755)
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TORCHCODEC_WHEEL_CACHE": str(tmp_path / "cache"),
        "TORCHCODEC_REPO": "https://github.com/pytorch/torchcodec.git",
        "TORCHCODEC_BRANCH": "v0.10.0",
        "TORCHCODEC_COMMIT": "0b261b98080925f2b709712a5491a1e8dd817065",
        "PYTORCH_ROCM_ARCH": "gfx942;gfx950",
        "FAKE_ABI": "cp312-torch2.12-rocm10",
        "FAKE_INSTALLED": str(tmp_path / "installed"),
        "FAKE_BUILDS": str(tmp_path / "builds"),
        "FAKE_WHEEL_NAME": "torchcodec-0.10.0-cp312-cp312-linux_x86_64.whl",
    }

    def install() -> str:
        (tmp_path / "installed").unlink(missing_ok=True)
        return subprocess.run(
            ["bash", str(TORCHCODEC_INSTALL)],
            env=env,
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    install()
    assert "Installed from cached wheel." in install()
    assert (tmp_path / "builds").read_text().splitlines() == ["build"]
    env["FAKE_ABI"] = "cp312-torch2.13-rocm10"
    install()
    env["TORCHCODEC_COMMIT"] = "1" * 40
    install()
    assert (tmp_path / "builds").read_text().splitlines() == ["build"] * 3
    assert len(list((tmp_path / "cache").glob(f"*/{env['FAKE_WHEEL_NAME']}"))) == 3


def test_rocm_ci_smoke_runs_in_shared_buildkit_graph() -> None:
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile.rocm").read_text()
    ci_hcl = (REPO_ROOT / "docker" / "ci-rocm.hcl").read_text()
    full_image_group = ci_hcl.split('group "test-rocm-ci-with-wheel"', maxsplit=1)[
        1
    ].split("}", maxsplit=1)[0]

    for expected in (
        "FROM test AS test_smoke",
        "smoke-test-image.sh --inside",
        "FROM scratch AS export_test_smoke",
        'target "smoke-test-rocm-ci"',
        'target     = "export_test_smoke"',
        'output     = ["type=local,dest=./build/rocm-smoke-export"]',
    ):
        assert expected in dockerfile or expected in ci_hcl
    assert '"smoke-test-rocm-ci"' in full_image_group
    assert 'target "smoke-test-rocm-ci"' in ROCM_CI_BAKE.read_text()


def prepare_rocm_smoke_test(
    tmp_path: Path,
    *,
    marker_id: str,
    build_id: str,
) -> tuple[dict[str, str], Path, Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_called = tmp_path / "docker-called"
    docker = fake_bin / "docker"
    docker.write_text('#!/bin/sh\ntouch "$FAKE_DOCKER_CALLED"\nexit 99\n')
    docker.chmod(0o755)

    marker = tmp_path / "build" / "rocm-smoke-export" / "vllm-smoke-ok"
    marker.parent.mkdir(parents=True)
    marker.write_text(f"{marker_id}\n")
    env = os.environ.copy()
    env.update(
        {
            "BUILDKITE_BUILD_ID": build_id,
            "FAKE_DOCKER_CALLED": str(docker_called),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )
    env.pop("ROCM_CI_ARTIFACT_ONLY", None)
    env.pop("VLLM_CI_SMOKE_IMAGE", None)
    return env, marker, docker, docker_called


def test_rocm_smoke_marker_avoids_host_image_pull(tmp_path: Path) -> None:
    env, marker, _, docker_called = prepare_rocm_smoke_test(
        tmp_path,
        marker_id="build-123",
        build_id="build-123",
    )

    result = subprocess.run(
        ["bash", str(ROCM_IMAGE_SMOKE)],
        check=True,
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert "verified inside BuildKit" in result.stdout
    assert not docker_called.exists()
    assert not marker.exists()


def test_rocm_smoke_rejects_marker_from_another_build(tmp_path: Path) -> None:
    env, marker, _, docker_called = prepare_rocm_smoke_test(
        tmp_path,
        marker_id="previous-build",
        build_id="current-build",
    )

    result = subprocess.run(
        ["bash", str(ROCM_IMAGE_SMOKE)],
        check=False,
        cwd=tmp_path,
        env=env,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 1
    assert "previous-build, not current-build" in result.stderr
    assert not docker_called.exists()
    assert marker.exists()


def test_rocm_smoke_override_streams_current_checks_to_docker(
    tmp_path: Path,
) -> None:
    env, marker, docker, _ = prepare_rocm_smoke_test(
        tmp_path,
        marker_id="build-123",
        build_id="build-123",
    )
    docker_args = tmp_path / "docker-args"
    docker_stdin = tmp_path / "docker-stdin"
    docker.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$FAKE_DOCKER_ARGS"\n'
        'cat > "$FAKE_DOCKER_STDIN"\n'
    )
    env.update(
        {
            "FAKE_DOCKER_ARGS": str(docker_args),
            "FAKE_DOCKER_STDIN": str(docker_stdin),
            "IMAGE_TAG": "rocm/vllm-ci:built",
            "VLLM_CI_SMOKE_IMAGE": "rocm/vllm-ci:override",
        }
    )

    subprocess.run(
        ["bash", str(ROCM_IMAGE_SMOKE)],
        check=True,
        cwd=tmp_path,
        env=env,
    )

    assert "rocm/vllm-ci:override" in docker_args.read_text().splitlines()
    assert docker_args.read_text().splitlines()[-3:] == ["-s", "--", "--inside"]
    assert "run_smoke_checks()" in docker_stdin.read_text()
    assert marker.exists()


def test_rocm_git_fetch_disables_automatic_maintenance(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    git = fake_bin / "git"
    git.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    git.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; git_fetch_with_timeout --quiet origin HEAD',
            "bash",
            str(ROCM_CI_BAKE),
        ],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert result.stdout.splitlines() == [
        "fetch",
        "--no-auto-maintenance",
        "--quiet",
        "origin",
        "HEAD",
    ]


LOCAL_RUNTIME = REPO_ROOT / "tools/vllm-rocm/local_runtime.py"


@pytest.mark.parametrize(
    ("filename", "section", "image_arg", "unrelated"),
    [
        ("Dockerfile.rocm_base", "runtime", "OS_IMAGE", "dependency-producer"),
        ("Dockerfile.rocm", "commit-build", "RUNTIME_IMAGE", "deployment"),
    ],
)
def test_rocm_shared_recipe_preserves_unrelated_target_cache(
    tmp_path: Path, filename: str, section: str, image_arg: str, unrelated: str
) -> None:
    digest = runpy.run_path(str(LOCAL_RUNTIME))["recipe_digest"]
    recipe = tmp_path / filename
    source = (REPO_ROOT / "docker" / filename).read_text()
    recipe.write_text(source)
    original = digest(recipe, section, image_arg)

    other_end = f"# END vllm-rocm {unrelated}"
    recipe.write_text(source.replace(other_end, f"ENV CACHE_PROBE=1\n{other_end}"))
    assert digest(recipe, section, image_arg) == original

    own_end = f"# END vllm-rocm {section}"
    recipe.write_text(source.replace(own_end, f"ENV CACHE_PROBE=1\n{own_end}"))
    assert digest(recipe, section, image_arg) != original

    recipe.write_text(
        source.replace(f"ARG {image_arg}=scratch", f"ARG {image_arg}=changed")
    )
    assert digest(recipe, section, image_arg) != original
    recipe.write_text(source.replace("dockerfile:1.7@", "dockerfile:1.8@"))
    assert digest(recipe, section, image_arg) != original


def run_local_runtime(
    *args: object, success: bool = True
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, str(LOCAL_RUNTIME), *(str(arg) for arg in args)],
        capture_output=True,
        text=True,
    )
    if success:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
    return result


def make_runtime_wheel(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}-1.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(f"{name}/__init__.py", "value = 1\n")
        wheel.writestr(
            f"{name}-1.0.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n",
        )
    return path


def freeze_runtime_fixture(tmp_path: Path) -> tuple[Path, Path]:
    directory = tmp_path / "runtime"
    make_runtime_wheel(directory, "rocm_sample")
    platform = tmp_path / "platform"
    (platform / "opt").mkdir(parents=True)
    (platform / "etc").mkdir()
    run_local_runtime(
        "freeze",
        "--kind",
        "runtime",
        "--wheelhouse",
        directory,
        "--platform",
        platform,
        "--os-image",
        f"os@sha256:{'a' * 64}",
        "--arches",
        "gfx942;gfx950",
        "--epoch",
        0,
    )
    return directory / "artifact-lock.json", platform


def test_runtime_lock_survives_mtime_and_ci_identity_but_tracks_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CI build UUIDs and file mtimes must not invalidate a dependency artifact."""
    lock, platform = freeze_runtime_fixture(tmp_path)
    original = json.loads(lock.read_text())["key"]
    wheel = next(lock.parent.glob("*.whl"))
    os.utime(wheel, (1, 1))
    monkeypatch.setenv("BUILDKITE_BUILD_ID", "a-different-build")
    assert run_local_runtime("verify", lock).stdout.strip() == original
    provenance = tmp_path / "provenance.json"
    for revision in ("ci-code-before", "ci-code-after"):
        provenance.write_text(json.dumps({"unrelated_ci_recipe": revision}))
        run_local_runtime(
            "freeze",
            "--kind",
            "runtime",
            "--wheelhouse",
            lock.parent,
            "--platform",
            platform,
            "--os-image",
            f"os@sha256:{'a' * 64}",
            "--arches",
            "gfx942;gfx950",
            "--epoch",
            0,
            "--provenance",
            provenance,
        )
        assert json.loads(lock.read_text())["key"] == original
    with wheel.open("ab") as stream:
        stream.write(b"changed")
    assert (
        "changed since freezing"
        in run_local_runtime(
            "verify",
            lock,
            success=False,
        ).stderr
    )


def test_runtime_rejects_platform_mutation_and_test_dependency_replacement(
    tmp_path: Path,
) -> None:
    lock, platform = freeze_runtime_fixture(tmp_path)
    (platform / "etc" / "patch.conf").write_text("changed patch")
    assert (
        "sidecar changed"
        in run_local_runtime(
            "verify",
            lock,
            "--platform",
            platform,
            success=False,
        ).stderr
    )
    tests = tmp_path / "test-tools"
    make_runtime_wheel(tests, "rocm_sample")
    assert (
        "must not replace runtime"
        in run_local_runtime(
            "freeze",
            "--kind",
            "test",
            "--wheelhouse",
            tests,
            "--runtime-lock",
            lock,
            "--epoch",
            0,
            success=False,
        ).stderr
    )


def test_local_assembly_reuses_parent_encoding_and_has_no_publish_option(
    tmp_path: Path,
) -> None:
    lock, _ = freeze_runtime_fixture(tmp_path)
    wheelhouse = tmp_path / "wheel"
    make_runtime_wheel(wheelhouse, "vllm")
    run_local_runtime(
        "freeze",
        "--kind",
        "wheel",
        "--wheelhouse",
        wheelhouse,
        "--runtime-lock",
        lock,
        "--epoch",
        1,
    )
    args = shlex.split(
        run_local_runtime(
            "assemble",
            "--lock",
            wheelhouse / "artifact-lock.json",
            "--runtime-image",
            f"runtime@sha256:{'b' * 64}",
            "--output",
            tmp_path / "image.oci.tar",
        ).stdout
    )
    assert "--push" not in args
    assert option_values(args, "--target") == ["deployment"]
    exporter = option_values(args, "--output")[0]
    assert exporter.startswith("type=oci,")
    assert "compression=zstd" in exporter and "force-compression=false" in exporter
    assert build_args(args)["DEPENDENCY_KEY"] == json.loads(lock.read_text())["key"]
    assert "--network=none" in args
    assert "vllm-rocm-local" in args
    rejected = run_local_runtime(
        "assemble",
        "--lock",
        wheelhouse / "artifact-lock.json",
        "--runtime-image",
        f"runtime@sha256:{'b' * 64}",
        "--output",
        tmp_path / "unscanned.oci",
        "--execute",
        success=False,
    )
    assert "requires an executable local --gitleaks" in rejected.stderr
    assert not (tmp_path / "unscanned.oci").exists()


def test_runtime_rejects_floating_images_and_mismatched_wheel_identity(
    tmp_path: Path,
) -> None:
    lock, platform = freeze_runtime_fixture(tmp_path)
    assert (
        "must be repo@sha256"
        in run_local_runtime(
            "freeze",
            "--kind",
            "runtime",
            "--wheelhouse",
            lock.parent,
            "--platform",
            platform,
            "--os-image",
            "os:latest",
            "--arches",
            "gfx942",
            "--epoch",
            0,
            success=False,
        ).stderr
    )
    wheel = next(lock.parent.glob("*.whl"))
    wheel.rename(wheel.parent / "other-1.0-py3-none-any.whl")
    assert (
        "filename and METADATA disagree"
        in run_local_runtime(
            "verify",
            lock,
            success=False,
        ).stderr
    )


def test_test_overlay_cannot_shadow_runtime_under_another_distribution_name(
    tmp_path: Path,
) -> None:
    lock, _ = freeze_runtime_fixture(tmp_path)
    tools = tmp_path / "tools"
    wheel = make_runtime_wheel(tools, "unrelated_distribution")
    with zipfile.ZipFile(wheel, "a") as package:
        package.writestr("rocm_sample/__init__.py", "value = 'shadowed'\n")
    result = run_local_runtime(
        "freeze",
        "--kind",
        "test",
        "--wheelhouse",
        tools,
        "--runtime-lock",
        lock,
        "--epoch",
        0,
        success=False,
    )
    assert "must not replace runtime" in result.stderr


def test_oci_layout_can_feed_local_assembly_and_rejects_wrong_runtime(
    tmp_path: Path,
) -> None:
    """A local runtime must feed the next build without a registry publication."""
    lock, _ = freeze_runtime_fixture(tmp_path)
    dependency_key = json.loads(lock.read_text())["key"]
    layout = tmp_path / "runtime.oci"
    blobs = layout / "blobs/sha256"
    blobs.mkdir(parents=True)
    (layout / "oci-layout").write_text('{"imageLayoutVersion": "1.0.0"}')

    def write_blob(value: dict, media_type: str) -> dict:
        data = json.dumps(value).encode()
        digest = hashlib.sha256(data).hexdigest()
        (blobs / digest).write_bytes(data)
        return dict(mediaType=media_type, digest=f"sha256:{digest}", size=len(data))

    def runtime_index(key: str) -> None:
        config = write_blob(
            {
                "os": "linux",
                "architecture": "amd64",
                "config": {
                    "Labels": {"vllm.rocm.dependency_key": key},
                },
                "rootfs": {"type": "layers", "diff_ids": []},
            },
            "application/vnd.oci.image.config.v1+json",
        )
        manifest = write_blob(
            {"schemaVersion": 2, "config": config, "layers": []},
            "application/vnd.oci.image.manifest.v1+json",
        )
        (layout / "index.json").write_text(
            json.dumps(
                {
                    "schemaVersion": 2,
                    "manifests": [manifest],
                }
            )
        )

    runtime_index(dependency_key)
    wheelhouse = tmp_path / "wheel"
    make_runtime_wheel(wheelhouse, "vllm")
    run_local_runtime(
        "freeze",
        "--kind",
        "wheel",
        "--wheelhouse",
        wheelhouse,
        "--runtime-lock",
        lock,
        "--epoch",
        0,
    )
    args = (
        "assemble",
        "--lock",
        wheelhouse / "artifact-lock.json",
        "--runtime-layout",
        layout,
        "--output",
        tmp_path / "deployment.oci",
    )
    command = shlex.split(run_local_runtime(*args).stdout)
    assert build_args(command)["RUNTIME_IMAGE"] == "locked_runtime"
    assert any(
        context.startswith("locked_runtime=oci-layout://")
        for context in option_values(command, "--build-context")
    )
    runtime_index("a-different-dependency-artifact")
    assert (
        "different dependency lock"
        in run_local_runtime(
            *args,
            success=False,
        ).stderr
    )


def test_runtime_handoff_keeps_cached_tools_and_excludes_checkout_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CI must get verified wheel inputs without a checkout vLLM on sys.path."""
    runtime, _ = freeze_runtime_fixture(tmp_path)
    wheel_dir, tools_dir = tmp_path / "wheel", tmp_path / "tools"
    make_runtime_wheel(wheel_dir, "vllm")
    make_runtime_wheel(tools_dir, "pytest_fixture")
    for kind, directory in (("wheel", wheel_dir), ("test", tools_dir)):
        run_local_runtime(
            "freeze",
            "--kind",
            kind,
            "--wheelhouse",
            directory,
            "--runtime-lock",
            runtime,
            *(("--source-commit", "a" * 40) if kind == "wheel" else ()),
            "--epoch",
            0,
        )
    checkout = tmp_path / "checkout"
    for name in ("tests/test_sample.py", "vllm/__init__.py"):
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value = 1\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        '#!/bin/sh\nif [ "$3" = rev-parse ]; then\n'
        f"echo {'a' * 40}\nelse\n"
        "printf 'tests/test_sample.py\\0vllm/__init__.py\\0'\nfi\n"
    )
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")

    def handoff(build_id: str) -> dict:
        output = tmp_path / build_id
        run_local_runtime(
            "pack",
            "--runtime-lock",
            runtime,
            "--runtime-image",
            f"runtime@sha256:{'b' * 64}",
            "--wheel-lock",
            wheel_dir / "artifact-lock.json",
            "--test-lock",
            tools_dir / "artifact-lock.json",
            "--checkout",
            checkout,
            "--producer-step",
            "wheel-producer",
            "--build-id",
            build_id,
            "--output",
            output,
        )
        return json.loads((output / "handoff.json").read_text())

    first, second = handoff("build-one"), handoff("build-two")
    assert first["test_tools_sha256"] == second["test_tools_sha256"]
    assert first["wheel_bundle_sha256"] == second["wheel_bundle_sha256"]
    with tarfile.open(tmp_path / "build-one/source.tar.gz") as bundle:
        assert bundle.getnames() == ["tests/test_sample.py", "source-identity.json"]
    with tarfile.open(tmp_path / "build-one/wheel.tar") as bundle:
        assert "artifact-lock.json" in bundle.getnames()
        assert "vllm-1.0-py3-none-any.whl" in bundle.getnames()

    worker = runpy.run_path(
        str(REPO_ROOT / ".buildkite/scripts/rocm/runtime_worker.py")
    )
    prefix, cache = tmp_path / "prefix", tmp_path / "cache"
    prefix.mkdir()
    (prefix / ".vllm-runtime-key").write_text(first["runtime_key"])
    installs = []
    fail_tools = False

    def offline_install(command: list[str], **kwargs) -> None:
        nonlocal fail_tools
        installs.append(command)
        assert option_values(command, "--runtime-key") == [first["runtime_key"]]
        target = option_values(command, "--target")
        if target:
            directory = Path(target[0])
            directory.mkdir(parents=True)
            if fail_tools:
                fail_tools = False
                raise subprocess.CalledProcessError(1, command)
            (directory / ".vllm-test-tools-key").write_text(first["test_tools_key"])
        else:
            (prefix / ".vllm-wheel-sha256").write_text(first["wheel_sha256"])

    monkeypatch.delenv("BUILDKITE_BUILD_ID", raising=False)
    monkeypatch.delenv("BUILDKITE_COMMIT", raising=False)
    monkeypatch.setattr(subprocess, "run", offline_install)
    for build_id, value in (("build-one", first), ("build-two", second)):
        workspace = tmp_path / f"{build_id}-workspace"
        tools = worker["prepare"](
            value,
            tmp_path / build_id,
            cache,
            workspace,
            Path("/opt/vllm-rocm/install_artifacts.py"),
            prefix,
        )
        assert (workspace / "tests/test_sample.py").is_file()
        assert not (workspace / "vllm").exists()
        assert (tools / ".vllm-test-tools-key").read_text() == first["test_tools_key"]
    assert len(installs) == 2, "the warm job must reuse verified wheel/tools installs"
    assert len(list((cache / "bundles").iterdir())) == 3
    recovery_cache = tmp_path / "recovery-cache"
    fail_tools = True
    arguments = (
        first,
        tmp_path / "build-one",
        recovery_cache,
        tmp_path / "recovery-workspace",
        Path("/opt/vllm-rocm/install_artifacts.py"),
        prefix,
    )
    with pytest.raises(subprocess.CalledProcessError):
        worker["prepare"](*arguments)
    assert not list((recovery_cache / "test-tools").iterdir())
    assert worker["prepare"](*arguments).is_dir()
    cached_lock = (
        cache / "bundles" / first["wheel_bundle_sha256"] / "artifact-lock.json"
    )
    original_lock = cached_lock.read_text()
    value = json.loads(original_lock) | {"source_commit": "b" * 40}
    cached_lock.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="commit wheel source"):
        worker["prepare"](
            first,
            tmp_path / "build-one",
            cache,
            tmp_path / "different-source",
            Path("/opt/vllm-rocm/install_artifacts.py"),
            prefix,
        )
    cached_lock.write_text(original_lock)
    with (tmp_path / "build-two/wheel.tar").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        worker["prepare"](
            second,
            tmp_path / "build-two",
            cache,
            tmp_path / "third-workspace",
            Path("/opt/vllm-rocm/install_artifacts.py"),
            prefix,
        )


def test_native_runtime_launch_cleans_private_workspace_and_returns_test_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = runpy.run_path(
        str(REPO_ROOT / ".buildkite/scripts/rocm/runtime_worker.py")
    )
    workspaces = []

    def child(command: list[str], *, env: dict[str, str], check: bool):
        workspace = Path(env["VLLM_CI_WORKSPACE"])
        assert workspace.is_dir() and not list(workspace.iterdir())
        (workspace / "test-output").write_text("temporary")
        workspaces.append(workspace)
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(subprocess, "run", child)
    assert worker["launch"]([], {"ROCM_CI_EXECUTION": "native"}) == 7
    assert workspaces and not workspaces[0].exists()


def test_dind_runtime_launch_rejects_floating_image_before_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = runpy.run_path(
        str(REPO_ROOT / ".buildkite/scripts/rocm/runtime_worker.py")
    )

    def unexpected_docker(*args, **kwargs):
        pytest.fail("a floating image must not reach Docker")

    monkeypatch.setattr(subprocess, "run", unexpected_docker)
    with pytest.raises(ValueError, match="pinned"):
        worker["launch"](
            [], {"ROCM_CI_EXECUTION": "dind", "ROCM_RUNTIME_IMAGE": "runtime:latest"}
        )


def test_runtime_pipeline_pins_native_and_dind_preserving_scheduler_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delayed rendering must select one digest while retaining scheduler behavior."""
    directory = REPO_ROOT / ".buildkite/scripts/rocm"
    monkeypatch.syspath_prepend(str(directory))
    render = runpy.run_path(str(directory / "runtime-pipeline.py"))["render"]
    handoff = {
        "schema": 1,
        "build_id": "local-build",
        "commit": "a" * 40,
        "producer_step": "wheel-producer",
        "runtime_image": f"runtime@sha256:{'b' * 64}",
        "runtime_key": "c" * 64,
        "test_tools_runtime_key": "c" * 64,
        "wheel_path": "wheel.tar",
        "test_tools_path": "test-tools.tar",
        "source_path": "source.tar.gz",
        **{
            name: "d" * 64
            for name in (
                "wheel_sha256",
                "wheel_bundle_sha256",
                "test_tools_key",
                "test_tools_sha256",
                "source_sha256",
            )
        },
    }
    guard = 'build.branch == "main"'
    native: dict = {
        "key": "amd-native",
        "agents": {"queue": "amd_mi300_1"},
        "depends_on": ["image-build-amd"],
        "commands": ["bash .buildkite/scripts/hardware_ci/run-amd-test.sh"],
        "env": {"VLLM_TEST_COMMANDS": "pytest -v test_example.py"},
        "if": guard,
        "concurrency": 1,
        "concurrency_group": "amd-example",
        "plugins": [
            {
                "kubernetes": {
                    "podSpecPatch": {
                        "containers": [{"name": "container-0", "image": "old:ci_base"}]
                    }
                }
            }
        ],
    }
    dind = {**native, "key": "amd-dind", "plugins": []}
    selected = {
        "steps": [
            native,
            dind,
            {
                "key": "refresh-rocm-base-amd",
                "agents": {"queue": "amd_cpu"},
                "commands": ["pip install old-build-tools"],
            },
        ]
    }
    pipeline, report = render(
        selected, handoff, tmp_path / "artifacts", tmp_path / "cache"
    )
    workers = pipeline["steps"][0]["steps"]
    assert report["supported"] == ["amd-native", "amd-dind"]
    assert not report["unsupported"] and not report["uploads_performed"]
    for worker in workers:
        assert worker["depends_on"] == ["wheel-producer"]
        assert worker["if"] == guard and worker["concurrency"] == 1
        assert worker["env"]["ROCM_RUNTIME_IMAGE"] == handoff["runtime_image"]
    container = workers[0]["plugins"][0]["kubernetes"]["podSpecPatch"]["containers"][0]
    assert container["image"] == handoff["runtime_image"]
    assert workers[1]["env"]["DOCKER_IMAGE_NAME"] == handoff["runtime_image"]
    native["env"]["VLLM_TEST_COMMANDS"] = "pip install changed-framework && pytest"
    pipeline, report = render(
        selected, handoff, tmp_path / "artifacts", tmp_path / "cache"
    )
    assert not pipeline["steps"]
    assert report["unsupported"][0]["key"] == "amd-native"
    handoff["runtime_image"] = "runtime:latest"
    with pytest.raises(ValueError, match="pinned"):
        render(selected, handoff, tmp_path / "artifacts", tmp_path / "cache")


@pytest.mark.parametrize("member_type", [tarfile.REGTYPE, tarfile.SYMTYPE])
def test_runtime_worker_rejects_archive_escape_before_staging(
    tmp_path: Path, member_type: bytes
) -> None:
    worker = runpy.run_path(
        str(REPO_ROOT / ".buildkite/scripts/rocm/runtime_worker.py")
    )
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as bundle:
        member = tarfile.TarInfo(
            "../escape" if member_type == tarfile.REGTYPE else "link"
        )
        member.type, member.linkname = member_type, "/outside"
        bundle.addfile(member)
    with pytest.raises(ValueError, match="unsafe artifact member"):
        worker["unpack_cached"](
            archive, tmp_path / "cache", worker["file_sha256"](archive)
        )
    assert not (tmp_path / "escape").exists()


def test_runtime_payload_groups_preserve_large_blobs_when_python_changes(
    tmp_path: Path,
) -> None:
    """A small runtime dependency update must preserve SDK/PyTorch payloads."""
    prefix = tmp_path / "venv"
    site = prefix / "lib/python3.12/site-packages"
    files = {
        "_rocm_sdk_core/lib/libhip.so": b"sdk runtime",
        "torch/__init__.py": b"torch runtime",
        "triton/__init__.py": b"triton runtime",
        "yaml/__init__.py": b"small dependency",
    }
    for name, data in files.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    alias = site / "_rocm_sdk_core/lib/libhip-alias.so"
    os.link(site / "_rocm_sdk_core/lib/libhip.so", alias)
    (prefix / "lib64").symlink_to("lib", target_is_directory=True)
    split = runpy.run_path(str(REPO_ROOT / "tools/vllm-rocm/split_runtime.py"))["split"]

    def contents(directory: Path) -> dict:
        return {
            str(path.relative_to(directory)): (
                os.readlink(path)
                if path.is_symlink()
                else hashlib.sha256(path.read_bytes()).hexdigest()
            )
            for path in directory.rglob("*")
            if path.is_symlink() or path.is_file()
        }

    original = contents(prefix)
    first = tmp_path / "first"
    split(prefix, first)
    sdk = first / "sdk/lib/python3.12/site-packages/_rocm_sdk_core/lib"
    assert (sdk / "libhip.so").stat().st_ino == (sdk / "libhip-alias.so").stat().st_ino
    for group in ("sdk", "torch", "triton", "python"):
        shutil.copytree(first / group, prefix, dirs_exist_ok=True, symlinks=True)
    assert contents(prefix) == original
    (site / "yaml/__init__.py").write_bytes(b"new small dependency")
    second = tmp_path / "second"
    split(prefix, second)
    for group in ("sdk", "torch", "triton"):
        assert contents(first / group) == contents(second / group)
    assert contents(first / "python") != contents(second / "python")
