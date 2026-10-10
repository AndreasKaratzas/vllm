# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install locked local artifacts and test the installed serving deployment."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import string
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

ALNUM = string.ascii_letters + string.digits
ARTIFACTS = (
    ("wheel_path", "wheel_bundle_sha256"),
    ("test_tools_path", "test_tools_sha256"),
    ("source_path", "source_sha256"),
)
FORWARDED_ENV = (
    "ROCM_CI_HANDOFF",
    "ROCM_ARTIFACT_ROOT",
    "ROCM_CACHE_ROOT",
    "VLLM_TEST_COMMANDS",
    "BUILDKITE_COMMIT",
    "BUILDKITE_BUILD_ID",
    "BUILDKITE_JOB_ID",
    "BUILDKITE_PARALLEL_JOB",
    "BUILDKITE_PARALLEL_JOB_COUNT",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "HF_HOME",
    "HF_HUB_OFFLINE",
)


def is_hex(value: str, length: int = 64) -> bool:
    return len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def is_digest(value: str) -> bool:
    repository, separator, digest = value.partition("@sha256:")
    return (
        bool(separator)
        and bool(repository)
        and repository[0] in ALNUM
        and all(character in ALNUM + "._:/-" for character in repository)
        and is_hex(digest)
    )


def is_env_name(value: str) -> bool:
    return (
        bool(value)
        and value[0] in string.ascii_letters + "_"
        and all(character in ALNUM + "_" for character in value)
    )


def launch(arguments: list[str], env: dict[str, str]) -> int:
    """Launch the same worker in native Kubernetes or a pinned Docker runtime."""
    execution = env.get("ROCM_CI_EXECUTION", "native")
    script = Path(__file__).resolve()
    if execution not in {"native", "dind"}:
        raise ValueError("ROCM_CI_EXECUTION must be native or dind")
    with tempfile.TemporaryDirectory(prefix="vllm-deployment-workspace-") as temporary:
        if execution == "native":
            child_env = env | {"VLLM_CI_WORKSPACE": temporary}
            return subprocess.run(
                [
                    env.get("ROCM_RUNTIME_PYTHON", "/opt/venv/bin/python"),
                    str(script),
                    *arguments,
                ],
                env=child_env,
                check=False,
            ).returncode
        image = env["ROCM_RUNTIME_IMAGE"]
        if not is_digest(image):
            raise ValueError("ROCM_RUNTIME_IMAGE must be pinned by repository digest")
        checkout = Path(env.get("BUILDKITE_BUILD_CHECKOUT_PATH", os.getcwd())).resolve()
        artifacts = Path(env["ROCM_ARTIFACT_ROOT"])
        cache = Path(env["ROCM_CACHE_ROOT"])
        for path in (checkout, artifacts, cache):
            if not path.is_absolute() or ":" in str(path) or "," in str(path):
                raise ValueError("worker mounts must be absolute local paths")
        cache.mkdir(parents=True, exist_ok=True)
        command = [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/opt/venv/bin/python",
            "--device",
            "/dev/kfd",
            "--device",
            "/dev/dri",
            "--group-add",
            "video",
            "--ipc",
            "host",
            "--cap-add",
            "IPC_LOCK",
            "-v",
            f"{checkout}:/ci-source:ro",
            "-v",
            f"{artifacts}:{artifacts}:ro",
            "-v",
            f"{cache}:{cache}",
            "-v",
            f"{temporary}:/vllm-workspace",
        ]
        names = {*FORWARDED_ENV, *env.get("ROCM_CI_ENV_NAMES", "").split()}
        for variable in sorted(names):
            if not is_env_name(variable):
                raise ValueError("invalid forwarded environment variable name")
            if variable in env:
                command += ["-e", variable]
        if env.get("HF_HOME") and Path(env["HF_HOME"]).is_dir():
            home = Path(env["HF_HOME"])
            if not home.is_absolute() or ":" in str(home) or "," in str(home):
                raise ValueError("HF_HOME must be an absolute local path")
            command += ["-v", f"{home}:{home}"]
        command += [
            image,
            "/ci-source/.buildkite/scripts/rocm/runtime_worker.py",
            *arguments,
        ]
        return subprocess.run(command, env=env, check=False).returncode


def check_deployment(workspace: Path) -> None:
    """Reject a source-shadowed or mismatched installed deployment before tests."""
    if (workspace / "vllm").exists() or (workspace / "vllm.py").exists():
        raise ValueError("test workspace must not contain root vLLM sources")
    prefix = Path(sys.prefix).resolve()
    spec = importlib.util.find_spec("vllm")
    if (
        not spec
        or not spec.origin
        or not Path(spec.origin).resolve().is_relative_to(prefix)
    ):
        raise ValueError("installed vLLM is missing or source-shadowed")
    for variable, marker in (
        ("ROCM_RUNTIME_KEY", ".vllm-runtime-key"),
        ("ROCM_WHEEL_SHA256", ".vllm-wheel-sha256"),
    ):
        expected = os.environ[variable]
        if not is_hex(expected) or (prefix / marker).read_text().strip() != expected:
            raise ValueError(f"{variable} mismatch")


def run_tests(
    workspace: Path,
    tools: Path,
    prefix: Path,
    handoff: dict,
    test_commands: str,
) -> int:
    """Preflight and execute tests against the installed artifact in isolation."""
    if (workspace / "vllm").exists() or (workspace / "vllm.py").exists():
        raise ValueError("test workspace must not contain root vLLM sources")
    python = prefix / "bin/python"
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env.update(
        ROCM_RUNTIME_KEY=handoff["runtime_key"],
        ROCM_WHEEL_SHA256=handoff["wheel_sha256"],
        ROCM_TEST_TOOLS_PATH=str(tools),
        VLLM_CI_WORKSPACE=str(workspace),
        PYTHONPATH=f"{tools}:{workspace}",
        PATH=f"{tools}/bin:{python.parent}:{env.get('PATH', '')}",
        PIP_NO_INDEX="1",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    subprocess.run(
        [
            str(python),
            str(Path(__file__).resolve()),
            "--check-deployment",
            "--workspace",
            str(workspace),
        ],
        env=env,
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        [str(python), "-m", "pip", "check"], env=env, cwd=workspace, check=True
    )
    return subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            test_commands.replace("/vllm-workspace", str(workspace)),
        ],
        env=env,
        cwd=workspace / "tests",
        check=False,
    ).returncode


def validate_handoff(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("runtime handoff must be an object")
    if value.get("schema") != 1:
        raise ValueError("unsupported runtime handoff schema")
    for field in (
        "runtime_image",
        "runtime_key",
        "wheel_sha256",
        "wheel_bundle_sha256",
        "test_tools_key",
        "test_tools_runtime_key",
        "test_tools_sha256",
        "source_sha256",
        "commit",
        "build_id",
        "producer_step",
        *(path for path, _ in ARTIFACTS),
    ):
        if not isinstance(value.get(field), str):
            raise ValueError(f"{field} must be a string")
    if not is_digest(value.get("runtime_image", "")):
        raise ValueError("runtime_image must be pinned by repository digest")
    for field in (
        "runtime_key",
        "wheel_sha256",
        "wheel_bundle_sha256",
        "test_tools_key",
        "test_tools_runtime_key",
        "test_tools_sha256",
        "source_sha256",
    ):
        if not is_hex(value.get(field, "")):
            raise ValueError(f"invalid {field}")
    if value["test_tools_runtime_key"] != value["runtime_key"]:
        raise ValueError("test tools and deployment runtime identities differ")
    if not is_hex(value.get("commit", ""), 40):
        raise ValueError("commit must be a complete Git SHA")
    for field in ("build_id", "producer_step"):
        if not value[field] or not all(
            character in ALNUM + "_:-" for character in value[field]
        ):
            raise ValueError(f"invalid {field}")
    for field, _ in ARTIFACTS:
        path = value.get(field, "")
        if (
            not path
            or not all(character in ALNUM + "._/-" for character in path)
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
        ):
            raise ValueError(f"{field} must be an exact relative artifact path")
    return value


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


@contextmanager
def cache_lock(cache: Path, key: str):
    directory = cache / "locks"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / key).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def local_artifact(root: Path, path: str, expected: str) -> Path:
    artifact = root / path
    if not artifact.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes its local root")
    if artifact.is_symlink() or not artifact.is_file():
        raise ValueError(f"artifact must be a regular local file: {path}")
    if file_sha256(artifact) != expected:
        raise ValueError(f"artifact checksum mismatch: {path}")
    return artifact


def unpack_cached(archive: Path, cache: Path, expected: str) -> Path:
    target = cache / "bundles" / expected
    with cache_lock(cache, expected):
        marker = target / ".archive-sha256"
        if marker.is_file() and marker.read_text().strip() == expected:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(dir=target.parent, prefix="extract-"))
        try:
            with tarfile.open(archive, "r:*") as source:
                for member in source.getmembers():
                    parts = PurePosixPath(member.name).parts
                    if (
                        PurePosixPath(member.name).is_absolute()
                        or ".." in parts
                        or not (member.isfile() or member.isdir())
                    ):
                        raise ValueError(f"unsafe artifact member: {member.name}")
                source.extractall(temporary, filter="data")
            (temporary / ".archive-sha256").write_text(expected + "\n")
            if target.exists():
                raise ValueError(f"incomplete cache entry requires review: {target}")
            temporary.rename(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return target


def install_command(
    installer: Path,
    lock: Path,
    handoff: dict,
    target: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(installer),
        "--lock",
        str(lock),
        "--runtime-key",
        handoff["runtime_key"],
    ]
    if target is None:
        command += ["--wheel-sha256", handoff["wheel_sha256"]]
    else:
        command += [
            "--test-tools-key",
            handoff["test_tools_key"],
            "--target",
            str(target),
        ]
    return command


def prepare(
    handoff: dict,
    artifacts: Path,
    cache: Path,
    workspace: Path,
    installer: Path,
    prefix: Path,
) -> Path:
    validate_handoff(handoff)
    for variable, field in (
        ("BUILDKITE_COMMIT", "commit"),
        ("BUILDKITE_BUILD_ID", "build_id"),
    ):
        if os.environ.get(variable) and os.environ[variable] != handoff[field]:
            raise ValueError(f"{variable} differs from the producer handoff")
    if (prefix / ".vllm-runtime-key").read_text().strip() != handoff["runtime_key"]:
        raise ValueError("running container does not match the locked runtime")
    bundles = {}
    for path_field, hash_field in ARTIFACTS:
        source = local_artifact(artifacts, handoff[path_field], handoff[hash_field])
        bundles[path_field] = unpack_cached(source, cache, handoff[hash_field])
    source = bundles["source_path"]
    identity = json.loads((source / "source-identity.json").read_text())
    if identity.get("commit") != handoff["commit"]:
        raise ValueError("test source commit differs from the wheel handoff")
    if (source / "vllm").exists() or (source / "vllm.py").exists():
        raise ValueError("test source must keep vLLM sources under src/")
    if not (source / "tests").is_dir():
        raise ValueError("test source bundle does not contain tests/")
    wheel_lock = bundles["wheel_path"] / "artifact-lock.json"
    wheel_identity = json.loads(wheel_lock.read_text())
    if wheel_identity.get("source_commit") != handoff["commit"]:
        raise ValueError("commit wheel source differs from the test handoff")
    wheel_marker = prefix / ".vllm-wheel-sha256"
    if (
        not wheel_marker.is_file()
        or wheel_marker.read_text().strip() != handoff["wheel_sha256"]
    ):
        subprocess.run(
            install_command(installer, wheel_lock, handoff),
            check=True,
        )
    tools = cache / "test-tools" / handoff["test_tools_key"]
    with cache_lock(cache, handoff["test_tools_key"]):
        marker = tools / ".vllm-test-tools-key"
        if (
            not marker.is_file()
            or marker.read_text().strip() != handoff["test_tools_key"]
        ):
            if tools.exists():
                raise ValueError(
                    f"unverified test tools cache requires review: {tools}"
                )
            tools.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(dir=tools.parent, prefix="install-"))
            staged = temporary / "tools"
            try:
                subprocess.run(
                    install_command(
                        installer,
                        bundles["test_tools_path"] / "artifact-lock.json",
                        handoff,
                        staged,
                    ),
                    check=True,
                )
                if (staged / marker.name).read_text().strip() != handoff[
                    "test_tools_key"
                ]:
                    raise ValueError("installer did not verify staged test tools")
                staged.rename(tools)
            finally:
                shutil.rmtree(temporary)
        if marker.read_text().strip() != handoff["test_tools_key"]:
            raise ValueError("installer did not produce the expected test tools")
    workspace.mkdir(parents=True, exist_ok=True)
    if any(workspace.iterdir()):
        raise ValueError("deployment test workspace must be empty before staging")
    for item in source.iterdir():
        if item.name == ".archive-sha256":
            continue
        if item.is_dir():
            shutil.copytree(item, workspace / item.name)
        else:
            shutil.copyfile(item, workspace / item.name)
    return tools


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--check-deployment", action="store_true")
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    try:
        if args.launch:
            raise SystemExit(
                launch(
                    [item for item in sys.argv[1:] if item != "--launch"],
                    dict(os.environ),
                )
            )
        if args.check_deployment:
            if args.workspace is None:
                raise ValueError("--check-deployment requires --workspace")
            check_deployment(args.workspace)
            return
        handoff = validate_handoff(
            json.loads(
                args.handoff.read_text()
                if args.handoff
                else os.environ["ROCM_CI_HANDOFF"]
            )
        )
        artifacts = args.artifact_root or Path(os.environ["ROCM_ARTIFACT_ROOT"])
        cache = args.cache_root or Path(os.environ["ROCM_CACHE_ROOT"])
        workspace = args.workspace or Path(
            os.environ.get("VLLM_CI_WORKSPACE", "/vllm-workspace")
        )
        prefix = Path(sys.prefix)
        tools = prepare(
            handoff,
            artifacts,
            cache,
            workspace,
            Path("/opt/vllm-rocm/install_artifacts.py"),
            prefix,
        )
        raise SystemExit(
            run_tests(
                workspace, tools, prefix, handoff, os.environ["VLLM_TEST_COMMANDS"]
            )
        )
    except (
        ValueError,
        OSError,
        KeyError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
