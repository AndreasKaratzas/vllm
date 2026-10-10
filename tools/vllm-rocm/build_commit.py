# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build only the commit wheel against a frozen ROCm runtime, locally/offline.

Prepare an inputs directory containing build-wheels/, native-debs/,
rust-toolchain/{cargo,rustup}/, cargo-vendor/{config.toml,crates}/ and
triton-kernels/. Populate these during an explicit dependency refresh, then
freeze their bytes. Cargo's vendor config must use /opt/cargo-vendor/crates.
The recipe performs no dependency resolution or downloads. A real BuildKit
builder and complete native build prerequisites are required to execute it.
"""

import argparse
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

import regex as re
from local_runtime import (
    ROOT,
    identity,
    image_ref,
    layout_context,
    recipe_digest,
    sha256,
    verify,
    wheel_records,
)
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from setuptools_scm import get_version

RECIPE = ROOT / "docker/Dockerfile.rocm"
INPUT_DIRS = (
    "build-wheels",
    "native-debs",
    "rust-toolchain",
    "cargo-vendor",
    "triton-kernels",
)
GIT_DESCRIBE = [
    "git",
    "describe",
    "--dirty",
    "--tags",
    "--long",
    "--abbrev=40",
    "--match",
    "v[0-9]*",
]
NATIVE_ROOTS = {"cmake", "csrc"}
NATIVE_FILES = {
    "setup.py",
    "CMakeLists.txt",
    "pyproject.toml",
    "tools/build_rust.py",
    "vllm/envs.py",
    "vllm/__init__.py",
}
RUST_ROOTS = {"rust"}
RUST_FILES = {
    "rust-toolchain.toml",
    "requirements/build/rust.txt",
    "tools/build_rust.py",
    "tools/build_rust.sh",
}
PACKAGE_ROOTS = {"vllm", "requirements", "cmake", "csrc"}
PACKAGE_FILES = {
    "setup.py",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "MANIFEST.in",
    "CMakeLists.txt",
    "tools/build_rust.py",
    "rust-toolchain.toml",
}


def tree_records(root: Path) -> list[dict]:
    """Hash a prepared prerequisite tree, rejecting external symlinks."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"expected an artifact directory: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        record = {"path": path.relative_to(root).as_posix()}
        record["mode"] = stat.S_IMODE(path.lstat().st_mode)
        if path.is_symlink():
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"artifact symlink escapes its directory: {path}")
            record["symlink"] = os.readlink(path)
        elif path.is_file():
            record.update(sha256=sha256(path), bytes=path.stat().st_size)
        elif not path.is_dir():
            raise ValueError(f"special file in build prerequisites: {path}")
        records.append(record)
    if not records:
        raise ValueError(f"empty build prerequisite directory: {root}")
    return records


def source_pins(checkout: Path) -> dict:
    channel = tomllib.loads((checkout / "rust-toolchain.toml").read_text())[
        "toolchain"
    ]["channel"]
    text = (checkout / "cmake/external_projects/triton_kernels.cmake").read_text()
    commits = re.findall(r'set\(TRITON_KERNELS_TAG "([0-9a-f]{40})"\)', text)
    if len(commits) != 1:
        raise ValueError("could not find the ROCm Triton-kernels source pin")
    return dict(
        rust_channel=channel,
        cargo_lock_sha256=sha256(checkout / "rust/Cargo.lock"),
        triton_kernels_commit=commits[0],
    )


def validate_inputs(inputs: Path, checkout: Path, runtime: dict) -> dict:
    pins = source_pins(checkout)
    wheels = wheel_records(inputs / "build-wheels")
    build_tools = {
        "cmake",
        "ninja",
        "packaging",
        "setuptools",
        "setuptools-scm",
        "setuptools-rust",
        "wheel",
        "jinja2",
        "markupsafe",
        "semantic-version",
        "vcs-versioning",
        "typing-extensions",
    }
    protected = {w["name"] for w in runtime["wheels"]} - build_tools
    protected_roots = {"torch", "triton", "pybind11", "vllm"}
    for wheel in runtime["wheels"]:
        if wheel["name"] in protected:
            protected_roots.update(wheel["import_roots"])
    for wheel in wheels:
        if wheel["name"] in protected or protected_roots.intersection(
            wheel["import_roots"]
        ):
            raise ValueError("build tools must reuse the runtime's GPU frameworks")
    versions = {w["name"]: w["version"] for w in runtime["wheels"]}
    versions.update({w["name"]: w["version"] for w in wheels})
    # Deliberately use the runtime's torch instead of pyproject's CUDA-oriented
    # torch pin: setup.py builds directly, without PEP 517 isolation/resolution.
    requirements = tomllib.loads((checkout / "pyproject.toml").read_text())[
        "build-system"
    ]["requires"]
    requirements += (checkout / "requirements/build/rust.txt").read_text().splitlines()
    for line in requirements:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line)
        name = canonicalize_name(requirement.name)
        if name == "torch":
            continue
        if requirement.marker and not requirement.marker.evaluate():
            continue
        if name not in versions or not requirement.specifier.contains(versions[name]):
            raise ValueError(f"missing/incompatible frozen build requirement: {line}")
    debs = sorted((inputs / "native-debs").glob("*.deb"))
    if not debs or len(list((inputs / "native-debs").iterdir())) != len(debs):
        raise ValueError("native-debs must contain only a complete local .deb closure")
    packages = []
    for deb in debs:
        package, version, architecture = subprocess.check_output(
            [
                "dpkg-deb",
                "--show",
                "--showformat=${Package}\\n${Version}\\n${Architecture}",
                str(deb),
            ],
            text=True,
        ).splitlines()
        if architecture not in {"amd64", "all"}:
            raise ValueError("native build packages must target linux/amd64")
        packages.append(dict(file=deb.name, name=package, version=version))
    rust = inputs / "rust-toolchain"
    if not all(
        (rust / "cargo/bin" / name).is_file() for name in ("rustup", "cargo", "rustc")
    ):
        raise ValueError("rust-toolchain needs cargo/bin/rustup and installed proxies")
    channel = pins["rust_channel"]
    toolchains = [
        p.name
        for p in (rust / "rustup/toolchains").glob("*")
        if p.name.startswith((channel + "-", channel + "."))
        and p.name.endswith("-x86_64-unknown-linux-gnu")
        and (p / "bin/rustc").is_file()
        and (p / "bin/cargo").is_file()
    ]
    if len(toolchains) != 1:
        raise ValueError(
            f"provide exactly one installed Rust {channel} amd64 toolchain"
        )
    config = tomllib.loads((inputs / "cargo-vendor/config.toml").read_text())
    sources = config.get("source", {})
    if not sources.get("crates-io", {}).get("replace-with"):
        raise ValueError("Cargo vendor config must replace crates-io")
    directories = [s["directory"] for s in sources.values() if "directory" in s]
    if directories != ["/opt/cargo-vendor/crates"]:
        raise ValueError(
            "Cargo vendor config directory must be /opt/cargo-vendor/crates"
        )
    if not list((inputs / "cargo-vendor/crates").glob("*/.cargo-checksum.json")):
        raise ValueError("missing vendored Cargo crates")
    if not (inputs / "triton-kernels/__init__.py").is_file():
        raise ValueError("triton-kernels must be the pinned Python package tree")
    provenance = (inputs / "triton-kernels-source.json").read_text()
    if json.loads(provenance).get("commit") != pins["triton_kernels_commit"]:
        raise ValueError("Triton-kernels provenance differs from the source pin")
    return dict(
        **pins,
        rust_toolchain=toolchains[0],
        wheels=wheels,
        native_packages=packages,
        trees={name: tree_records(inputs / name) for name in INPUT_DIRS},
        triton_kernels_source_sha256=sha256(inputs / "triton-kernels-source.json"),
    )


def freeze(args: argparse.Namespace) -> Path:
    runtime = verify(args.runtime_lock)
    if runtime["kind"] != "runtime":
        raise ValueError("build inputs require a runtime lock")
    inputs = args.inputs.resolve()
    manifest = dict(
        schema=1,
        kind="rocm-build-inputs",
        runtime_key=runtime["key"],
        recipe_sha256=recipe_digest(RECIPE, "commit-build", "RUNTIME_IMAGE"),
        **validate_inputs(inputs, args.checkout.resolve(), runtime),
    )
    manifest["key"] = identity(manifest)
    (inputs / "build-requirements.lock").write_text(
        "".join(
            f"{w['name']}=={w['version']} --hash=sha256:{w['sha256']}\n"
            for w in manifest["wheels"]
        )
    )
    destination = inputs / "build-lock.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return destination


def verify_build(lock: Path, runtime_lock: Path, checkout: Path) -> tuple[dict, dict]:
    manifest = json.loads(lock.read_text())
    payload = {k: v for k, v in manifest.items() if k != "key"}
    if (
        manifest.get("schema") != 1
        or manifest.get("kind") != "rocm-build-inputs"
        or manifest.get("key") != identity(payload)
    ):
        raise ValueError("build prerequisite identity mismatch")
    runtime = verify(runtime_lock)
    if runtime["kind"] != "runtime" or runtime["key"] != manifest["runtime_key"]:
        raise ValueError("build prerequisites target a different runtime")
    if (
        recipe_digest(RECIPE, "commit-build", "RUNTIME_IMAGE")
        != manifest["recipe_sha256"]
    ):
        raise ValueError("commit recipe changed since freezing")
    current = validate_inputs(lock.parent, checkout, runtime)
    if any(manifest.get(k) != value for k, value in current.items()):
        raise ValueError("build prerequisites or pinned source inputs changed")
    expected = "".join(
        f"{w['name']}=={w['version']} --hash=sha256:{w['sha256']}\n"
        for w in manifest["wheels"]
    )
    if (lock.parent / "build-requirements.lock").read_text() != expected:
        raise ValueError("build requirements differ from the frozen wheelhouse")
    return manifest, runtime


def contexts(checkout: Path, destination: Path, version: str) -> dict[str, Path]:
    """Snapshot tracked build inputs without the CI workspace or Git metadata."""
    if destination.exists():
        raise ValueError("context output exists; choose a new local directory")
    tracked = subprocess.check_output(
        ["git", "-C", str(checkout), "ls-files", "-z"],
        text=True,
    ).split("\0")
    output = {name: destination / name for name in ("native", "rust", "package")}
    for name, root in output.items():
        root.mkdir(parents=True)
        roots, files = {
            "native": (NATIVE_ROOTS, NATIVE_FILES),
            "rust": (RUST_ROOTS, RUST_FILES),
            "package": (PACKAGE_ROOTS, PACKAGE_FILES),
        }[name]
        for relative in sorted(set(tracked) - {""}):
            if relative.split("/")[0] not in roots and relative not in files:
                continue
            source = checkout / relative
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"build source must be a regular file: {relative}")
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            target.chmod(stat.S_IMODE(source.stat().st_mode))
            os.utime(target, (0, 0))
    output["version"] = destination / "version"
    output["version"].mkdir()
    (output["version"] / "vllm-version.txt").write_text(version + "\n")
    os.utime(output["version"] / "vllm-version.txt", (0, 0))
    return output


def build_command(args: argparse.Namespace) -> list[str]:
    checkout = args.checkout.resolve()
    manifest, runtime = verify_build(args.lock.resolve(), args.runtime_lock, checkout)
    if not re.fullmatch(r"vllm-rocm-[a-z0-9_-]+", args.builder):
        raise ValueError("use a dedicated local vllm-rocm-* builder")
    for path in (args.output, args.contexts, args.cache_to, args.cache_from):
        if path and any(c in str(path) for c in (",", "\n", "\r")):
            raise ValueError("local Buildx paths cannot contain commas or newlines")
    if args.output.exists():
        raise ValueError("wheel output exists; choose a new local directory")
    if args.max_jobs < 1:
        raise ValueError("--max-jobs must be positive")
    version = get_version(root=checkout, git_describe_command=GIT_DESCRIBE)
    snapshot = contexts(checkout, args.contexts.resolve(), version)
    metadata = dict(
        schema=1,
        commit=subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        source_version=version,
        runtime_key=runtime["key"],
        build_key=manifest["key"],
        contexts={
            name: identity(tree_records(path)) for name, path in snapshot.items()
        },
    )
    (args.contexts / "source-identity.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    command = [
        "docker",
        "buildx",
        "build",
        "--builder",
        args.builder,
        "--file",
        str(RECIPE),
        "--target",
        "wheel-export",
        "--network=none",
        "--provenance=false",
        "--sbom=false",
        "--platform=linux/amd64",
        "--output",
        f"type=local,dest={args.output.resolve()}",
    ]
    runtime_source = (
        image_ref(args.runtime_image) if args.runtime_image else "locked_runtime"
    )
    for name, value in dict(
        RUNTIME_IMAGE=runtime_source,
        DEPENDENCY_KEY=runtime["key"],
        SOURCE_DATE_EPOCH=runtime["epoch"],
        RUSTUP_TOOLCHAIN=manifest["rust_toolchain"],
        MAX_JOBS=args.max_jobs,
        PYTORCH_ROCM_ARCH=";".join(runtime["arches"]),
    ).items():
        command += ["--build-arg", f"{name}={value}"]
    if args.runtime_layout:
        command += [
            "--build-context",
            "locked_runtime="
            + layout_context(
                args.runtime_layout,
                key=runtime["key"],
            ),
        ]
    for name, path in snapshot.items():
        command += ["--build-context", f"{name}_source={path}"]
    for name in INPUT_DIRS:
        command += [
            "--build-context",
            f"{name.replace('-', '_')}={args.lock.parent.resolve() / name}",
        ]
    command += ["--build-context", f"build_manifest={args.lock.parent.resolve()}"]
    for name in ("from", "to"):
        cache = getattr(args, f"cache_{name}")
        if cache:
            options = (
                f"type=local,{'src' if name == 'from' else 'dest'}={cache.resolve()}"
            )
            if name == "to":
                options += (
                    ",mode=max,compression=zstd,compression-level=3,oci-mediatypes=true"
                )
            command += [f"--cache-{name}", options]
    command.append(str(ROOT / "docker"))
    return command


def freeze_output(args: argparse.Namespace) -> list[str]:
    source = json.loads((args.contexts / "source-identity.json").read_text())
    runtime = verify(args.runtime_lock)
    return [
        sys.executable,
        str(ROOT / "tools/vllm-rocm/local_runtime.py"),
        "freeze",
        "--kind",
        "wheel",
        "--wheelhouse",
        str(args.output.resolve()),
        "--runtime-lock",
        str(args.runtime_lock.resolve()),
        "--epoch",
        str(runtime["epoch"]),
        "--source-commit",
        source["commit"],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser(
        "freeze", help="freeze prepared local BUILD inputs"
    )
    freeze_parser.add_argument("--inputs", type=Path, required=True)
    freeze_parser.add_argument("--runtime-lock", type=Path, required=True)
    freeze_parser.add_argument("--checkout", type=Path, default=ROOT)
    build = commands.add_parser(
        "build", help="prepare source contexts and print a local build"
    )
    build.add_argument("--lock", type=Path, required=True)
    build.add_argument("--runtime-lock", type=Path, required=True)
    build.add_argument("--checkout", type=Path, default=ROOT)
    runtime = build.add_mutually_exclusive_group(required=True)
    runtime.add_argument("--runtime-image")
    runtime.add_argument("--runtime-layout", type=Path)
    build.add_argument("--contexts", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--builder", default="vllm-rocm-local")
    build.add_argument("--max-jobs", type=int, default=4)
    build.add_argument("--cache-from", type=Path)
    build.add_argument("--cache-to", type=Path)
    build.add_argument(
        "--execute", action="store_true", help="export local wheels only"
    )
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            print(freeze(args))
        else:
            command = build_command(args)
            print(shlex.join(command))
            output_command = freeze_output(args)
            print(shlex.join(output_command))
            if args.execute:
                subprocess.run(command, check=True)
                wheels = wheel_records(args.output)
                if len(wheels) != 1 or wheels[0]["name"] != "vllm":
                    raise ValueError("build must export exactly one vLLM wheel")
                subprocess.run(output_command, check=True)
    except (
        ValueError,
        OSError,
        KeyError,
        LookupError,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
