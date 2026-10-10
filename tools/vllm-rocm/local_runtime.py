# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze local wheels and assemble ROCm OCI artifacts without publishing.

Dependency refresh is handled by produce_runtime.py. This assembler consumes
frozen wheelhouses and prepares identical offline inputs for serving and CI.
"""

import argparse
import configparser
import contextlib
import gzip
import hashlib
import io
import json
import os
import posixpath
import shlex
import shutil
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

import regex as re
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "docker/Dockerfile.rocm_base"
DEPLOY_RECIPE = ROOT / "docker/Dockerfile.rocm"
INSTALLER = ROOT / "tools/vllm-rocm/install_artifacts.py"
SPLITTER = ROOT / "tools/vllm-rocm/split_runtime.py"
DIGEST_REF = re.compile(r"[^\s,@]+@sha256:[0-9a-f]{64}\Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def recipe_digest(path: Path, section: str, image_arg: str) -> str:
    """Hash a target family without invalidating unrelated targets in the file."""
    lines = path.read_text().splitlines(keepends=True)
    begin = f"# BEGIN vllm-rocm {section}\n"
    end = f"# END vllm-rocm {section}\n"
    if lines.count(begin) != 1 or lines.count(end) != 1:
        raise ValueError(f"recipe must declare exactly one {section} section")
    first, last = lines.index(begin), lines.index(end)
    if first >= last or not lines[0].startswith("# syntax="):
        raise ValueError("invalid scoped recipe or missing pinned frontend")
    globals = []
    for line in lines:
        if line.startswith("FROM "):
            break
        if line.startswith(f"ARG {image_arg}="):
            globals.append(line)
    if len(globals) != 1:
        raise ValueError(f"recipe needs one global {image_arg} argument")
    payload = lines[0] + globals[0] + "".join(lines[first : last + 1])
    return hashlib.sha256(payload.encode()).hexdigest()


def image_ref(value: str) -> str:
    if not DIGEST_REF.fullmatch(value):
        raise ValueError("image references must be repo@sha256:<64 hex digits>")
    return value


def wheel_record(path: Path) -> dict:
    # Read the wheel's identity instead of guessing from local filename aliases.
    if path.is_symlink() or not path.is_file() or not path.name.endswith(".whl"):
        raise ValueError(f"expected a regular wheel file: {path}")
    if not re.fullmatch(r"[A-Za-z0-9_.+!-]+\.whl", path.name):
        raise ValueError(f"unsupported wheel filename: {path.name}")
    with zipfile.ZipFile(path) as wheel:
        metadata = [n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError(f"expected one METADATA file: {path}")
        message = BytesParser().parsebytes(wheel.read(metadata[0]))
        name, version = message["Name"], message["Version"]
        if not name or not version:
            raise ValueError(f"missing wheel identity: {path}")
        roots, scripts = set(), set()
        for member in wheel.namelist():
            parts = member.split("/")
            if member.startswith("/") or ".." in parts:
                raise ValueError(f"unsafe wheel member: {member}")
            if parts[0].endswith(".dist-info"):
                if member.endswith("/entry_points.txt"):
                    config = configparser.ConfigParser()
                    config.read_string(wheel.read(member).decode())
                    if config.has_section("console_scripts"):
                        scripts.update(config["console_scripts"])
                continue
            if parts[0].endswith(".data"):
                if len(parts) < 3 or parts[1] not in ("purelib", "platlib"):
                    continue
                parts = parts[2:]
            if parts[0]:
                roots.add(parts[0].split(".")[0])
    normalized, filename_version, _, _ = parse_wheel_filename(path.name)
    if normalized != canonicalize_name(name) or filename_version != Version(version):
        raise ValueError(f"wheel filename and METADATA disagree: {path}")
    return dict(
        file=path.name,
        name=normalized,
        version=version,
        sha256=sha256(path),
        bytes=path.stat().st_size,
        import_roots=sorted(roots),
        scripts=sorted(scripts),
    )


def wheel_records(directory: Path) -> list[dict]:
    wheels = [wheel_record(p) for p in sorted(directory.glob("*.whl"))]
    if not wheels:
        raise ValueError(f"no wheels in {directory}")
    names = [record["name"] for record in wheels]
    if len(names) != len(set(names)):
        raise ValueError("wheelhouse contains multiple wheels for one distribution")
    return wheels


def platform_records(directory: Path) -> list[dict]:
    if not (directory / "opt").is_dir() or not (directory / "etc").is_dir():
        raise ValueError("platform sidecar must contain opt/ and etc/ directories")
    if set(p.name for p in directory.iterdir()) - {"opt", "etc"}:
        raise ValueError("platform sidecar may contain only opt/ and etc/")
    if any(p.name != "venv" for p in (directory / "opt").iterdir()):
        raise ValueError("platform opt/ may contain only venv/")
    if (directory / "opt").is_symlink() or (directory / "etc").is_symlink():
        raise ValueError("platform roots must be real directories")
    records = []
    for path in sorted(directory.rglob("*")):
        info = path.lstat()
        record = dict(path=path.relative_to(directory).as_posix())
        record["mode"] = stat.S_IMODE(info.st_mode)
        if path.is_symlink():
            target = os.readlink(path)
            if Path(target).is_absolute():
                if not posixpath.normpath(target).startswith("/opt/venv/"):
                    raise ValueError(f"sidecar symlink points outside venv: {path}")
            elif (
                not (path.parent / target).resolve().is_relative_to(directory.resolve())
            ):
                raise ValueError(f"sidecar symlink escapes its root: {path}")
            record["symlink"] = target
        elif path.is_file():
            record["sha256"] = sha256(path)
        elif not path.is_dir():
            raise ValueError(f"special file in platform sidecar: {path}")
        records.append(record)
    return records


def check_test_overlay(runtime: dict, tools: dict) -> None:
    protected = {w["name"] for w in runtime["wheels"]} | {"vllm"}
    modules, scripts = {"vllm"}, {"vllm"}
    for wheel in runtime["wheels"]:
        modules.update(wheel["import_roots"])
        scripts.update(wheel["scripts"])
    for wheel in tools["wheels"]:
        if (
            wheel["name"] in protected
            or modules.intersection(wheel["import_roots"])
            or scripts.intersection(wheel["scripts"])
        ):
            raise ValueError(
                "test tools must not replace runtime distributions/modules"
            )


def write_lock(directory: Path, manifest: dict) -> Path:
    manifest["key"] = identity(manifest)
    lines = [
        f"{w['name']}=={w['version']} --hash=sha256:{w['sha256']}\n"
        for w in manifest["wheels"]
    ]
    (directory / "requirements.lock").write_text("".join(lines))
    dest = directory / "artifact-lock.json"
    dest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return dest


def freeze(args: argparse.Namespace) -> Path:
    directory = args.wheelhouse.resolve()
    manifest = dict(
        schema=1,
        kind=args.kind,
        python="cp312",
        epoch=args.epoch,
        wheels=wheel_records(directory),
    )
    if args.kind == "runtime":
        if any(wheel["name"] == "vllm" for wheel in manifest["wheels"]):
            raise ValueError("dependency runtime must not contain a commit vLLM wheel")
        if not args.os_image or not args.platform or not args.arches:
            raise ValueError("runtime needs --os-image, --platform and --arches")
        arches = sorted(set(args.arches.split(";")))
        if not all(re.fullmatch(r"gfx[0-9a-f]+", arch) for arch in arches):
            raise ValueError("invalid GPU architecture list")
        manifest.update(
            os_image=image_ref(args.os_image),
            arches=arches,
            platform=platform_records(args.platform),
            recipe_sha256=recipe_digest(RECIPE, "runtime", "OS_IMAGE"),
            installer_sha256=sha256(INSTALLER),
            splitter_sha256=sha256(SPLITTER),
            encoding={
                "algorithm": "zstd",
                "level": args.compression_level,
                "oci": True,
            },
        )
    elif args.kind == "wheel":
        if len(manifest["wheels"]) != 1 or manifest["wheels"][0]["name"] != "vllm":
            raise ValueError("commit artifact must contain exactly one vllm wheel")
        if not args.runtime_lock:
            raise ValueError("commit artifact requires --runtime-lock")
        runtime = verify(args.runtime_lock)
        if runtime["kind"] != "runtime":
            raise ValueError("commit artifact needs a runtime dependency lock")
        manifest["runtime_key"] = runtime["key"]
        commit = (
            args.source_commit
            or subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip()
        )
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("commit artifact requires a complete source Git SHA")
        manifest["source_commit"] = commit
    elif args.kind == "test":
        if not args.runtime_lock:
            raise ValueError("test tools require --runtime-lock")
        runtime = verify(args.runtime_lock)
        if runtime["kind"] != "runtime":
            raise ValueError("test tools need a runtime dependency lock")
        check_test_overlay(runtime, manifest)
        manifest["runtime_key"] = runtime["key"]
    if args.provenance:
        # Audit records include source recipes and validation logs. They must
        # not invalidate identical runtime bytes when unrelated CI code changes.
        provenance = json.loads(args.provenance.read_text())
        (directory / "artifact-provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
    return write_lock(directory, manifest)


def verify(path: Path, platform: Path | None = None) -> dict:
    manifest = json.loads(path.read_text())
    payload = {k: v for k, v in manifest.items() if k != "key"}
    if manifest.get("schema") != 1 or identity(payload) != manifest.get("key"):
        raise ValueError("artifact manifest identity mismatch")
    if wheel_records(path.parent) != manifest["wheels"]:
        raise ValueError("wheelhouse changed since freezing")
    expected = "".join(
        f"{w['name']}=={w['version']} --hash=sha256:{w['sha256']}\n"
        for w in manifest["wheels"]
    )
    if (path.parent / "requirements.lock").read_text() != expected:
        raise ValueError("offline requirements do not match the artifact lock")
    if manifest["kind"] == "runtime":
        image_ref(manifest["os_image"])
        if recipe_digest(RECIPE, "runtime", "OS_IMAGE") != manifest["recipe_sha256"]:
            raise ValueError("runtime recipe changed since freezing")
        if sha256(INSTALLER) != manifest["installer_sha256"]:
            raise ValueError("offline installer changed since freezing")
        if sha256(SPLITTER) != manifest["splitter_sha256"]:
            raise ValueError("runtime payload grouping changed since freezing")
        if platform is not None and platform_records(platform) != manifest["platform"]:
            raise ValueError("platform sidecar changed since freezing")
    return manifest


def layout_context(
    directory: Path, key: str | None = None, digest: str | None = None
) -> str:
    """Read a single-platform OCI layout and verify its manifest/config blobs."""
    layout = json.loads((directory / "oci-layout").read_text())
    if layout.get("imageLayoutVersion") != "1.0.0":
        raise ValueError("unsupported OCI layout version")

    def read_blob(descriptor: dict) -> dict:
        value = descriptor["digest"]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
            raise ValueError("invalid OCI descriptor digest")
        path = directory / "blobs/sha256" / value.split(":")[1]
        if (
            sha256(path) != value.split(":")[1]
            or path.stat().st_size != descriptor["size"]
        ):
            raise ValueError("OCI blob digest/size mismatch")
        return json.loads(path.read_text())

    index = json.loads((directory / "index.json").read_text())
    descriptors = index["manifests"]
    while len(descriptors) == 1 and descriptors[0]["mediaType"].endswith(
        "index.v1+json"
    ):
        descriptors = read_blob(descriptors[0])["manifests"]
    if len(descriptors) != 1 or not descriptors[0]["mediaType"].endswith(
        "manifest.v1+json"
    ):
        raise ValueError("provide a single-platform OCI layout with no attestations")
    descriptor = descriptors[0]
    manifest = read_blob(descriptor)
    config = read_blob(manifest["config"])
    if config.get("os") != "linux" or config.get("architecture") != "amd64":
        raise ValueError("prototype requires linux/amd64 OCI images")
    if (
        key
        and config.get("config", {}).get("Labels", {}).get("vllm.rocm.dependency_key")
        != key
    ):
        raise ValueError("OCI runtime was assembled from a different dependency lock")
    if digest and descriptor["digest"] != digest:
        raise ValueError("OCI OS manifest does not match the pinned OS image")
    return f"oci-layout://{directory.resolve()}@{descriptor['digest']}"


def audit_image(args: argparse.Namespace) -> dict:
    """Scan every referenced OCI layer and image metadata with offline Gitleaks."""
    report = args.report.resolve()
    if report.exists():
        raise ValueError("audit report exists; choose a new local report path")
    directory = args.layout.resolve()
    scanner = args.gitleaks.resolve()
    if not scanner.is_file() or not os.access(scanner, os.X_OK):
        raise ValueError("provide an executable local Gitleaks binary")
    manifest_digest = layout_context(directory).rsplit("@", 1)[1]

    def blob(descriptor: dict) -> Path:
        digest = descriptor["digest"]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError("invalid OCI descriptor digest")
        path = directory / "blobs/sha256" / digest.split(":")[1]
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(directory)
            or not path.is_file()
            or path.stat().st_size != descriptor["size"]
            or sha256(path) != digest.split(":")[1]
        ):
            raise ValueError("OCI blob digest/size mismatch")
        return path

    manifest_path = directory / "blobs/sha256" / manifest_digest.split(":")[1]
    manifest = json.loads(manifest_path.read_text())
    config_path = blob(manifest["config"])
    layers = [(descriptor, blob(descriptor)) for descriptor in manifest["layers"]]
    env = {k: v for k, v in os.environ.items() if not k.startswith("GITLEAKS_")}
    with tempfile.TemporaryDirectory(prefix="vllm-rocm-audit-") as temporary:
        work = Path(temporary)
        scan = work / "scan"
        scan.mkdir()
        trusted_config = work / "trusted.toml"
        trusted_config.write_text(
            "[extend]\nuseDefault = true\n"
            '[[rules]]\nid = "generic-api-key"\n'
            "[[rules.allowlists]]\n"
            'description = "Public runtime artifact SHA256 label"\n'
            'regexTarget = "match"\n'
            "regexes = ['''^vllm\\.rocm\\.dependency_key\"\\s*:\\s*\""
            "[a-f0-9]{64}\"$''']\n"
        )
        ignore = work / "ignore"
        ignore.mkdir()
        shutil.copyfile(config_path, scan / "image-config.json")
        embedded_wheels = 0
        for number, (descriptor, source) in enumerate(layers):
            media = descriptor["mediaType"]
            suffix = {
                "application/vnd.oci.image.layer.v1.tar": ".tar",
                "application/vnd.oci.image.layer.v1.tar+gzip": ".tar.gz",
                "application/vnd.oci.image.layer.v1.tar+zstd": ".tar.zst",
                "application/vnd.docker.image.rootfs.diff.tar.gzip": ".tar.gz",
            }.get(media)
            if suffix is None:
                raise ValueError("unsupported OCI layer encoding for audit")
            destination = scan / f"layer-{number}{suffix}"
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
            # Gitleaks recognizes archive extensions, while wheel ZIP files
            # inside tar archives use .whl. Stage those separately as .zip.
            with contextlib.ExitStack() as stack:
                stream = stack.enter_context(source.open("rb"))
                if suffix == ".tar.gz":
                    stream = stack.enter_context(gzip.GzipFile(fileobj=stream))
                elif suffix == ".tar.zst":
                    try:
                        import zstandard
                    except ImportError as error:
                        raise ValueError(
                            "zstd image audits require the optional zstandard package"
                        ) from error
                    stream = stack.enter_context(
                        zstandard.ZstdDecompressor().stream_reader(stream)
                    )
                archive = stack.enter_context(tarfile.open(fileobj=stream, mode="r|"))
                for member in archive:
                    if (
                        member.isfile()
                        and member.name.endswith(".whl")
                        and not posixpath.basename(member.name).startswith(".wh.")
                    ):
                        wheel = archive.extractfile(member)
                        if wheel is None:
                            raise ValueError("unreadable wheel in OCI layer")
                        output = scan / f"embedded-wheel-{embedded_wheels}.zip"
                        with wheel, output.open("wb") as target:
                            shutil.copyfileobj(wheel, target)
                        embedded_wheels += 1
        version = subprocess.run(
            [str(scanner), "version"], capture_output=True, env=env, timeout=30
        )
        tool_version = version.stdout.decode(errors="replace").strip()
        if version.returncode or not re.fullmatch(r"v?\d+\.\d+\.\d+", tool_version):
            raise ValueError("local Gitleaks version check failed")
        raw_report = work / "findings.json"
        command = [
            str(scanner),
            "dir",
            "--config",
            str(trusted_config),
            "--gitleaks-ignore-path",
            str(ignore),
            "--redact=100",
            "--no-banner",
            "--no-color",
            "--log-level=error",
            "--ignore-gitleaks-allow",
            "--max-archive-depth=4",
            "--max-decode-depth=5",
            "--max-target-megabytes=0",
            "--report-format=json",
            "--report-path",
            str(raw_report),
            str(scan),
        ]
        process = subprocess.run(command, capture_output=True, env=env)
        if (
            process.returncode not in (0, 1)
            or process.stderr.strip()
            or not raw_report.is_file()
        ):
            raise ValueError("offline Gitleaks scan failed; raw output was withheld")
        findings = json.loads(raw_report.read_text())
        if not isinstance(findings, list) or bool(findings) != bool(process.returncode):
            raise ValueError("offline Gitleaks report and exit status disagree")
        sanitized = [
            {
                "rule": finding["RuleID"],
                "file": str(finding["File"]).removeprefix(str(scan) + "/"),
                "line": finding["StartLine"],
            }
            for finding in findings
        ]
    result = dict(
        schema=1,
        manifest=manifest_digest,
        scanner=dict(version=tool_version, sha256=sha256(scanner)),
        layers=[descriptor["digest"] for descriptor, _ in layers],
        embedded_wheels=embedded_wheels,
        exclusions=["exact public vllm.rocm.dependency_key SHA256 label match"],
        findings=sanitized,
        passed=not sanitized,
        scope=(
            "all referenced layers, including whiteouted files, "
            "and image config/history"
        ),
        limits=(
            "Gitleaks rules; nested archives depth 4 and decode depth 5; "
            "not proof of absence"
        ),
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def build_command(args: argparse.Namespace) -> list[str]:
    manifest = verify(args.lock, args.platform)
    if manifest["kind"] == "test":
        raise ValueError("build runtime or wheel artifacts; test tools are a context")
    if not re.fullmatch(r"vllm-rocm-[a-z0-9_-]+", args.builder):
        raise ValueError("use a dedicated local vllm-rocm-* builder")
    output = args.output.resolve()
    if "," in str(output):
        raise ValueError("OCI output path must not contain commas")
    target = "runtime" if manifest["kind"] == "runtime" else "deployment"
    command = [
        "docker",
        "buildx",
        "build",
        "--builder",
        args.builder,
        "--file",
        str(RECIPE if target == "runtime" else DEPLOY_RECIPE),
        "--target",
        target,
        "--network=none",
        "--platform=linux/amd64",
        "--provenance=false",
        "--sbom=false",
        "--build-arg",
        f"SOURCE_DATE_EPOCH={manifest['epoch']}",
        "--build-context",
        f"wheelhouse={args.lock.parent.resolve()}",
        "--build-context",
        f"installer={INSTALLER.parent}",
    ]
    force = "true" if target == "runtime" else "false"
    level = manifest["encoding"]["level"] if target == "runtime" else 3
    command += [
        "--output",
        (
            f"type=oci,dest={output},tar=false,compression=zstd,compression-level={level},"
            f"force-compression={force},oci-mediatypes=true,rewrite-timestamp=true"
        ),
    ]
    if target == "runtime":
        if not args.platform:
            raise ValueError("runtime assembly requires its frozen --platform sidecar")
        command += [
            "--build-context",
            f"platform={args.platform.resolve()}",
            "--build-arg",
            f"OS_IMAGE={manifest['os_image']}",
            "--build-arg",
            f"DEPENDENCY_KEY={manifest['key']}",
        ]
        if args.os_layout:
            command[command.index(f"OS_IMAGE={manifest['os_image']}")] = (
                "OS_IMAGE=locked_os"
            )
            command += [
                "--build-context",
                "locked_os="
                + layout_context(
                    args.os_layout, digest=manifest["os_image"].split("@")[1]
                ),
            ]
    else:
        if not args.runtime_image and not args.runtime_layout:
            raise ValueError("wheel assembly needs --runtime-layout or --runtime-image")
        command += [
            "--build-arg",
            "RUNTIME_IMAGE="
            + (
                image_ref(args.runtime_image)
                if args.runtime_image
                else "locked_runtime"
            ),
            "--build-arg",
            f"DEPENDENCY_KEY={manifest['runtime_key']}",
            "--build-arg",
            f"WHEEL_SHA256={manifest['wheels'][0]['sha256']}",
        ]
        if args.runtime_layout:
            command += [
                "--build-context",
                "locked_runtime="
                + layout_context(args.runtime_layout, key=manifest["runtime_key"]),
            ]
        if args.test_lock:
            tools = verify(args.test_lock)
            runtime = verify(args.runtime_lock) if args.runtime_lock else None
            if (
                not runtime
                or tools["kind"] != "test"
                or tools.get("runtime_key") != runtime["key"]
                or manifest["runtime_key"] != runtime["key"]
            ):
                raise ValueError("test tools and runtime dependency locks differ")
            check_test_overlay(runtime, tools)
            command[command.index("--target") + 1] = "deployment-test"
            command += [
                "--build-context",
                f"test_wheelhouse={args.test_lock.parent.resolve()}",
                "--build-arg",
                f"TEST_TOOLS_KEY={tools['key']}",
            ]
    command += [str(ROOT / "docker")]
    return command


def add_bundle_file(bundle: tarfile.TarFile, path: Path, name: str) -> None:
    info = bundle.gettarinfo(str(path), arcname=name)
    if not info.isfile():
        raise ValueError(f"artifact bundle requires regular files: {path}")
    info.uid = info.gid = info.mtime = 0
    info.uname = info.gname = ""
    with path.open("rb") as stream:
        bundle.addfile(info, stream)


def pack(args: argparse.Namespace) -> Path:
    """Prepare local wheel/tool/source bundles and a worker handoff contract."""
    runtime = verify(args.runtime_lock)
    wheel = verify(args.wheel_lock)
    tools = verify(args.test_lock)
    if (
        runtime["kind"] != "runtime"
        or wheel["kind"] != "wheel"
        or tools["kind"] != "test"
        or wheel.get("runtime_key") != runtime["key"]
        or tools.get("runtime_key") != runtime["key"]
    ):
        raise ValueError("wheel, test tools and selected runtime must match")
    check_test_overlay(runtime, tools)
    image_ref(args.runtime_image)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.producer_step):
        raise ValueError("invalid producer step key")
    checkout = args.checkout.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if wheel.get("source_commit") != commit:
        raise ValueError("vLLM wheel and test checkout source commits differ")
    output = args.output.resolve()
    if output.exists():
        raise ValueError("output exists; choose a new local artifact path")
    output.mkdir(parents=True)
    for name, lock, manifest in (
        ("wheel.tar", args.wheel_lock, wheel),
        ("test-tools.tar", args.test_lock, tools),
    ):
        with tarfile.open(output / name, "w", format=tarfile.PAX_FORMAT) as bundle:
            for filename in ["artifact-lock.json", "requirements.lock"] + [
                record["file"] for record in manifest["wheels"]
            ]:
                add_bundle_file(bundle, lock.parent / filename, filename)
    tracked = subprocess.check_output(
        [
            "git",
            "-C",
            str(checkout),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        text=True,
    ).split("\0")
    roots = {
        "tests",
        "benchmarks",
        "examples",
        "requirements",
        "tools",
        "plugins",
        "docker",
        ".buildkite",
    }
    configs = {"pyproject.toml", "setup.py", "setup.cfg", "rust-toolchain.toml"}
    names = sorted(
        {name for name in tracked if name.split("/")[0] in roots or name in configs}
    )
    with (
        (output / "source.tar.gz").open("wb") as stream,
        gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as compressed,
        tarfile.open(
            fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT
        ) as bundle,
    ):
        for name in names:
            path = checkout / name
            if path.is_symlink():
                raise ValueError(f"thin workspace contains a symlink: {name}")
            add_bundle_file(bundle, path, name)
        data = json.dumps({"commit": commit}, sort_keys=True).encode()
        info = tarfile.TarInfo("source-identity.json")
        info.size, info.mode = len(data), 0o644
        bundle.addfile(info, io.BytesIO(data))
    manifest = dict(
        schema=1,
        build_id=args.build_id,
        commit=commit,
        producer_step=args.producer_step,
        runtime_image=args.runtime_image,
        runtime_key=runtime["key"],
        wheel_path="wheel.tar",
        wheel_bundle_sha256=sha256(output / "wheel.tar"),
        wheel_sha256=wheel["wheels"][0]["sha256"],
        test_tools_path="test-tools.tar",
        test_tools_sha256=sha256(output / "test-tools.tar"),
        test_tools_key=tools["key"],
        test_tools_runtime_key=runtime["key"],
        source_path="source.tar.gz",
        source_sha256=sha256(output / "source.tar.gz"),
    )
    dest = output / "handoff.json"
    dest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    lock = commands.add_parser("freeze", help="hash already resolved local artifacts")
    lock.add_argument("--kind", choices=("runtime", "wheel", "test"), required=True)
    lock.add_argument("--wheelhouse", type=Path, required=True)
    lock.add_argument("--os-image")
    lock.add_argument("--platform", type=Path)
    lock.add_argument("--arches")
    lock.add_argument("--epoch", type=int, required=True)
    lock.add_argument("--runtime-lock", type=Path)
    lock.add_argument("--provenance", type=Path)
    lock.add_argument("--source-commit", help="wheel source SHA; defaults to repo HEAD")
    lock.add_argument("--compression-level", type=int, choices=(1, 3, 9, 15), default=9)
    check = commands.add_parser("verify")
    check.add_argument("lock", type=Path)
    check.add_argument("--platform", type=Path)
    bundles = commands.add_parser("pack", help="prepare local CI handoff bundles")
    bundles.add_argument("--runtime-lock", type=Path, required=True)
    bundles.add_argument("--runtime-image", required=True)
    bundles.add_argument("--wheel-lock", type=Path, required=True)
    bundles.add_argument("--test-lock", type=Path, required=True)
    bundles.add_argument("--build-id", required=True)
    bundles.add_argument("--producer-step", required=True)
    bundles.add_argument("--checkout", type=Path, default=ROOT)
    bundles.add_argument("--output", type=Path, required=True)
    build = commands.add_parser("assemble", help="print a local OCI build command")
    build.add_argument("--lock", type=Path, required=True)
    build.add_argument("--platform", type=Path)
    runtime = build.add_mutually_exclusive_group()
    runtime.add_argument("--runtime-image")
    runtime.add_argument("--runtime-layout", type=Path)
    build.add_argument("--os-layout", type=Path)
    build.add_argument("--runtime-lock", type=Path)
    build.add_argument("--test-lock", type=Path)
    build.add_argument("--builder", default="vllm-rocm-local")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--execute", action="store_true", help="build local OCI only")
    build.add_argument(
        "--gitleaks", type=Path, help="local scanner required with --execute"
    )
    build.add_argument("--audit-report", type=Path)
    audit = commands.add_parser("audit", help="scan all local OCI layers for secrets")
    audit.add_argument("--layout", type=Path, required=True)
    audit.add_argument("--gitleaks", type=Path, required=True)
    audit.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            if args.epoch < 0:
                raise ValueError("epoch must be nonnegative")
            print(freeze(args))
        elif args.command == "verify":
            print(verify(args.lock, args.platform)["key"])
        elif args.command == "pack":
            print(pack(args))
        elif args.command == "audit":
            result = audit_image(args)
            print(args.report)
            if not result["passed"]:
                raise ValueError(
                    "secret audit failed; inspect the sanitized local report"
                )
        else:
            command = build_command(args)
            print(shlex.join(command), flush=True)
            if args.execute:
                if (
                    args.gitleaks is None
                    or not args.gitleaks.is_file()
                    or not os.access(args.gitleaks, os.X_OK)
                ):
                    raise ValueError(
                        "--execute requires an executable local --gitleaks"
                    )
                if args.output.exists():
                    raise ValueError("output exists; choose a new local artifact path")
                report = args.audit_report or args.output.with_name(
                    args.output.name + ".audit.json"
                )
                if report.exists():
                    raise ValueError(
                        "audit report exists; choose a new local report path"
                    )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(command, check=True)
                result = audit_image(
                    argparse.Namespace(
                        layout=args.output, gitleaks=args.gitleaks, report=report
                    )
                )
                if not result["passed"]:
                    raise ValueError(
                        "secret audit failed; inspect the local audit report"
                    )
    except (
        ValueError,
        OSError,
        KeyError,
        tarfile.TarError,
        zipfile.BadZipFile,
        subprocess.SubprocessError,
    ) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
