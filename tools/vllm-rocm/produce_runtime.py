# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refresh ROCm dependency artifacts using local source targets and exports.

Only --execute runs the printed local-only Bake plan. Dependency refresh can
read package indexes and source repositories; the resulting installer is offline.
The initial explicit NIC profile is CX7. Other vendor profiles need seed recipes.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "docker/Dockerfile.rocm_base"
SOURCE_FILES = (
    ".dockerignore",
    "docker/Dockerfile.rocm_base",
    "docker/Dockerfile.rocm",
    "requirements/common.txt",
    "requirements/rocm.txt",
    "tools/install_torchcodec_rocm.sh",
    "tools/vllm-rocm/therock_wheels.py",
    "tools/vllm-rocm/produce_runtime.py",
)
XNACK_KPACKS = {
    "torch_gfx90a:xnack+.kpack",
    "torch_gfx90a:xnack-.kpack",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wheel_identity(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as archive:
        metadata = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError(f"wheel has ambiguous metadata: {path}")
        message = BytesParser().parsebytes(archive.read(metadata[0]))
    name = message["Name"].lower().replace("_", "-").replace(".", "-")
    while "--" in name:
        name = name.replace("--", "-")
    return name, message["Version"]


def remove_xnack_kpacks(path: Path) -> list[str]:
    """Carry the baseline's MI250 deletion into the owning wheel and RECORD."""
    with zipfile.ZipFile(path) as archive:
        removed = [n for n in archive.namelist() if Path(n).name in XNACK_KPACKS]
        if not removed:
            return []
        records = [n for n in archive.namelist() if n.endswith(".dist-info/RECORD")]
        if len(records) != 1:
            raise ValueError(f"wheel has ambiguous RECORD: {path}")
        rows = csv.reader(io.StringIO(archive.read(records[0]).decode()))
        record = io.StringIO(newline="")
        csv.writer(record, lineterminator="\n").writerows(
            row for row in rows if row[0] not in removed
        )
        temporary = path.with_suffix(".tmp")
        with zipfile.ZipFile(temporary, "w") as patched:
            for member in archive.infolist():
                if member.filename in removed:
                    continue
                data = (
                    record.getvalue().encode()
                    if member.filename == records[0]
                    else archive.read(member)
                )
                patched.writestr(member, data, compress_type=member.compress_type)
    temporary.replace(path)
    return removed


def capture_platform(reference: Path, output: Path) -> None:
    old_site = Path("/usr/local/lib/python3.12/dist-packages")
    new_site = Path("/opt/venv/lib/python3.12/site-packages")
    source = reference / old_site.relative_to("/")
    target = output / new_site.relative_to("/")
    (output / "etc/ld.so.conf.d").mkdir(parents=True)
    target.mkdir(parents=True)
    loader = source / "rocm_sdk/__init__.py"
    if "rtld_global: bool = False" not in loader.read_text():
        raise ValueError("reference is missing the PR #58761 ROCm loader patch")
    paths = [loader]
    patterns = (
        "libhsa-runtime64.so*",
        "libamdhip64.so*",
        "librocm_smi64.so*",
        "librocprofiler-sdk.so*",
        "librocprofiler-register.so*",
        "librocprofiler-sdk-roctx.so*",
    )
    for sdk in ("_rocm_sdk_devel", "_rocm_sdk_core"):
        for pattern in patterns:
            paths.extend(sorted((source / sdk / "lib").glob(pattern)))
    if not any(p.name.startswith("libhsa-runtime64.so") for p in paths):
        raise ValueError("reference is missing patched ROCr libraries")
    if not any(p.name.startswith("libamdhip64.so") for p in paths):
        raise ValueError("reference is missing patched CLR libraries")
    hardlinks = {}
    for path in paths:
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            link = os.readlink(path)
            if Path(link).is_absolute():
                if not Path(link).is_relative_to(old_site):
                    raise ValueError(f"unsupported SDK link target: {path}: {link}")
                link = str(new_site / Path(link).relative_to(old_site))
            destination.symlink_to(link)
        else:
            info = path.stat()
            inode = (info.st_dev, info.st_ino)
            if inode in hardlinks:
                os.link(hardlinks[inode], destination)
            else:
                shutil.copy2(path, destination)
                hardlinks[inode] = destination
    (output / "etc/ld.so.conf.d/rocm-pip.conf").write_text(
        f"{new_site}/_rocm_sdk_devel/lib\n{new_site}/_rocm_sdk_core/lib\n"
    )
    (output / "etc/kineto.conf").write_text("ROCTRACER_MAX_EVENTS=10000000\n")


def resolve(args: argparse.Namespace) -> None:
    output = args.output
    wheels = output / "wheelhouse"
    wheels.mkdir(parents=True)
    sources = sorted(args.source_wheels.glob("*.whl"))
    names = [wheel_identity(path)[0] for path in sources]
    required = {
        "torch",
        "triton",
        "amdsmi",
        "flash-attn",
        "amd-aiter",
        "mori",
        "nixl-rocm",
        "deep-ep",
        "lmcache",
        "torchcodec",
        "fastsafetensors",
    }
    # DeepEP's normalized distribution name depends on its wheel variant.
    if "deep-ep-rocm" in names:
        required.remove("deep-ep")
        required.add("deep-ep-rocm")
    if required - set(names) or len(names) != len(set(names)):
        raise ValueError(f"source wheel providers incomplete/ambiguous: {names}")
    arches = args.arches.split(";")
    extras = ",".join(f"device-{arch}" for arch in arches)
    command = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--dest",
        str(wheels),
        "--only-binary=:all:",
        "--find-links",
        str(args.source_wheels),
        "--index-url",
        "https://pypi.org/simple",
        "-r",
        str(args.requirements),
        "-c",
        "/etc/rocm-constraints.txt",
    ]
    for index in args.index:
        command.extend(["--extra-index-url", index])
    command.extend(str(path) for path in sources)
    command.extend(
        [
            f"rocm[libraries,devel,{extras}]=={args.sdk_version}",
            f"torch[{extras}]=={args.torch_version}",
            f"torchvision[{extras}]=={args.vision_version}",
            f"torchaudio=={args.audio_version}",
            "pybind11",
            "cmake<4",
            "quart",
            "msgpack",
            "blinker",
        ]
    )
    subprocess.run(command, check=True)
    patches = {}
    inventory = []
    lock = []
    for path in sorted(wheels.glob("*.whl")):
        before = sha256(path)
        removed = remove_xnack_kpacks(path)
        name, version = wheel_identity(path)
        digest = sha256(path)
        inventory.append(
            dict(file=path.name, name=name, version=version, sha256=digest)
        )
        lock.append(f"{name}=={version} --hash=sha256:{digest}\n")
        if removed:
            patches[path.name] = dict(original_sha256=before, removed=removed)
    if len({record["name"] for record in inventory}) != len(inventory):
        raise ValueError("resolver produced duplicate distributions")
    (wheels / "requirements.lock").write_text("".join(lock))
    capture_platform(args.reference, output / "platform")
    provenance = output / "provenance"
    provenance.mkdir(exist_ok=True)
    seed_python = json.loads((provenance / "seed/python.json").read_text())
    if seed_python["abi"] != sysconfig.get_config_var("SOABI"):
        raise ValueError("source wheel and clean seed Python ABIs disagree")
    source_commits = {}
    source_submodules = {}
    for name in ("fa", "aiter", "triton", "mori", "nixl", "ucx", "deepep"):
        source_commits[name] = subprocess.run(
            ["git", "-C", f"/refs/{name}", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if len(source_commits[name]) != 40 or any(
            c not in "0123456789abcdef" for c in source_commits[name]
        ):
            raise ValueError(f"invalid resolved source commit for {name}")
        source_submodules[name] = subprocess.run(
            ["git", "-C", f"/refs/{name}", "submodule", "status", "--recursive"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    (provenance / "producer.json").write_text(
        json.dumps(
            dict(
                schema=1,
                nic_profile="cx7",
                arches=arches,
                source_commits=source_commits,
                source_submodules=source_submodules,
                torchcodec_commit="0b261b98080925f2b709712a5491a1e8dd817065",
                lmcache_commit="140819c9d57a975dbc5678a6459a218e544cb58b",
                fastsafetensors_version="0.3.3",
                source_archives={
                    path.name: sha256(path)
                    for path in sorted(Path("/source-tarballs").glob("*"))
                },
                framework_arches=dict(
                    torch=arches,
                    flash_attn=arches,
                    aiter=["gfx942", "gfx950"],
                    deepep=["gfx942", "gfx950"],
                    lmcache=["gfx942", "gfx950"],
                ),
                wheels=inventory,
                wheel_patches=patches,
                reference_versions=(args.reference / "app/versions.txt").read_text(),
                reference_packages=subprocess.run(
                    [sys.executable, "-m", "pip", "freeze", "--all"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout,
                seed_files={
                    p.name: p.read_text()
                    for p in sorted((provenance / "seed").glob("*"))
                },
                hardware_validation="not performed",
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def validate(args: argparse.Namespace) -> None:
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    probe = (
        "import torch,triton; assert torch.version.hip; "
        "print(torch.__version__,triton.__version__)"
    )
    versions = subprocess.run(
        [sys.executable, "-c", probe], check=True, capture_output=True, text=True
    ).stdout.strip()
    site = args.site_packages
    devel, core = site / "_rocm_sdk_devel", site / "_rocm_sdk_core"
    env = os.environ | {
        "LD_LIBRARY_PATH": ":".join(
            str(p)
            for p in (
                site / "torch/lib",
                devel / "lib",
                devel / "lib/rocm_sysdeps/lib",
                core / "lib",
                core / "lib/rocm_sysdeps/lib",
            )
        )
    }
    examined, missing = 0, []
    for path in sorted(site.rglob("*.so*")):
        if path.is_symlink() or not path.is_file():
            continue
        with path.open("rb") as stream:
            header = stream.read(20)
        # Inspect only host x86_64 shared objects; AMD device code is not host ELF.
        if len(header) < 20 or header[:4] != b"\x7fELF" or header[5] != 1:
            continue
        if struct.unpack_from("<HH", header, 16) != (3, 62):
            continue
        result = subprocess.run(
            ["ldd", str(path)], env=env, capture_output=True, text=True
        )
        examined += 1
        if "not found" in result.stdout + result.stderr:
            missing.append(dict(path=str(path), ldd=result.stdout + result.stderr))
        elif result.returncode:
            raise ValueError(f"ldd failed for {path}: {result.stdout}{result.stderr}")
    if missing:
        raise ValueError(
            f"unresolved ELF dependencies on clean seed: {json.dumps(missing)}"
        )
    clang = subprocess.run(
        [str(devel / "lib/llvm/bin/clang"), "--version"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    args.output.mkdir(parents=True)
    (args.output / "validation.json").write_text(
        json.dumps(
            dict(
                pip_check="passed",
                torch_triton=versions,
                host_elf_checked=examined,
                sdk_clang=clang,
                gpu_execution="not performed",
                external_host_filesystem="not used",
            ),
            indent=2,
        )
        + "\n"
    )


def bake_config(args: argparse.Namespace) -> dict:
    image, sep, digest = args.ubuntu_image.partition("@sha256:")
    if (
        not sep
        or not image
        or any(c.isspace() or c in ",@" for c in image)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("provide the pinned Ubuntu 22.04 image manifest digest")
    if not all(
        arch.startswith("gfx")
        and len(arch) > 3
        and all(c in "0123456789abcdef" for c in arch[3:])
        for arch in args.arches.split(";")
    ):
        raise ValueError("invalid GPU architectures")
    if set(args.arches.split(";")) - {"gfx90a", "gfx942", "gfx950"}:
        raise ValueError("the first provider covers only CI gfx90a/gfx942/gfx950")
    if args.nic_profile != "cx7":
        raise ValueError(
            "only explicit CX7 has a seed recipe; vendor profiles are unsupported"
        )
    if not args.builder.startswith("vllm-rocm-") or not all(
        c in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in args.builder
    ):
        raise ValueError("use a dedicated local vllm-rocm-* builder")
    if args.jobs < 1 or args.epoch < 0 or "," in str(args.output.resolve()):
        raise ValueError("invalid jobs/output path")
    targets = {}
    base = dict(
        context=str(ROOT),
        platforms=["linux/amd64"],
        dockerfile="docker/Dockerfile.rocm_base",
        args=dict(
            BASE_IMAGE=args.ubuntu_image,
            PYTORCH_ROCM_ARCH=args.arches,
            SOURCE_DATE_EPOCH=str(args.epoch),
        ),
        output=["type=cacheonly"],
    )
    for name, stage in (
        ("framework-reference", "final"),
        ("framework-wheels", "debs_wheel_release"),
        ("mori-build", "build_mori"),
        ("fa-build", "build_fa"),
        ("aiter-build", "build_aiter"),
        ("triton-build", "build_triton"),
    ):
        targets[name] = base | {"target": stage}
    for name, stage in (
        ("nixl-build", "build_nixl"),
        ("deepep-build", "build_deepep"),
        ("lmcache-build", "build_lmcache"),
    ):
        targets[name] = dict(
            context=str(ROOT),
            platforms=["linux/amd64"],
            dockerfile="docker/Dockerfile.rocm",
            target=stage,
            args=dict(
                BASE_IMAGE="framework-reference",
                ARG_PYTORCH_ROCM_ARCH=args.arches,
                REMOTE_VLLM="0",
                NIC_BACKEND="none",
                DEEPEP_NIC="cx7",
                max_jobs=str(args.jobs),
                SOURCE_DATE_EPOCH=str(args.epoch),
            ),
            contexts={"framework-reference": "target:framework-reference"},
            output=["type=cacheonly"],
        )
    common = dict(
        context=str(ROOT),
        platforms=["linux/amd64"],
        dockerfile=str(RECIPE),
        args=dict(
            UBUNTU_IMAGE=args.ubuntu_image,
            PYTORCH_ROCM_ARCH=args.arches,
            MAX_JOBS=str(args.jobs),
            SOURCE_DATE_EPOCH=str(args.epoch),
        ),
        attest=["type=provenance,disabled=true", "type=sbom,disabled=true"],
    )
    targets["runtime-seed"] = common | dict(
        target="runtime-seed",
        output=[
            (
                f"type=oci,dest={args.output.resolve()}/seed.oci,tar=false,"
                "compression=zstd,compression-level=9,force-compression=true,"
                "rewrite-timestamp=true,oci-mediatypes=true,"
                "name=local/vllm-rocm-seed"
            )
        ],
    )
    targets["dependency-artifacts"] = common | dict(
        target="export-artifacts",
        contexts={name: f"target:{name}" for name in targets if name != "runtime-seed"},
        output=[f"type=local,dest={args.output.resolve()}/bundle"],
    )
    return dict(
        group={"default": {"targets": ["runtime-seed", "dependency-artifacts"]}},
        target=targets,
    )


def finalize(args: argparse.Namespace) -> None:
    import local_runtime

    context = local_runtime.layout_context(args.output / "seed.oci")
    digest = context.rpartition("@")[2]
    lock = local_runtime.freeze(
        argparse.Namespace(
            kind="runtime",
            wheelhouse=args.output / "bundle/wheelhouse",
            os_image=f"local/vllm-rocm-seed@{digest}",
            platform=args.output / "bundle/platform",
            arches=args.arches,
            epoch=args.epoch,
            compression_level=9,
            provenance=args.output / "bundle/provenance/producer.json",
        )
    )
    print(lock)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("produce")
    plan.add_argument("--ubuntu-image", required=True)
    plan.add_argument("--nic-profile", required=True)
    plan.add_argument("--arches", default="gfx90a;gfx942;gfx950")
    plan.add_argument("--output", required=True, type=Path)
    plan.add_argument("--builder", default="vllm-rocm-local")
    plan.add_argument("--jobs", type=int, default=16)
    plan.add_argument("--epoch", type=int, required=True)
    plan.add_argument("--execute", action="store_true")
    refresh = sub.add_parser("_resolve")
    for name in ("output", "source-wheels", "reference", "requirements"):
        refresh.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "arches",
        "sdk-version",
        "torch-version",
        "vision-version",
        "audio-version",
    ):
        refresh.add_argument(f"--{name}", required=True)
    refresh.add_argument("--index", action="append", required=True)
    check = sub.add_parser("_validate")
    check.add_argument("--site-packages", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "_resolve":
        resolve(args)
    elif args.command == "_validate":
        validate(args)
    else:
        args.output = args.output.resolve()
        config = bake_config(args)
        if not args.execute:
            print(json.dumps(config, indent=2))
            return
        if (args.output / "bundle").exists() or (args.output / "seed.oci").exists():
            raise ValueError(
                "producer output already contains artifacts; use a fresh directory"
            )
        args.output.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as plan:
            json.dump(config, plan)
            plan.flush()
            subprocess.run(
                [
                    "docker",
                    "buildx",
                    "bake",
                    "--builder",
                    args.builder,
                    "--file",
                    plan.name,
                ],
                check=True,
            )
        provenance = args.output / "bundle/provenance/producer.json"
        record = json.loads(provenance.read_text())
        record["recipe_inputs"] = {name: sha256(ROOT / name) for name in SOURCE_FILES}
        provenance.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        finalize(args)


if __name__ == "__main__":
    main()
