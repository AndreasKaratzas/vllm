# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install frozen commit/test wheels offline in both serving assembly and CI."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_lock(path: Path, runtime_key: str) -> dict:
    manifest = json.loads(path.read_text())
    payload = {k: v for k, v in manifest.items() if k != "key"}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest.get("schema") != 1 or digest != manifest.get("key"):
        raise ValueError("artifact manifest identity mismatch")
    if manifest.get("runtime_key") != runtime_key:
        raise ValueError("artifact is for a different dependency runtime")
    wheels = manifest["wheels"]
    expected = set()
    lines = []
    for wheel in wheels:
        filename = wheel["file"]
        if Path(filename).name != filename or not filename.endswith(".whl"):
            raise ValueError("unsafe wheel filename")
        local = path.parent / filename
        if (
            local.is_symlink()
            or not local.is_file()
            or sha256(local) != wheel["sha256"]
            or local.stat().st_size != wheel["bytes"]
        ):
            raise ValueError(f"wheel hash/size mismatch: {filename}")
        expected.add(filename)
        lines.append(
            f"{wheel['name']}=={wheel['version']} --hash=sha256:{wheel['sha256']}\n"
        )
    if not expected or {p.name for p in path.parent.glob("*.whl")} != expected:
        raise ValueError("wheelhouse must contain exactly the locked wheels")
    if (path.parent / "requirements.lock").read_text() != "".join(lines):
        raise ValueError("requirements do not match the artifact lock")
    return manifest


def install(args: argparse.Namespace) -> None:
    """Check identity, install without resolution, and record the installed key."""
    prefix = Path(sys.prefix)
    if (prefix / ".vllm-runtime-key").read_text().strip() != args.runtime_key:
        raise ValueError("selected image does not match the dependency runtime")
    manifest = read_lock(args.lock.resolve(), args.runtime_key)
    kind = manifest["kind"]
    if kind == "wheel":
        if (
            args.target
            or args.test_tools_key
            or len(manifest["wheels"]) != 1
            or manifest["wheels"][0]["name"] != "vllm"
            or manifest["wheels"][0]["sha256"] != args.wheel_sha256
        ):
            raise ValueError("commit installation requires the selected vLLM wheel")
        marker = prefix / ".vllm-wheel-sha256"
        installed_key = args.wheel_sha256
    elif kind == "test":
        if (
            not args.target
            or not args.target.is_absolute()
            or args.wheel_sha256
            or args.test_tools_key != manifest["key"]
        ):
            raise ValueError("test installation requires its key and absolute target")
        if args.target.exists():
            raise ValueError("test tools target already exists; use its verified cache")
        marker = args.target / ".vllm-test-tools-key"
        installed_key = args.test_tools_key
    else:
        raise ValueError("only commit wheels and test tools can be installed here")
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("PIP_")
    }
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PIP_NO_INDEX": "1",
            "PIP_NO_CACHE_DIR": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_CONFIG_FILE": os.devnull,
        }
    )
    command = [
        sys.executable,
        "-B",
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--no-compile",
        "--require-hashes",
        "--find-links",
        str(args.lock.resolve().parent),
        "-r",
        str(args.lock.resolve().parent / "requirements.lock"),
    ]
    if kind == "wheel":
        command.append("--force-reinstall")
        marker.unlink(missing_ok=True)
    if kind == "test":
        command += ["--target", str(args.target)]
        env["PYTHONPATH"] = str(args.target)
    subprocess.run(command, check=True, env=env)
    subprocess.run([sys.executable, "-B", "-m", "pip", "check"], check=True, env=env)
    marker.write_text(f"{installed_key}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--runtime-key", required=True)
    parser.add_argument("--wheel-sha256")
    parser.add_argument("--test-tools-key")
    parser.add_argument("--target", type=Path)
    args = parser.parse_args()
    try:
        install(args)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
