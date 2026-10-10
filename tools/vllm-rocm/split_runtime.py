# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate stable SDK/PyTorch/Triton payloads into reusable OCI blob groups."""

import argparse
import shutil
from pathlib import Path


def split(prefix: Path, output: Path) -> None:
    """Move a checked venv without changing its installed paths or contents."""
    if output.exists():
        raise ValueError("runtime payload destination already exists")
    site = prefix / "lib/python3.12/site-packages"
    if not site.is_dir() or not (site / "torch").is_dir():
        raise ValueError("expected the complete Python 3.12 ROCm runtime")
    relative = site.relative_to(prefix)
    for group in ("sdk", "torch", "triton"):
        (output / group / relative).mkdir(parents=True)
    for path in sorted(site.iterdir()):
        name = path.name
        if name.startswith(("_rocm_sdk_", "rocm")):
            group = "sdk"
        elif name in {"torch", "torchgen"} or name.startswith("torch-"):
            group = "torch"
        elif name == "triton" or name.startswith("triton-"):
            group = "triton"
        else:
            continue
        shutil.move(str(path), output / group / relative / name)
    shutil.move(str(prefix), output / "python")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=Path("/opt/venv"))
    parser.add_argument("--output", type=Path, default=Path("/runtime-payloads"))
    args = parser.parse_args()
    split(args.prefix, args.output)


if __name__ == "__main__":
    main()
