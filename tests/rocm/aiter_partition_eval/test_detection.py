# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""(a) What AITER, HIP, amd-smi and rocminfo report for the GPU of the job."""

import shutil
import subprocess

import pytest
import regex as re
import torch

from .conftest import tuned_cu_nums


def _run(*cmd: str) -> str:
    if shutil.which(cmd[0]) is None:
        return f"<{cmd[0]} not found>"
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return (proc.stdout + proc.stderr).strip()


def test_report(chip_info, expected):
    """Evidence only: every source side by side."""
    print(f"\nexpected: {expected}")
    for d in range(torch.accelerator.device_count()):
        p = torch.cuda.get_device_properties(d)
        print(f"torch[{d}]: {p.name} {p.gcnArchName} CUs={p.multi_processor_count}")
    if hasattr(chip_info, "get_partition_info"):
        for d in range(torch.accelerator.device_count()):
            print(f"aiter get_partition_info({d}) = {chip_info.get_partition_info(d)}")
    else:
        print("aiter has no get_partition_info (before #8224)")
    print(f"aiter get_cu_num() = {chip_info.get_cu_num()}")
    print(f"aiter tuned CU counts = {tuned_cu_nums(chip_info)}")
    print(
        "amd-smi static --partition --asic:\n"
        + _run("amd-smi", "static", "--partition", "--asic")
    )
    print("amd-smi list:\n" + _run("amd-smi", "list"))
    rocminfo = _run("rocminfo")
    print(
        "rocminfo:\n"
        + "\n".join(
            line
            for line in rocminfo.splitlines()
            if re.match(r"  (Name|Marketing Name|Compute Unit):", line)
        )
    )


def test_hip_cu_count_matches_queue(expected):
    """The job landed on the GPU kind its queue promises."""
    if expected.emulated:
        pytest.skip("emulated partition: HIP reports the whole chip")
    hip_cu = torch.cuda.get_device_properties(0).multi_processor_count
    assert hip_cu == expected.cu, (
        f"HIP reports {hip_cu} CUs, the step expects {expected.cu} "
        f"({expected.partition}); wrong queue or partition mode changed"
    )


def test_aiter_cu_count_matches_hip(chip_info, expected):
    if expected.emulated:
        pytest.skip("emulated partition: HIP reports the whole chip")
    for d in range(torch.accelerator.device_count()):
        hip_cu = torch.cuda.get_device_properties(d).multi_processor_count
        aiter_cu = (
            chip_info.get_partition_info(d).cu_num
            if hasattr(chip_info, "get_partition_info")
            else chip_info.get_cu_num()
        )
        assert aiter_cu == hip_cu, f"GPU {d}: AITER {aiter_cu} CUs, HIP {hip_cu}"


def test_partition_reported(chip_info, expected):
    """Amdsmi reports, or AITER infers, the partition and the chip size (#8224)."""
    if not hasattr(chip_info, "get_partition_info"):
        pytest.fail("aiter has no get_partition_info (before #8224)")
    info = chip_info.get_partition_info(0)
    print(f"\n{info}")
    if expected.is_partition:
        assert info.num_partitions == expected.full_chip_cu // expected.cu, info
        assert info.full_chip_cu_num == expected.full_chip_cu, info
        assert info.compute_partition == expected.partition, info
    else:
        assert info.num_partitions in (None, 1), info
        assert info.full_chip_cu_num in (None, expected.cu), info
    if expected.inferred is not None:
        assert getattr(info, "partition_inferred", False) == expected.inferred, info


def test_tuned_cu_fallback_enabled(chip_info, expected):
    """Lookups try the partition's own rows, then the whole chip's (#8225)."""
    got = tuned_cu_nums(chip_info)
    want = (
        (expected.cu, expected.full_chip_cu)
        if expected.is_partition
        else (expected.cu,)
    )
    print(f"\ntuned CU counts {got}, expected {want}")
    assert got == want
