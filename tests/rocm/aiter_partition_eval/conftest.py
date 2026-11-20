# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER on compute partitions (DPX) versus whole chips (SPX).

The CI step says what the GPU should be:
  AITER_EVAL_PARTITION     DPX or SPX
  AITER_EVAL_CU            CUs HIP should report for the GPU
  AITER_EVAL_FULL_CHIP_CU  CUs of the whole chip (equal to AITER_EVAL_CU on SPX)
  AITER_EVAL_PARTITION_INFERRED  1 if AITER must infer the partition (amd-smi
                           reports none, as on an SR-IOV VF), 0 if it must not
AITER_EVAL_EMULATE_PARTITION=DPX (local runs only) makes AITER see every GPU
as one partition of a chip twice its size. HIP still reports the whole chip,
so the detection checks against HIP skip under emulation.
"""

import logging
import os
from dataclasses import dataclass

import pytest
import torch

from vllm.platforms import current_platform


@dataclass(frozen=True)
class Expected:
    partition: str | None
    cu: int
    full_chip_cu: int
    gfx: str
    emulated: str | None
    inferred: bool | None = None

    @property
    def is_partition(self) -> bool:
        return self.full_chip_cu != self.cu


@pytest.fixture(scope="session")
def aiter_mod():
    if not current_platform.is_rocm():
        pytest.skip("needs ROCm")
    return pytest.importorskip("aiter")


@pytest.fixture(scope="session")
def chip_info(aiter_mod):
    from aiter.jit.utils import chip_info

    emulate = os.environ.get("AITER_EVAL_EMULATE_PARTITION")
    if not emulate:
        yield chip_info
        return
    parts = {"DPX": 2, "QPX": 4, "CPX": 8}[emulate]
    if not hasattr(chip_info, "_query_node_gpus"):  # before #8224: only CU_NUM
        cu = torch.cuda.get_device_properties(0).multi_processor_count
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("CU_NUM", str(cu // parts))
            chip_info.get_cu_num.cache_clear()
            yield chip_info
        chip_info.get_cu_num.cache_clear()
        return
    real = chip_info._query_node_gpus

    def emulated():
        gpus, from_amdsmi, error = real()
        return (
            [
                (
                    uuid,
                    info._replace(
                        cu_num=info.cu_num // parts,
                        full_chip_cu_num=info.cu_num,
                        compute_partition=emulate,
                        num_partitions=parts,
                    ),
                )
                for uuid, info in gpus
            ],
            from_amdsmi,
            error,
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(chip_info, "_query_node_gpus", emulated)
        chip_info.get_cu_num.cache_clear()
        yield chip_info
    chip_info.get_cu_num.cache_clear()


@pytest.fixture(scope="session")
def expected(chip_info) -> Expected:
    props = torch.cuda.get_device_properties(0)
    gfx = props.gcnArchName.split(":")[0]
    hip_cu = props.multi_processor_count
    emulate = os.environ.get("AITER_EVAL_EMULATE_PARTITION")
    if emulate:
        parts = {"DPX": 2, "QPX": 4, "CPX": 8}[emulate]
        return Expected(emulate, hip_cu // parts, hip_cu, gfx, emulate)
    cu = int(os.environ.get("AITER_EVAL_CU", hip_cu))
    full = int(os.environ.get("AITER_EVAL_FULL_CHIP_CU", cu))
    inferred = os.environ.get("AITER_EVAL_PARTITION_INFERRED")
    return Expected(
        os.environ.get("AITER_EVAL_PARTITION"),
        cu,
        full,
        gfx,
        None,
        None if inferred is None else inferred == "1",
    )


def tuned_cu_nums(chip_info) -> tuple[int, ...]:
    """CU counts AITER looks rows up under; only the own count before #8225."""
    if hasattr(chip_info, "get_tuned_cu_nums"):
        return tuple(chip_info.get_tuned_cu_nums())
    return (chip_info.get_cu_num(),)


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def aiter_log(aiter_mod):
    """INFO messages of the aiter logger during the test."""
    log = logging.getLogger("aiter")
    handler, level = _Records(), log.level
    log.addHandler(handler)
    log.setLevel(min(level or logging.INFO, logging.INFO))
    yield handler.messages
    log.removeHandler(handler)
    log.setLevel(level)


def full_chip_fallbacks(messages: list[str]) -> list[str]:
    return [m for m in messages if "using the rows tuned for the full chip" in m]
