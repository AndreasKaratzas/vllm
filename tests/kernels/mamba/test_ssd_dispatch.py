# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.linear_attention.kernel import (
    LinearAttnSelectorConfig,
    select_linear_attn_kernel,
)
from vllm.model_executor.layers.mamba.ops import ssd_dispatch
from vllm.platforms.interface import DeviceCapability, PlatformEnum
from vllm.v1.kv_cache_interface import MambaSpec


def _cfg(platform, capability, head_dim=64, state_size=128, dtype=torch.bfloat16):
    """A layer with Nemotron's geometry by default, which FlashInfer's SSD scan
    is built for."""
    spec = MambaSpec(
        block_size=256,
        shapes=((3, 1024), (8, head_dim, state_size)),
        dtypes=(dtype, torch.float32),
    )
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(dtype=dtype))
    return LinearAttnSelectorConfig(platform, capability, vllm_config, spec)


SM100 = (PlatformEnum.CUDA, DeviceCapability(10, 0))
HOPPER = _cfg(PlatformEnum.CUDA, DeviceCapability(9, 0))
BLACKWELL = _cfg(*SM100)
CPU = _cfg(PlatformEnum.CPU, None)


def _select(cfg, requested=None, explicit=False):
    return select_linear_attn_kernel(
        "Mamba2 SSD", ssd_dispatch.SSD_KERNELS, cfg, requested, explicit
    )


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.parametrize(
    ("backend", "cfg", "reason"),
    [
        ("flashinfer", HOPPER, "compute capability 9.0"),
        ("flashinfer", _cfg(*SM100, head_dim=128), "head_dim 64"),
        ("flashinfer", _cfg(*SM100, dtype=torch.float16), "bfloat16"),
        ("triton", CPU, "CPU"),
        ("cpu", BLACKWELL, "only on CPU"),
    ],
)
def test_explicit_prefill_backend_that_cannot_run_fails_at_selection(
    backend, cfg, reason
):
    """An explicit choice fails at startup with its reason, not at the first
    prefill."""
    with pytest.raises(ValueError, match=reason):
        _select(cfg, backend, explicit=True)


@pytest.mark.parametrize("cfg", [HOPPER, _cfg(*SM100, head_dim=128)])
def test_inherited_flashinfer_falls_back_to_triton(caplog_vllm, cfg):
    """`--mamba-backend flashinfer` (FlashInfer decode) keeps working where
    FlashInfer's SSD scan can't run."""
    assert _select(cfg, "flashinfer") is ssd_dispatch.TritonSSDKernel
    assert len(_warnings(caplog_vllm)) == 1


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [(BLACKWELL, ssd_dispatch.TritonSSDKernel), (CPU, ssd_dispatch.CPUSSDKernel)],
)
def test_unset_backend_selects_the_platform_default(caplog_vllm, cfg, expected):
    assert _select(cfg) is expected
    assert not _warnings(caplog_vllm)
