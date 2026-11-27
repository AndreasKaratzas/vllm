# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel selection rules shared by every linear-attention op."""

import logging

import pytest

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear_attention.kernel import (
    LinearAttnKernel,
    LinearAttnSelectorConfig,
    select_linear_attn_kernel,
)
from vllm.platforms.interface import DeviceCapability, PlatformEnum

CFG = LinearAttnSelectorConfig(
    platform=PlatformEnum.CUDA,
    device_capability=DeviceCapability(9, 0),
    vllm_config=VllmConfig(),
)


def _kernel(name: str, *reasons: str) -> type[LinearAttnKernel]:
    return type(
        f"Fake{name}",
        (LinearAttnKernel,),
        {
            "name": name,
            "validate_configuration": classmethod(lambda c, cfg: [*reasons]),
        },
    )


FAST = _kernel("fast", "needs SM10x")
SLOW = _kernel("slow")
FALLBACK = _kernel("fallback")


def _records(caplog, level):
    return [r for r in caplog.records if r.levelno == level]


def test_auto_picks_the_first_valid_kernel_quietly(caplog_vllm):
    """With nothing requested, skipping kernels that can't run is not news."""
    selected = select_linear_attn_kernel("op_auto", [FAST, SLOW, FALLBACK], CFG)
    assert selected is SLOW
    assert not _records(caplog_vllm, logging.WARNING)


def test_explicit_valid_choice_wins_over_priority():
    selected = select_linear_attn_kernel(
        "op_explicit", [SLOW, FALLBACK], CFG, "fallback", explicit=True
    )
    assert selected is FALLBACK


def test_explicit_invalid_choice_fails_with_its_reasons():
    """Users who name a backend must get it or an error, never another one."""
    with pytest.raises(ValueError, match="needs SM10x"):
        select_linear_attn_kernel("op_strict", [FAST, SLOW], CFG, "fast", explicit=True)


def test_explicit_unknown_choice_lists_the_op_kernels():
    with pytest.raises(ValueError, match="fast, slow"):
        select_linear_attn_kernel(
            "op_unknown", [FAST, SLOW], CFG, "other", explicit=True
        )


@pytest.mark.parametrize("requested", ["fast", "other"])
def test_inherited_unusable_choice_falls_back_with_one_warning(caplog_vllm, requested):
    """`--mamba-backend flashinfer` on Hopper must not fail ops it can't serve."""
    for _ in range(2):
        selected = select_linear_attn_kernel(
            f"op_inherited_{requested}", [FAST, SLOW], CFG, requested
        )
        assert selected is SLOW
    assert len(_records(caplog_vllm, logging.WARNING)) == 1


def test_no_valid_kernel_fails_with_every_reason():
    with pytest.raises(ValueError, match=r"fast.*needs SM10x.*other.*needs CPU"):
        select_linear_attn_kernel("op_none", [FAST, _kernel("other", "needs CPU")], CFG)
