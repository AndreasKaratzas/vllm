# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel selection shared by linear-attention ops."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheSpec

logger = init_logger(__name__)


@dataclass(frozen=True)
class LinearAttnSelectorConfig:
    """What a kernel may check, captured without querying the device: the
    platform, the engine config, and the layer's KV cache spec (None when
    unknown)."""

    platform: PlatformEnum
    device_capability: DeviceCapability | None
    vllm_config: "VllmConfig"
    kv_cache_spec: "KVCacheSpec | None" = None

    @classmethod
    def from_current_platform(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "KVCacheSpec | None" = None,
    ) -> "LinearAttnSelectorConfig":
        from vllm.platforms import current_platform

        try:
            capability = current_platform.get_device_capability()
        except NotImplementedError:
            capability = None
        return cls(current_platform._enum, capability, vllm_config, kv_cache_spec)


class LinearAttnKernel:
    """A kernel for one linear-attention op.

    Kernels are stateless classes that are never instantiated, like attention
    backends: layers hold the selected class, and per-layer state reaches it
    through the layer argument.
    """

    name: ClassVar[str]

    @classmethod
    def validate_configuration(cls, cfg: LinearAttnSelectorConfig) -> list[str]:
        """Return why this kernel can't run under `cfg`; empty if it can."""
        return []


def select_linear_attn_kernel(
    op: str,
    kernels: Sequence[type[LinearAttnKernel]],
    cfg: LinearAttnSelectorConfig,
    requested: str | None = None,
    explicit: bool = False,
) -> type[LinearAttnKernel]:
    """Select a kernel for `op` from `kernels`, listed in priority order.

    Args:
        op: The op's name, for messages.
        kernels: The op's kernels, in priority order.
        cfg: What the kernels validate against.
        requested: The requested kernel's name; None selects automatically.
        explicit: Whether `requested` was set for this op, rather than
            inherited from a broader setting. An explicit request that can't
            run is an error; an inherited one falls back with a warning.

    Raises:
        ValueError: If an explicit request is unusable, or no kernel is.

    """
    by_name = {kernel.name: kernel for kernel in kernels}
    if requested is not None:
        kernel = by_name.get(requested)
        if kernel is None:
            reasons = [f"no {requested!r} kernel; options: {', '.join(by_name)}"]
        else:
            reasons = kernel.validate_configuration(cfg)
        if kernel is not None and not reasons:
            logger.info_once("Using %s kernel for %s.", requested, op)
            return kernel
        if explicit:
            raise ValueError(
                f"Backend {requested!r} can't run {op}: {'; '.join(reasons)}"
            )
        logger.warning_once(
            "Backend %r can't run %s (%s); selecting automatically.",
            requested,
            op,
            "; ".join(reasons),
        )

    all_reasons = []
    for kernel in kernels:
        reasons = kernel.validate_configuration(cfg)
        if not reasons:
            logger.info_once("Using %s kernel for %s.", kernel.name, op)
            return kernel
        all_reasons.append(f"{kernel.name}: {'; '.join(reasons)}")
    raise ValueError(f"No kernel can run {op}. {' | '.join(all_reasons)}")
