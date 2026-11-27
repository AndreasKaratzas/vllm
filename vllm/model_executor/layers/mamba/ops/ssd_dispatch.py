# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Kernels for the Mamba2 SSD prefill scan, and their selection."""

import functools
from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.linear_attention.kernel import (
    LinearAttnKernel,
    LinearAttnSelectorConfig,
    select_linear_attn_kernel,
)
from vllm.platforms.interface import PlatformEnum
from vllm.v1.kv_cache_interface import MambaSpec

from .gather_initial_states import gather_initial_states
from .scatter_states import scatter_states

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
    from vllm.v1.attention.backends.mamba2_attn import (
        Mamba2AttentionMetadata,
        SSDMetadataBuilder,
    )


# Shapes and activation dtype FlashInfer's SSD scan is built for.
FLASHINFER_SSD_HEAD_DIM = 64
FLASHINFER_SSD_STATE_SIZE = 128
FLASHINFER_SSD_DTYPE = torch.bfloat16


class MambaSSDKernel(LinearAttnKernel):
    """The Mamba2 SSD prefill scan, including its state cache reads and writes."""

    @classmethod
    def make_metadata_builder(cls, kv_cache_spec: MambaSpec) -> "SSDMetadataBuilder":
        from vllm.v1.attention.backends import mamba2_attn

        return mamba2_attn.SSDMetadataBuilder(kv_cache_spec)

    @classmethod
    def read_initial_states(
        cls, ssm_state: torch.Tensor, metadata: "Mamba2AttentionMetadata"
    ) -> torch.Tensor | None:
        """Each prefill's cached state, zeros where it has none."""
        if metadata.has_initial_states_p is None or not metadata.prep_initial_states:
            return metadata.zero_initial_states_p
        assert metadata.state_indices_tensor_p is not None
        return cls.gather_states(
            ssm_state, metadata.state_indices_tensor_p, metadata.has_initial_states_p
        )

    @classmethod
    def gather_states(
        cls,
        ssm_state: torch.Tensor,
        indices: torch.Tensor,
        has_initial_states: torch.Tensor,
    ) -> torch.Tensor:
        return gather_initial_states(ssm_state, indices, has_initial_states)

    @classmethod
    def write_final_states(
        cls,
        ssm_state: torch.Tensor,
        final_states: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        scatter_states(ssm_state, final_states, indices)

    @classmethod
    def forward(
        cls,
        layer: "MambaMixer2",
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        out: torch.Tensor,
        ssm_state: torch.Tensor,
        metadata: "Mamba2AttentionMetadata",
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Scan the prefills from their cached states into `out`, and cache
        their final states.

        Returns:
            The checkpoint states and each prefill's row in them, or None when
            no checkpoints are exported.

        """
        assert metadata.state_indices_tensor_p is not None
        final_states, checkpoint_states = cls._scan(
            layer,
            x,
            dt,
            B,
            C,
            out,
            ssm_state,
            metadata,
            cls.read_initial_states(ssm_state, metadata),
        )
        if final_states is not None:
            cls.write_final_states(
                ssm_state, final_states, metadata.state_indices_tensor_p
            )
        if checkpoint_states is None:
            return None
        assert metadata.checkpoint_chunk_idx is not None
        return checkpoint_states, metadata.checkpoint_chunk_idx

    @classmethod
    def _scan(
        cls,
        layer: "MambaMixer2",
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        out: torch.Tensor,
        ssm_state: torch.Tensor,
        metadata: "Mamba2AttentionMetadata",
        initial_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return the final states, or None if already written into
        `ssm_state`, and the checkpoint states, or None when none are
        exported."""
        raise NotImplementedError


class TritonSSDKernel(MambaSSDKernel):
    name = "triton"

    @classmethod
    def validate_configuration(cls, cfg: LinearAttnSelectorConfig) -> list[str]:
        if cfg.platform == PlatformEnum.CPU:
            return ["Triton SSD does not run on CPU; use cpu"]
        return []

    @staticmethod
    @functools.cache
    def _fn():
        from .ssd_combined import mamba_chunk_scan_combined_varlen

        return mamba_chunk_scan_combined_varlen

    @classmethod
    def _scan(
        cls,
        layer: "MambaMixer2",
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        out: torch.Tensor,
        ssm_state: torch.Tensor,
        metadata: "Mamba2AttentionMetadata",
        initial_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # With checkpoints the kernel returns every chunk's state; otherwise it
        # writes the final states itself.
        exports = metadata.checkpoint_chunk_idx is not None
        states = cls._fn()(
            x,
            dt,
            layer.A,
            B,
            C,
            metadata.chunk_size,
            cu_seqlens=metadata.query_start_loc_p,
            cu_chunk_seqlens=metadata.cu_chunk_seqlen_p,
            last_chunk_indices=metadata.last_chunk_indices_p,
            seq_idx=metadata.seq_idx_p,
            out=out,
            D=layer.D,
            dt_bias=layer.dt_bias,
            initial_states=initial_states,
            dt_softplus=True,
            return_intermediate_states=exports,
            state_dtype=ssm_state.dtype,
            final_states_out=None if exports else ssm_state,
            final_state_indices=None if exports else metadata.state_indices_tensor_p,
        )
        if not exports:
            return None, None
        return states[metadata.last_chunk_indices_p], states


class FlashInferSSDKernel(MambaSSDKernel):
    name = "flashinfer"

    @classmethod
    def validate_configuration(cls, cfg: LinearAttnSelectorConfig) -> list[str]:
        reasons = []
        if cfg.platform != PlatformEnum.CUDA:
            reasons.append("FlashInfer SSD requires CUDA")
        capability = cfg.device_capability
        if capability is None or capability.major != 10:
            found = "unknown" if capability is None else capability.as_version_str()
            reasons.append(f"FlashInfer SSD requires SM10x, compute capability {found}")
        if isinstance(cfg.kv_cache_spec, MambaSpec):
            _, head_dim, state_size = cfg.kv_cache_spec.shapes[1]
            if head_dim != FLASHINFER_SSD_HEAD_DIM:
                reasons.append(
                    f"FlashInfer SSD requires head_dim {FLASHINFER_SSD_HEAD_DIM}, "
                    f"got {head_dim}"
                )
            if state_size != FLASHINFER_SSD_STATE_SIZE:
                reasons.append(
                    "FlashInfer SSD requires ssm_state_size "
                    f"{FLASHINFER_SSD_STATE_SIZE}, got {state_size}"
                )
        model_config = cfg.vllm_config.model_config
        dtype = model_config.dtype if model_config is not None else None
        if dtype not in (None, FLASHINFER_SSD_DTYPE):
            reasons.append(f"FlashInfer SSD requires bfloat16 activations, got {dtype}")
        try:
            from flashinfer.mamba import ssd_combined_fwd  # noqa: F401
        except ImportError:
            reasons.append("flashinfer-python with ssd_combined_fwd is not installed")
        return reasons

    @classmethod
    def make_metadata_builder(cls, kv_cache_spec: MambaSpec) -> "SSDMetadataBuilder":
        from vllm.v1.attention.backends import mamba2_attn

        return mamba2_attn.FlashInferSSDMetadataBuilder(kv_cache_spec)

    @staticmethod
    @functools.cache
    def _fn():
        from flashinfer.mamba import ssd_combined_fwd

        return ssd_combined_fwd

    @classmethod
    def _scan(
        cls,
        layer: "MambaMixer2",
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        out: torch.Tensor,
        ssm_state: torch.Tensor,
        metadata: "Mamba2AttentionMetadata",
        initial_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # FlashInfer takes the state dtype from the initial states, so batches
        # without cached state pass zeros.
        _, final_states = cls._fn()(
            x.unsqueeze(0),
            dt.unsqueeze(0),
            layer.A,
            B.unsqueeze(0),
            C.unsqueeze(0),
            D=layer.D,
            dt_bias=layer.dt_bias,
            dt_softplus=True,
            initial_states=initial_states,
            cu_seqlens=metadata.query_start_loc_p,
            checkpoint_token_indices=metadata.checkpoint_token_indices_p,
            checkpoint_state_slots=metadata.checkpoint_chunk_idx,
            checkpoint_states=metadata.checkpoint_states_p,
            out=out.unsqueeze(0),
        )
        return final_states, metadata.checkpoint_states_p


class CPUSSDKernel(MambaSSDKernel):
    name = "cpu"

    @classmethod
    def validate_configuration(cls, cfg: LinearAttnSelectorConfig) -> list[str]:
        if cfg.platform != PlatformEnum.CPU:
            return ["the cpu SSD kernel runs only on CPU"]
        return []

    @staticmethod
    @functools.cache
    def _fn():
        from .cpu.mamba_ssm import _mamba_chunk_scan_combined_fwd_cpu

        return _mamba_chunk_scan_combined_fwd_cpu

    @classmethod
    def gather_states(
        cls,
        ssm_state: torch.Tensor,
        indices: torch.Tensor,
        has_initial_states: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(
            has_initial_states[:, None, None, None], ssm_state[indices], 0
        )

    @classmethod
    def write_final_states(
        cls,
        ssm_state: torch.Tensor,
        final_states: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        ssm_state[indices] = final_states

    @classmethod
    def _scan(
        cls,
        layer: "MambaMixer2",
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        out: torch.Tensor,
        ssm_state: torch.Tensor,
        metadata: "Mamba2AttentionMetadata",
        initial_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        assert metadata.checkpoint_chunk_idx is None, (
            "The CPU SSD kernel cannot export prefill checkpoints."
        )
        final_states = cls._fn()(
            x,
            dt,
            layer.A,
            B,
            C,
            metadata.chunk_size,
            out,
            D=layer.D,
            dt_bias=layer.dt_bias,
            initial_states=initial_states,
            cu_seqlens=metadata.query_start_loc_p,
            dt_softplus=True,
            state_dtype=ssm_state.dtype,
        )
        return final_states, None


# Priority order for automatic selection.
SSD_KERNELS: tuple[type[MambaSSDKernel], ...] = (
    TritonSSDKernel,
    CPUSSDKernel,
    FlashInferSSDKernel,
)


def select_mamba2_ssd_kernel(
    vllm_config: "VllmConfig", kv_cache_spec: MambaSpec | None
) -> type[MambaSSDKernel]:
    """Select the SSD kernel for a layer from `--mamba-prefill-backend`, else
    `--mamba-backend`, else automatically."""
    mamba_config = vllm_config.mamba_config
    requested = mamba_config.effective_prefill_backend
    kernel = select_linear_attn_kernel(
        "Mamba2 SSD",
        SSD_KERNELS,
        LinearAttnSelectorConfig.from_current_platform(vllm_config, kv_cache_spec),
        requested.value if requested is not None else None,
        explicit=mamba_config.prefill_backend is not None,
    )
    assert issubclass(kernel, MambaSSDKernel)
    return kernel
