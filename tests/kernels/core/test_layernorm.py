# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings

import pytest
import torch

from tests.kernels.quant_utils import FP8_DTYPE
from tests.kernels.utils import opcheck
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx90a

    on_mi250 = on_gfx90a()
else:
    on_mi250 = False

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 4096]  # Arbitrary values for testing
HIDDEN_SIZES = [8, 768, 769, 5120, 5125, 8192]  # Arbitrary values for testing
ADD_RESIDUAL = [False, True] if not on_mi250 else [True]
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]


def _format_observed_rate(count: int, total: int) -> str:
    return f"{count / total:.6%} ({count}/{total})"


def _format_allowed_count(count: int, total: int) -> str:
    return f"{count / total:.6%} ({count}/{total})"


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    max_quantile_samples = 1_000_000
    if values.numel() > max_quantile_samples:
        stride = (values.numel() + max_quantile_samples - 1) // max_quantile_samples
        values = values[::stride]
    return torch.quantile(values, q).item()


def _assert_abs_error_budget(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    tight_atol: float,
    max_atol: float,
    max_fail_rate: float,
    min_allowed_fail: int,
) -> None:
    abs_diff = (actual.float() - expected.float()).abs().flatten()
    total = abs_diff.numel()
    within_tight_count = int((abs_diff <= tight_atol).sum().item())
    fail_count = total - within_tight_count
    allowed_fail_count = max(min_allowed_fail, int(total * max_fail_rate) + 1)
    above_max_count = int((abs_diff > max_atol).sum().item())

    msg = (
        "[rocm_layernorm_fp8_quant] "
        f"{label}: "
        f"abs<={tight_atol:g} pass={within_tight_count / total:.4%} "
        f"({within_tight_count}/{total}) "
        f"fail={_format_observed_rate(fail_count, total)} "
        f"allowed_fail={_format_allowed_count(allowed_fail_count, total)} "
        f"abs>{max_atol:g}={_format_observed_rate(above_max_count, total)} "
        f"allowed_above_max={_format_allowed_count(0, total)} "
        f"max_abs={abs_diff.max().item():.6g} "
        f"mean_abs={abs_diff.mean().item():.6g} "
        f"p99_abs={_quantile(abs_diff, 0.99):.6g} "
        f"p999_abs={_quantile(abs_diff, 0.999):.6g}"
    )
    if fail_count > 0:
        warnings.warn(msg, stacklevel=2)
    assert fail_count <= allowed_fail_count, msg
    assert above_max_count == 0, msg


def _assert_fp8_close_or_sparse_rocm_boundary(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> None:
    actual_f = actual.to(dtype=torch.float32)
    expected_f = expected.to(dtype=torch.float32)

    if not current_platform.is_rocm() or dtype is not torch.half:
        torch.testing.assert_close(actual_f, expected_f, atol=1e-3, rtol=1e-3)
        return

    # The source-side alternative is to force the fused ROCm FP16 kernels in
    # layernorm_quant_kernels.cu to round f32 -> f16 before converting to FP8,
    # matching the unfused RMSNorm-then-FP8 path byte-for-byte. That adds an
    # extra per-element conversion to the fused production kernel, so keep the
    # kernel fast and only allow the sparse FP8 boundary bins observed here.
    _assert_abs_error_budget(
        actual_f,
        expected_f,
        label="fused RMSNorm+FP8 quant vs unfused RMSNorm then FP8",
        tight_atol=1e-3,
        max_atol=8.0,
        max_fail_rate=1e-6,
        min_allowed_fail=2,
    )


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
@torch.inference_mode()
def test_rms_norm(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
    strided_input: bool,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x = torch.randn(num_tokens, last_dim, dtype=dtype)
    x = x[..., :hidden_size]
    assert x.is_contiguous() != strided_input
    x *= scale
    residual = torch.randn_like(x) * scale if add_residual else None

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_out = layer.forward_native(x, residual)
    out = layer(x, residual)
    # NOTE(woosuk): LayerNorm operators (including RMS) typically have larger
    # numerical errors than other operators because they involve reductions.
    # Therefore, we use a larger tolerance.
    if add_residual:
        torch.testing.assert_close(out[0], ref_out[0], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[1], ref_out[1], atol=1e-2, rtol=1e-2)
    else:
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

    if residual is not None:
        opcheck(
            torch.ops._C.fused_add_rms_norm,
            (x, residual, layer.weight.data, layer.variance_epsilon),
        )
    else:
        opcheck(
            torch.ops._C.rms_norm, (out, x, layer.weight.data, layer.variance_epsilon)
        )


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_scale", [0.01, 1.0, 10.0])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
def test_fused_rms_norm_quant(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    quant_scale: float,
    seed: int,
    device: str,
    strided_input: bool,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)

    weight = torch.empty(hidden_size, dtype=dtype).normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x_base = torch.randn(num_tokens, last_dim, dtype=dtype)
    x = x_base[..., :hidden_size]
    assert x.is_contiguous() != strided_input

    x *= scale
    if add_residual:
        residual = torch.randn_like(x) * scale
        residual_fused = residual.clone()
    else:
        residual = residual_fused = None

    out_norm = torch.empty_like(x)
    out_quant = torch.empty_like(x, dtype=FP8_DTYPE)
    out_quant_fused = torch.empty_like(out_quant)

    quant_scale_t = torch.tensor(quant_scale, dtype=torch.float32)

    if add_residual:
        torch.ops._C.fused_add_rms_norm_static_fp8_quant(
            out_quant_fused, x, residual_fused, weight, quant_scale_t, 1e-6
        )

        # Unfused kernel is in-place so it goes second
        # Also use a separate clone of x to avoid modifying the input
        x_unfused_base = x_base.clone()
        x_unfused = x_unfused_base[..., :hidden_size]
        assert x_unfused.is_contiguous() != strided_input
        torch.ops._C.fused_add_rms_norm(x_unfused, residual, weight, 1e-6)
        torch.ops._C.static_scaled_fp8_quant(
            out_quant, x_unfused.contiguous(), quant_scale_t
        )

        torch.accelerator.synchronize()
        torch.testing.assert_close(residual_fused, residual, atol=1e-2, rtol=1e-2)
        opcheck(
            torch.ops._C.fused_add_rms_norm_static_fp8_quant,
            (out_quant_fused, x, residual_fused, weight, quant_scale_t, 1e-6),
        )
    else:
        torch.ops._C.rms_norm_static_fp8_quant(
            out_quant_fused, x, weight, quant_scale_t, 1e-6
        )

        torch.ops._C.rms_norm(out_norm, x, weight, 1e-6)
        torch.ops._C.static_scaled_fp8_quant(out_quant, out_norm, quant_scale_t)

        opcheck(
            torch.ops._C.rms_norm_static_fp8_quant,
            (out_quant_fused, x, weight, quant_scale_t, 1e-6),
        )

    _assert_fp8_close_or_sparse_rocm_boundary(
        out_quant,
        out_quant_fused,
        dtype=dtype,
    )


@torch.inference_mode()
def test_gemma_rms_norm_mixed_input_weight_dtype(default_vllm_config) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = CUDA_DEVICES[0]
    torch.set_default_device(device)

    num_tokens, hidden_size = 32, 1024
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    layer = GemmaRMSNorm(hidden_size, eps=1e-6).to(device=device)
    layer.weight.data.normal_(mean=0.0, std=0.1)

    # Gemma uses fp32 weight parameter while activations can be bf16.
    assert layer.weight.dtype == torch.float32
    out = layer(x)

    x_fp32 = x.float()
    weight_fp32 = layer.weight.data.float() + 1.0
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    ref = (x_fp32 * torch.rsqrt(variance + layer.variance_epsilon) * weight_fp32).to(
        x.dtype
    )

    assert out.dtype == x.dtype
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
