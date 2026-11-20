# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""(d) test_online_mxfp4_hadamard[aiter-asm] without forcing the tuned checks.

#68188 forces is_asm_fp4_gemm_tuned because a DPX MI355 found no A4W4 rows.
With the whole-chip fallback the real check must pass and the layer must take
the ASM path, which fuses the Hadamard rotation.
"""

import pytest
import torch

from tests.quantization.test_input_hadamard import _online_mxfp4
from tests.single_rank_group import SingleRankGroup, override_parallel_groups
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.model import ModelConfig
from vllm.model_executor.kernels.linear.mxfp4.aiter import AiterMxfp4LinearKernel
from vllm.model_executor.layers.quantization.input_hadamard import (
    block_hadamard_rotate,
    hadamard_matrix,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


@pytest.mark.parametrize("num_tokens", [16, 256])
@pytest.mark.skip_global_cleanup
@override_parallel_groups(SingleRankGroup())
def test_online_mxfp4_hadamard_asm_unforced(
    num_tokens, default_vllm_config, monkeypatch
):
    supported, reason = AiterMxfp4LinearKernel.is_supported()
    if not supported:
        pytest.skip(reason)
    # CI does not set VLLM_ROCM_USE_AITER; the upstream test forces it too.
    monkeypatch.setattr(rocm_aiter_ops, "_AITER_ENABLED", True)
    if not rocm_aiter_ops.is_asm_fp4_gemm_supported():
        pytest.skip("AITER's ASM FP4 GEMM needs gfx950.")
    input_size, output_size = 1536, 3072
    asm_tuned = rocm_aiter_ops.is_asm_fp4_gemm_tuned(output_size, input_size // 2)
    triton_tuned = rocm_aiter_ops.is_triton_gemm_afp4wfp4_presh_ws_tuned(
        output_size, input_size // 2
    )
    print(
        f"\nCUs {torch.cuda.get_device_properties(0).multi_processor_count}: "
        f"is_asm_fp4_gemm_tuned={asm_tuned} "
        f"is_triton_gemm_afp4wfp4_presh_ws_tuned={triton_tuned}"
    )
    assert asm_tuned and triton_tuned

    default_vllm_config.model_config = ModelConfig()
    set_random_seed(0)
    with torch.device(current_platform.device_type):
        x = torch.randn(num_tokens, input_size, dtype=torch.bfloat16)
        x[:, torch.randperm(input_size)[:16]] *= 50.0
        weight = torch.randn(output_size, input_size, dtype=torch.bfloat16) * 0.02
        weight[:, torch.randperm(input_size)[:16]] *= 10.0
        want = x.float() @ weight.float().T
        kernel = AiterMxfp4LinearKernel
        _, plain = _online_mxfp4(weight, x, kernel, None)
        layer, rotated = _online_mxfp4(weight, x, kernel, 32)
        h = hadamard_matrix(32, x.device)
        _, explicit = _online_mxfp4(
            block_hadamard_rotate(weight, h), block_hadamard_rotate(x, h), kernel, None
        )

    def rel_err(y: torch.Tensor) -> float:
        return ((y.float() - want).norm() / want.norm()).item()

    print(f"rel err plain {rel_err(plain):.3e} rotated {rel_err(rotated):.3e}")
    assert layer.input_hadamard_fused, "the layer did not take the ASM path"
    assert rel_err(rotated) < 0.9 * rel_err(plain)
    assert ((rotated.float() - explicit.float()).norm() / want.norm()).item() < 1e-2
