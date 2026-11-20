# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""(c) Kernels that partitions now run with whole-chip configs, vs torch.

Shapes come from the whole-chip rows of the tuned tables, so on a partition
these launch the configs that the fallback selects.
"""

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from vllm.utils.torch_utils import set_random_seed

from .conftest import full_chip_fallbacks


def _rel(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.float() - ref.float()).norm() / ref.float().norm()).item()


def _full_chip_rows(attr: str, expected, max_mnk: int = 2**32) -> pd.DataFrame:
    from aiter.jit.core import AITER_CONFIGS

    t = pd.read_csv(getattr(AITER_CONFIGS, attr)).drop_duplicates()
    t = t[(t.gfx == expected.gfx) & (t.cu_num == expected.full_chip_cu)]
    return t[max_mnk >= t.M * t.N * t.K]


def _report(name, shape, err, tol, aiter_log):
    print(f"\n{name} {shape}: rel err {err:.2e} (tol {tol:.0e})")
    for message in full_chip_fallbacks(aiter_log):
        print(f"aiter log: {message}")


@pytest.mark.parametrize("bpreshuffle", [False, True], ids=["ck", "bpreshuffle"])
def test_ck_gemm_a8w8(bpreshuffle, expected, aiter_log):
    """A whole-chip tuned shape, or a fixed one (heuristic dispatch) if none."""
    import aiter
    from aiter.ops.shuffle import shuffle_weight

    attr = "AITER_CONFIG_GEMM_A8W8_" + ("BPRESHUFFLE_FILE" if bpreshuffle else "FILE")
    rows = _full_chip_rows(attr, expected, 2**36)
    # gemm_a8w8_bpreshuffle runs INT8 on gfx950 only.
    rows = rows[rows.q_dtype_w == str(aiter.dtypes.fp8)]
    if rows.empty:
        m, n, k, qdtype, name = 128, 1280, 8192, aiter.dtypes.fp8, "untuned"
    else:
        r = rows.sort_values("M").iloc[len(rows) // 2]
        m, n, k, name = int(r.M), int(r.N), int(r.K), r.kernelName
        qdtype = getattr(torch, r.q_dtype_w.split(".")[-1])
    set_random_seed(0)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    xq, xs = aiter.pertoken_quant(x, quant_dtype=qdtype)
    wq, ws = aiter.pertoken_quant(w, quant_dtype=qdtype)
    ref = (xq.float() * xs) @ (wq.float() * ws).T
    if bpreshuffle:
        wq = shuffle_weight(wq, layout=(16, 16))
        out = aiter.gemm_a8w8_bpreshuffle(xq, wq, xs, ws, None, torch.bfloat16)
    else:
        out = aiter.gemm_a8w8_CK(xq, wq, xs, ws, None, torch.bfloat16)
    err = _rel(out, ref)
    _report(f"A8W8 {qdtype} {name}", (m, n, k), err, 1e-2, aiter_log)
    assert err < 1e-2


def test_bf16_tuned_gemm(expected, aiter_log):
    from aiter.tuned_gemm import gemm_a16w16

    rows = _full_chip_rows("AITER_CONFIG_GEMM_BF16_FILE", expected, 2**34)
    rows = rows[
        ~rows.bias.astype(bool)
        & ~rows.scaleAB.astype(bool)
        & ~rows.bpreshuffle.astype(bool)
        & (rows.dtype == "torch.bfloat16")
        & (rows.outdtype == "torch.bfloat16")
    ]
    if rows.empty:
        pytest.skip(f"no {expected.gfx} BF16 rows for {expected.full_chip_cu} CUs")
    # One shape per library (hipblaslt, asm, opus, flydsl, triton, ...).
    picks = rows.sort_values("M").groupby("libtype").head(1)
    failures = []
    for _, r in picks.iterrows():
        m, n, k = int(r.M), int(r.N), int(r.K)
        set_random_seed(0)
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / k**0.5
        err = _rel(
            gemm_a16w16(x, w, otype=torch.bfloat16), F.linear(x.float(), w.float())
        )
        _report(f"BF16 {r.libtype} {r.kernelName}", (m, n, k), err, 1e-2, aiter_log)
        if not err < 1e-2:
            failures.append((r.libtype, (m, n, k), err))
    assert not failures


def test_fused_moe_bf16(expected, aiter_log):
    """A bf16 MoE shape tuned for every whole chip (CK 2-stage on gfx942/950)."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe, torch_moe
    from aiter.ops.shuffle import shuffle_weight

    tokens, dim, inter, experts, topk = 32, 4096, 384, 128, 8
    set_random_seed(0)
    dev, bf16 = "cuda", torch.bfloat16
    x = torch.randn(tokens, dim, dtype=bf16, device=dev)
    w1 = torch.randn(experts, 2 * inter, dim, dtype=bf16, device=dev) / dim**0.5
    w2 = torch.randn(experts, dim, inter, dtype=bf16, device=dev) / inter**0.5
    logits = torch.randn(tokens, experts, device=dev)
    topk_weight, topk_ids = torch.topk(logits.softmax(-1), topk, dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    ref = torch_moe(x, w1, w2, topk_weight, topk_ids, activation=ActivationType.Silu)
    out = fused_moe(
        x,
        shuffle_weight(w1, layout=(16, 16)),
        shuffle_weight(w2, layout=(16, 16)),
        topk_weight,
        topk_ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.No,
    )
    err = _rel(out, ref)
    _report("fused_moe bf16", (tokens, dim, inter, experts, topk), err, 2e-2, aiter_log)
    assert err < 2e-2
    if expected.is_partition:
        assert any("fused_moe" in m for m in full_chip_fallbacks(aiter_log)), (
            "fused_moe did not use the whole chip's rows on this partition"
        )


def test_a4w4_gemm(expected, aiter_log):
    if expected.gfx != "gfx950":
        pytest.skip("A4W4 ASM GEMM is gfx950 only")
    import aiter
    from aiter.ops.gemm_op_a4w4 import gemm_a4w4
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility import fp4_utils

    rows = _full_chip_rows("AITER_CONFIG_GEMM_A4W4_FILE", expected, 2**33)
    rows = rows[rows.kernelName.astype(str).str.startswith("_ZN")]
    if rows.empty:
        pytest.skip(f"no A4W4 ASM rows for {expected.full_chip_cu} CUs")
    r = rows.sort_values("M").iloc[len(rows) // 2]
    m, n, k = int(r.M), int(r.N), int(r.K)
    quant = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    set_random_seed(0)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    _, xs = quant(x, shuffle=False)
    _, ws = quant(w, shuffle=False)
    xq, xs_shuffled = quant(x, shuffle=True)
    wq, ws_shuffled = quant(w, shuffle=True)

    def dequant(q, s, rows_):
        s = s.view(torch.uint8)[:rows_].repeat_interleave(32, dim=1)
        return fp4_utils.mxfp4_to_f32(q) * fp4_utils.e8m0_to_f32(s)

    ref = dequant(xq, xs, m) @ dequant(wq, ws, n).T
    out = gemm_a4w4(
        xq, shuffle_weight(wq, layout=(16, 16)), xs_shuffled, ws_shuffled,
        dtype=torch.bfloat16,
    )  # fmt: skip
    err = _rel(out[:m], ref)
    _report(f"A4W4 {r.kernelName}", (m, n, k), err, 1e-2, aiter_log)
    assert err < 1e-2
