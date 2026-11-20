# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""(b) Real tuned tables, real device: shapes tuned for the whole chip.

On SPX each shape must hit its own row. On a partition each must hit too: the
partition's own row when the table has one, else the whole chip's row. The
row is identified by its kernel name.
"""

import pandas as pd
import pytest
import torch

from .conftest import tuned_cu_nums


def _dtype(name: str) -> torch.dtype:
    return getattr(torch, str(name).split(".")[-1])


def _ck_a8w8(r):
    from aiter.jit.core import AITER_CONFIGS
    from aiter.ops.gemm_op_a8w8 import get_GEMM_config_with_quant_type

    return get_GEMM_config_with_quant_type(
        int(r.M),
        int(r.N),
        int(r.K),
        _dtype(r.q_dtype_w),
        AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_FILE,
    )


def _ck_a8w8_bpreshuffle(r):
    from aiter.ops.gemm_op_a8w8 import get_GEMM_config_with_quant_type

    return get_GEMM_config_with_quant_type(
        int(r.M), int(r.N), int(r.K), _dtype(r.q_dtype_w)
    )


def _bf16(r, cu_num=None):
    from aiter.tuned_gemm import get_GEMM_A16W16_tuned_config

    return get_GEMM_A16W16_tuned_config(
        int(r["M"]), int(r["N"]), int(r["K"]), bool(r["bias"]), _dtype(r["dtype"]),
        _dtype(r["outdtype"]), bool(r["scaleAB"]), bool(r["bpreshuffle"]), None, cu_num,
    )  # fmt: skip


def _a4w4(r):
    from aiter.ops.gemm_op_a4w4 import get_GEMM_config

    return get_GEMM_config(int(r.M), int(r.N), int(r.K))


# name: (AITER_CONFIGS attribute, shape columns, lookup, PR)
BACKENDS = {
    "ck_a8w8": (
        "AITER_CONFIG_GEMM_A8W8_FILE",
        ["M", "N", "K", "q_dtype_w"],
        _ck_a8w8,
        "ck_gemm",
    ),
    "ck_a8w8_bpreshuffle": (
        "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE",
        ["M", "N", "K", "q_dtype_w"],
        _ck_a8w8_bpreshuffle,
        "ck_gemm",
    ),
    "bf16": (
        "AITER_CONFIG_GEMM_BF16_FILE",
        ["M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle"],
        _bf16,
        "bf16_gemm, opus_tuned, hipblaslt_stale_solidx",
    ),
    "a4w4": ("AITER_CONFIG_GEMM_A4W4_FILE", ["M", "N", "K"], _a4w4, "a4w4_gemm"),
}


def _pick(rows: pd.DataFrame, n: int, by: str | None = None) -> pd.DataFrame:
    """Up to n rows spread over the table, one per `by` value first."""
    if rows.empty:
        return rows
    if by is not None:
        first = rows.groupby(by, sort=True).head(1)
        rest = rows.drop(first.index)
        rows = pd.concat([first, rest])
        if len(first) >= n:
            return first.head(n)
        step = max(1, len(rest) // (n - len(first)))
        return pd.concat([first, rest.iloc[::step].head(n - len(first))])
    step = max(1, len(rows) // n)
    return rows.iloc[::step].head(n)


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_full_chip_shapes_find_a_row(backend, chip_info, expected, aiter_log):
    from aiter.jit.core import AITER_CONFIGS

    attr, keys, lookup, pr = BACKENDS[backend]
    table = pd.read_csv(getattr(AITER_CONFIGS, attr)).drop_duplicates()
    table = table[table.gfx == expected.gfx]
    full = table[table.cu_num == expected.full_chip_cu]
    if full.empty:
        pytest.skip(f"{attr}: no {expected.gfx} rows for {expected.full_chip_cu} CUs")
    own = table[table.cu_num == expected.cu].set_index(keys)
    own = own[~own.index.duplicated()]
    full = full.drop_duplicates(keys)
    has_own = full.set_index(keys).index.isin(own.index)
    by = "libtype" if "libtype" in full.columns else None
    if expected.is_partition:
        shapes = pd.concat([_pick(full[~has_own], 6, by), _pick(full[has_own], 2, by)])
    else:
        shapes = _pick(full, 6, by)

    print(f"\n{backend} (PR {pr}), {expected.gfx}, own {expected.cu} CUs, "
          f"chip {expected.full_chip_cu} CUs, tuned CU counts "
          f"{tuned_cu_nums(chip_info)}")  # fmt: skip
    print(f"{'shape':<44} {'want cu':>7} {'result':<10} kernel")
    misses, checked = [], 0
    for _, r in shapes.iterrows():
        key = tuple(r[k] for k in keys)
        want_cu = expected.cu if key in own.index else expected.full_chip_cu
        want = own.loc[key] if want_cu == expected.cu else r
        want_name = str(want["kernelName"])
        if backend == "bf16" and _bf16(r, cu_num=want_cu) is None:
            print(f"{str(key):<44} {want_cu:>7} {'n/a':<10} row rejected on its own "
                  f"CU count too (catalog/solidx check): {want_name}")  # fmt: skip
            continue
        checked += 1
        row = lookup(r)
        got = None if row is None else str(row["kernelName"])
        status = "hit" if got == want_name else ("MISS" if got is None else "OTHER")
        print(f"{str(key):<44} {want_cu:>7} {status:<10} {got or want_name}")
        if status != "hit":
            misses.append((key, want_cu, want_name, got))
    for message in aiter_log:
        if "full chip" in message:
            print(f"aiter log: {message}")
    if not checked:
        pytest.skip("no usable shape")
    assert not misses, f"{len(misses)}/{checked} shapes found no or another row"
