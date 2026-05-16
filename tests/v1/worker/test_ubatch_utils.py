# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np

from vllm import forward_context
from vllm.v1.worker.ubatching import UBatchContext
from vllm.v1.worker.ubatch_utils import (
    can_create_request_aligned_ubatch_slices,
    maybe_create_ubatch_slices,
)


def test_ubatch_slices_use_request_boundaries_for_mixed_prefill_batch():
    # Regression shape from DP+EP+DBO DeepSeek-V2-Lite failures:
    # one long fresh prefill followed by two chunked prefill continuations.
    # MLA chunked-prefill metadata is request scoped, so DBO must not split a
    # request token span even when the token-balanced split point lands inside
    # the long first request.
    num_scheduled_tokens = np.array([751, 59, 77], dtype=np.int32)

    assert can_create_request_aligned_ubatch_slices(
        num_scheduled_tokens,
        num_tokens_padded=1024,
        num_ubatches=2,
    )

    ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
        should_ubatch=True,
        num_scheduled_tokens=num_scheduled_tokens,
        num_tokens_padded=1024,
        num_reqs_padded=4,
        num_ubatches=2,
    )

    assert ubatch_slices is not None
    assert [s.token_slice for s in ubatch_slices] == [
        slice(0, 751),
        slice(751, 887),
    ]
    assert [s.request_slice for s in ubatch_slices] == [
        slice(0, 1),
        slice(1, 3),
    ]

    assert ubatch_slices_padded is not None
    assert ubatch_slices_padded[-1].token_slice == slice(751, 1024)
    assert ubatch_slices_padded[-1].request_slice == slice(1, 4)


def test_ubatch_slices_decline_when_no_request_boundary_can_split_batch():
    num_scheduled_tokens = np.array([887], dtype=np.int32)

    assert not can_create_request_aligned_ubatch_slices(
        num_scheduled_tokens,
        num_tokens_padded=1024,
        num_ubatches=2,
    )

    ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
        should_ubatch=True,
        num_scheduled_tokens=num_scheduled_tokens,
        num_tokens_padded=1024,
        num_reqs_padded=1,
        num_ubatches=2,
    )

    assert ubatch_slices is None
    assert ubatch_slices_padded is None


def test_ubatch_restore_context_installs_thread_local_mla_prefill_metadata():
    class Backend:
        def __init__(self):
            self.metadata = None

        def prepare_metadata(self, metadata):
            self.metadata = metadata

    backend = Backend()
    prefill_metadata = SimpleNamespace(prefill_backend=backend)
    attn_metadata = {"layer.0.attn": SimpleNamespace(prefill=prefill_metadata)}

    old_forward_context = forward_context._forward_context
    ctx = object.__new__(UBatchContext)
    ctx.forward_context = SimpleNamespace(attn_metadata=attn_metadata)
    try:
        ctx._restore_context()
        assert backend.metadata is prefill_metadata
    finally:
        forward_context._forward_context = old_forward_context
