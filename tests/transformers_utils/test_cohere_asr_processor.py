# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.transformers_utils.processors import cohere_asr
from vllm.transformers_utils.processors.cohere_asr import FilterbankFeatures


def test_cohere_asr_dither_is_batch_invariant():
    featurizer = FilterbankFeatures(
        sample_rate=16000,
        n_window_size=16,
        n_window_stride=8,
        nfilt=4,
        n_fft=16,
        pad_to=0,
        dither=1e-5,
    )

    single = torch.zeros(1, 8)
    single_len = torch.tensor([4])
    batch = torch.zeros(2, 8)
    batch_len = torch.tensor([4, 7])

    single_out = featurizer._apply_dither(single.clone(), single_len)
    batch_out = featurizer._apply_dither(batch.clone(), batch_len)

    torch.testing.assert_close(batch_out[0], single_out[0])
    assert torch.count_nonzero(batch_out[0, 4:]) == 0
    assert torch.count_nonzero(batch_out[1, 7:]) == 0


def test_cohere_asr_filterbank_matches_librosa_when_available():
    librosa = cohere_asr.librosa
    if librosa is None:
        pytest.skip("librosa is not installed")

    featurizer = FilterbankFeatures(
        sample_rate=16000,
        n_window_size=16,
        n_window_stride=8,
        nfilt=4,
        n_fft=16,
        pad_to=0,
        dither=0.0,
    )

    expected = torch.tensor(
        librosa.filters.mel(
            sr=16000,
            n_fft=16,
            n_mels=4,
            fmin=0,
            fmax=8000,
            norm="slaney",
        ),
        dtype=torch.float,
    ).unsqueeze(0)

    torch.testing.assert_close(
        featurizer.fb.float(),
        expected,
        atol=5e-5,
        rtol=1e-2,
    )
