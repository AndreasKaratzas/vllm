# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import numpy as np
from PIL import Image
from transformers.processing_utils import ProcessingKwargs
from typing_extensions import Unpack

from vllm.transformers_utils.processor import (
    get_processor_kwargs_keys,
    get_processor_kwargs_type,
)
from vllm.transformers_utils.processors.pixtral import MistralCommonImageProcessor
from vllm.transformers_utils.processors.voxtral import MistralCommonFeatureExtractor


class _FakeProcessorKwargs(ProcessingKwargs, total=False):  # type: ignore
    pass


def _assert_has_all_expected(keys: set[str]) -> None:
    # text
    for k in ("text_pair", "text_target", "text_pair_target"):
        assert k in keys
    # image
    for k in ("do_convert_rgb", "do_resize"):
        assert k in keys
    # audio
    for k in (
        "fps",
        "do_sample_frames",
        "input_data_format",
        "default_to_square",
    ):
        assert k in keys
    # audio
    for k in ("padding", "return_attention_mask"):
        assert k in keys


# Path 1: __call__ method has kwargs: Unpack[*ProcessorKwargs]
class _ProcWithUnpack:
    def __call__(self, *args, **kwargs: Unpack[_FakeProcessorKwargs]):  # type: ignore
        return None


def test_get_processor_kwargs_from_processor_unpack_path_returns_full_union():
    proc = _ProcWithUnpack()
    keys = get_processor_kwargs_keys(get_processor_kwargs_type(proc))
    _assert_has_all_expected(keys)


# ---- Path 2: No Unpack, fallback to scanning *ProcessorKwargs in module ----


class _ProcWithoutUnpack:
    def __call__(self, *args, **kwargs):
        return None


def test_get_processor_kwargs_from_processor_module_scan_returns_full_union():
    # ensure the module scanned by fallback is this test module
    module_name = _ProcWithoutUnpack.__module__
    mod = importlib.import_module(module_name)
    assert hasattr(mod, "_FakeProcessorKwargs")

    proc = _ProcWithoutUnpack()
    keys = get_processor_kwargs_keys(get_processor_kwargs_type(proc))
    _assert_has_all_expected(keys)


class _FakeAudioConfig:
    sampling_rate = 16000
    frame_rate = 50
    is_streaming = True


class _FakeAudioEncoder:
    audio_config = _FakeAudioConfig()

    def pad(self, audio, sampling_rate):
        return audio


def test_voxtral_feature_extractor_fetch_audio_accepts_hf_processor_inputs():
    extractor = MistralCommonFeatureExtractor(_FakeAudioEncoder())
    audio = np.zeros(16, dtype=np.float32)

    assert extractor.fetch_audio(audio) is audio
    assert extractor.fetch_audio([audio])[0] is audio
    assert extractor.fetch_audio([[audio]])[0][0] is audio

    try:
        extractor.fetch_audio(object())
    except TypeError as exc:
        assert "only a single audio input or a list of audio inputs" in str(exc)
    else:
        raise AssertionError("fetch_audio should reject non-audio inputs")


def test_pixtral_image_processor_fetch_images_accepts_hf_processor_inputs():
    processor = MistralCommonImageProcessor(object())
    image = Image.new("RGB", (4, 4))

    assert processor.fetch_images(image) is image
    assert processor.fetch_images([image])[0] is image
    assert processor.fetch_images([[image]])[0][0] is image

    try:
        processor.fetch_images(object())
    except TypeError as exc:
        assert "only a single image input or a list of image inputs" in str(exc)
    else:
        raise AssertionError("fetch_images should reject non-image inputs")
