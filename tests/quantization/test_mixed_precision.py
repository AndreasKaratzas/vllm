# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test quark-quantized {MXFP4, FP8} mixed precision models.

Run `pytest tests/quantization/test_mixed_precision.py`.

"""

import importlib
import importlib.metadata
import importlib.util
from dataclasses import dataclass

import lm_eval
import pytest
import torch
from packaging import version

QUARK_MXFP4_AVAILABLE = importlib.util.find_spec("quark") is not None and version.parse(
    importlib.metadata.version("amd-quark")
) >= version.parse("0.8.99")


@dataclass
class EvaluationConfig:
    model_name: str
    accuracy_numbers: dict[str, float]
    tp: int

    def get_model_args(self) -> str:
        return (
            f"pretrained={self.model_name},"
            f"tensor_parallel_size={self.tp},"
            "dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=False"
        )


TEST_CONFIGS = [
    # Mixed-precision (AMP) model
    # - Demonstrates end-to-end pipeline functionality
    EvaluationConfig(
        model_name="amd/Qwen3-8B-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8",
        accuracy_numbers={"arc_challenge": 0.52, "mmlu": 0.72},
        tp=1,
    ),
    # Non-mixed-precision (PTQ) model
    # - Reference for pipeline compatibility verification -> No conflicts or breakings
    EvaluationConfig(
        model_name="amd/Llama-2-70b-chat-hf_FP8_MLPerf_V2",
        accuracy_numbers={"arc_challenge": 0.53, "mmlu": 0.61},
        tp=4,
    ),
]


@pytest.mark.parametrize(
    "eval_config",
    TEST_CONFIGS,
    ids=[config.model_name for config in TEST_CONFIGS],
)
@pytest.mark.skipif(not QUARK_MXFP4_AVAILABLE, reason="amd-quark>=0.9 is not available")
def test_mixed_precision_model_accuracies(eval_config: EvaluationConfig):
    device_count = torch.accelerator.device_count()
    if device_count < eval_config.tp:
        pytest.skip(
            f"{eval_config.model_name} requires TP={eval_config.tp}, "
            f"but only {device_count} accelerator(s) are visible."
        )

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=eval_config.get_model_args(),
        tasks=list(eval_config.accuracy_numbers.keys()),
        batch_size=8,
    )

    rtol = 0.05

    for task, expect_accuracy in eval_config.accuracy_numbers.items():
        measured_accuracy = results["results"][task]["acc,none"]
        assert (
            measured_accuracy - rtol < expect_accuracy
            and measured_accuracy + rtol > expect_accuracy
        ), f"Expected: {expect_accuracy} |  Measured: {measured_accuracy}"
