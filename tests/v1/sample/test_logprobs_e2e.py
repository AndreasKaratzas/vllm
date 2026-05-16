# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import lm_eval
from lm_eval.tasks import TaskManager

from ...utils import RemoteOpenAIServer

# arc-easy uses prompt_logprobs=1, logprobs=1
TASK = "arc_easy"
TASKS = [TASK]
FILTER = "acc_norm,none"
RTOL = 0.03
EXPECTED_VALUE = 0.62

# FIXME(rob): enable prefix caching once supported.
MODEL = "meta-llama/Llama-3.2-1B-Instruct"
MAX_MODEL_LEN = 2048
GPU_MEMORY_UTILIZATION = 0.6
MODEL_ARGS = (
    f"pretrained={MODEL},enforce_eager=True,enable_prefix_caching=False,"
    f"gpu_memory_utilization={GPU_MEMORY_UTILIZATION},"
    f"max_model_len={MAX_MODEL_LEN}"
)
SERVER_ARGS = [
    "--enforce_eager",
    "--no_enable_prefix_caching",
    f"--gpu-memory-utilization={GPU_MEMORY_UTILIZATION}",
    f"--max-model-len={MAX_MODEL_LEN}",
]
NUM_CONCURRENT = 100


def _register_legacy_datasets_list_feature_type() -> None:
    # HF datasets 3.6 removed the legacy "List" feature type, but ROCm CI
    # workers can still have older dataset_info.json files in the shared cache.
    from datasets.features import features as datasets_features

    datasets_features._FEATURE_TYPES.setdefault("List", datasets_features.LargeList)


def _simple_evaluate(**kwargs):
    # lm-eval uses model_args as task metadata when it constructs the
    # TaskManager internally. Keeping task loading explicit avoids compatibility
    # issues when model_args contain connection-only fields such as base_url.
    return lm_eval.simple_evaluate(task_manager=TaskManager(), **kwargs)


def test_prompt_logprobs_e2e():
    _register_legacy_datasets_list_feature_type()
    results = _simple_evaluate(
        model="vllm", model_args=MODEL_ARGS, tasks=TASKS, batch_size="auto"
    )

    measured_value = results["results"][TASK][FILTER]
    assert (
        measured_value - RTOL < EXPECTED_VALUE
        and measured_value + RTOL > EXPECTED_VALUE
    ), f"Expected: {EXPECTED_VALUE} |  Measured: {measured_value}"


def test_prompt_logprobs_e2e_server():
    _register_legacy_datasets_list_feature_type()
    with RemoteOpenAIServer(MODEL, SERVER_ARGS) as remote_server:
        url = f"{remote_server.url_for('v1')}/completions"

        model_args = (
            f"model={MODEL},"
            f"base_url={url},"
            f"num_concurrent={NUM_CONCURRENT},tokenized_requests=False"
        )

        results = _simple_evaluate(
            model="local-completions",
            model_args=model_args,
            tasks=TASKS,
        )

        measured_value = results["results"][TASK][FILTER]
        assert (
            measured_value - RTOL < EXPECTED_VALUE
            and measured_value + RTOL > EXPECTED_VALUE
        ), f"Expected: {EXPECTED_VALUE} |  Measured: {measured_value}"
