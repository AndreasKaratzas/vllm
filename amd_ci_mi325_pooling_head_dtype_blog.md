# AMD CI Regression Notes: MI325 Extended Pooling

## Test Group Name

AMD: Language Models Test (Extended Pooling) (mi325_1)

Buildkite job:
https://buildkite.com/vllm/ci/builds/69186/canvas?sid=019e823a-1049-40ca-b405-f8d49963493f&tab=output

## Failure

`tests/models/language/pooling/test_head_dtype.py::test_classify_models[half-nie3e/sentiment-polish-gpt2-small]`

## Problem

The failed CI run did not expose a mismatch in `head_dtype`, vLLM logits,
or the HF/vLLM classification comparison. It failed before that comparison
could run, while the HF reference path was initializing:

`AutoTokenizer.from_pretrained` called `hf_hub_download`, which timed out in
`http_get` with `httpx.ReadTimeout` while waiting for a Hugging Face Hub
response. The ROCm image already sets `HF_HUB_DOWNLOAD_TIMEOUT=60`, so the
existing 60 second read cap was not enough for this cold or partially-cold
cache path on the MI325 CI worker.

The direct evidence is one failing Buildkite sample for this test group. I am
therefore treating this as a targeted CI network-budget fix for the observed
external model setup failure, not as proof that AMD inference behavior changed
or that the pooling test should be marked flaky.

## Proposed Solution

Make the AMD CI runner set explicit, longer Hugging Face Hub timeout defaults
before launching the test container, and pass those values through `docker run`:

- `HF_HUB_DOWNLOAD_TIMEOUT=300`
- `HF_HUB_ETAG_TIMEOUT=60`

This keeps the test itself unchanged. It still uses
`nie3e/sentiment-polish-gpt2-small`, still constructs the HF reference model,
still checks `head_dtype` for both `float32` and `model`, and still compares
HF and vLLM classification outputs with the existing tolerance.

The timeout belongs in `.buildkite/scripts/hardware_ci/run-amd-test.sh`
instead of this individual test because the failure is in AMD CI's model
artifact acquisition layer, not in pooling model semantics. The runner already
owns the HF cache mount and forwards `HF_TOKEN`; adding the Hub timeout there
keeps download behavior consistent for AMD hardware jobs without changing
test logic or model coverage.

The timeout is set host-side before `docker run` and then explicitly passed
into the container. That matters because the existing value comes from the ROCm
Docker image; setting the runner default to `300` is what overrides the image's
`60` for AMD CI jobs. I am not changing `docker/Dockerfile.rocm` because this
is a CI runner policy knob, and changing it in the runner avoids rebuilding and
republishing the ROCm image for a hardware-job reliability setting.

## Motivation Behind The New Functionality

This is not a model deprecation, a threshold change, or a platform skip. The
new behavior is a validation-enabling CI guardrail: AMD model tests should not
fail before reaching the vLLM behavior under test just because a Hub read stalls
longer than the ROCm image's default 60 second cap.

The selected values are intentionally conservative:

- `300s` gives large model-test downloads room to survive transient Hub or CI
  network slowness without making the request unbounded. AMD hardware jobs are
  expensive enough that waiting a few extra minutes for a slow but progressing
  model artifact is preferable to failing the full shard before validation.
- `60s` for metadata/etag checks is longer than Hugging Face Hub's default
  while still bounded. `from_pretrained` resolution performs metadata/etag work
  as well as file downloads, so it should have a consistent CI budget.
- Both values remain overridable from the Buildkite environment.

I considered a retry-only fix, but the observed failure is a single Hub read
exceeding the existing `60s` cap. Extending the bounded read timeout addresses
that failure mode directly; retries would still be vulnerable to repeated
premature read timeouts on slow responses.

## Local Reproduction And Verification

The exact failed node was run locally under `/app/vllm-mi325-pooling` on the
MI300 machine after rebuilding the ROCm extension stack with `heka vllm rebuild`
because the merged PR stack includes native code changes.

Results:

- Before rebuild, the HF model and tokenizer download completed locally, but
  the run failed later because the worktree's native vLLM extensions were not
  installed.
- After `heka vllm rebuild`, the exact test passed locally.
- After the timeout runner change, the exact test passed again with
  `HF_HUB_DOWNLOAD_TIMEOUT=300` and `HF_HUB_ETAG_TIMEOUT=60` in the environment:
  `1 passed, 17 warnings in 55.80s`.
- `bash -n .buildkite/scripts/hardware_ci/run-amd-test.sh` passes.
- A local Hugging Face Hub constants check confirms those environment values
  are read as `300` and `60` when set before import.
- No test was skipped, no model was swapped, no seed was changed, and no
  numerical threshold was relaxed.

## Personal Notes

I initially suspected the failing test name might point at a pooling
`head_dtype` regression. The log contradicted that: the only failing stack is
inside the HF download path, and local execution reached and passed the actual
classification assertions.

I also checked the Buildkite environment dump in the failing job and found
`HF_HUB_DOWNLOAD_TIMEOUT=60` already present. That changed the fix from "pass
the existing timeout through" to "override the ROCm image default for AMD CI
jobs with a larger bounded timeout."

## Q & A For Review

Q: Why not skip this model on ROCm?

A: The model and assertion path are valid on ROCm. The exact test passes
locally on MI300 after rebuild. Skipping would remove coverage for the pooling
classification head dtype path when the real failure is a transient Hub read
timeout.

Q: Why not relax the `torch.allclose` tolerance?

A: The failing stack never reaches the output comparison. Changing numerical
tolerance would not address the observed failure and would weaken the test.

Q: Why change the AMD runner instead of `test_head_dtype.py`?

A: The test logic is not the source of the failure. The AMD runner owns the HF
cache mount and container environment, and the same download behavior can affect
other AMD model tests. Keeping the fix in the runner avoids per-test download
plumbing and preserves model coverage.

Q: Why is `60s` not enough?

A: The failed Buildkite container already had `HF_HUB_DOWNLOAD_TIMEOUT=60` and
still timed out in `hf_hub_download`/`http_get`. That proves the current ROCm
image default is insufficient for at least this MI325 extended pooling run.

Q: Are these environment variables supported by the installed Hub version?

A: Yes. In the local environment, importing `huggingface_hub.constants` after
setting `HF_HUB_DOWNLOAD_TIMEOUT=300` and `HF_HUB_ETAG_TIMEOUT=60` reports those
exact values. The failing stack is in the same Hugging Face Hub download path
that reads these constants.

Q: Is `300s` too broad?

A: It is a bounded per-request/download timeout, not an unbounded retry loop.
The job already has an overall Buildkite timeout. This only gives the model
download path enough room to handle slow Hub reads before pytest reaches the
actual vLLM validation.

Q: What remains risky?

A: If the Hub is fully unavailable, the job can still fail. This change is not
intended to hide outages; it only prevents slow-but-eventually-successful reads
from failing prematurely. A future improvement could prewarm known heavy model
artifacts earlier in the pipeline, but that is larger in scope than this
single regression.

Q: What should reviewers look for on the next CI run?

A: The AMD job environment should show `HF_HUB_DOWNLOAD_TIMEOUT=300` and
`HF_HUB_ETAG_TIMEOUT=60` inside the container, and the extended pooling shard
should progress past `test_head_dtype.py::test_classify_models[...]` into the
actual model assertions rather than failing in `hf_hub_download`.
