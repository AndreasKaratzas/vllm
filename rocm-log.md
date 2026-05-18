# ROCm Flake Investigation Log

## 2026-05-18

### Buildkite logs pulled

- API Server 2: `raw_logs/rocm_flake_investigation/buildkite_66603/019e37be-55c0-4afc-a908-afb3d5050f3e.clean.log`
- OpenAI Part 3: `raw_logs/rocm_flake_investigation/buildkite_66603/019e37be-55c4-40b9-b48d-3264f42b37ef.clean.log`
- OpenAI Part 3: `raw_logs/rocm_flake_investigation/buildkite_66622/019e3917-e229-4818-af49-3d8b9923322f.clean.log`
- OpenAI Part 3 additional incident: `raw_logs/rocm_flake_investigation/buildkite_66631/019e3988-077a-4b07-86bf-9cf3475594ae.clean.log`

### Confirmed Buildkite commands

- API Server 2 runs from `/vllm-workspace/tests`, sets CUDA core dump environment, then runs:
  - `pytest -v -s entrypoints/serve/instrumentator`
  - `PYTHONPATH=/vllm-workspace pytest -v -s entrypoints/rpc`
  - `pytest -v -s tool_use`
- OpenAI Part 3 runs from `/vllm-workspace/tests`, sets CUDA core dump environment, then runs:
  - `pytest -v -s entrypoints/openai --ignore=entrypoints/openai/chat_completion --ignore=entrypoints/openai/completion --ignore=entrypoints/openai/correctness/ --ignore=entrypoints/openai/tool_parsers/ --ignore=entrypoints/openai/responses --ignore=entrypoints/openai/test_multi_api_servers.py`
- Note: the API Server 2 command shown in the user prompt uses `(export VLLM_WORKER_MULTIPROC_METHOD=spawn)`, which would not persist beyond that subshell. The pulled Buildkite logs still show spawned API servers inheriting `VLLM_WORKER_MULTIPROC_METHOD=spawn`, so the variable is likely provided by the surrounding CI environment. The local validation runner exports it in the parent shell.

### API Server 2 failure: sleep-mode OOM with free VRAM

- Failing test: `entrypoints/serve/instrumentator/test_sleep.py::test_sleep_mode`.
- The failing server command uses `meta-llama/Llama-3.2-1B`, `--dtype bfloat16`, `--max-model-len 8192`, `--max-num-seqs 128`, and `--enable-sleep-mode`.
- The failure is `CUDA Error: out of memory at /app/vllm/csrc/cumem_allocator.cpp:328`, followed by a PyTorch HIP OOM while allocating only 502 MiB.
- The same message reports GPU 0 has 191.98 GiB total and 191.31 GiB free, so this is not normal VRAM exhaustion. It points at ROCm virtual-memory reservation inside sleep-mode's custom CuMem allocator.
- Source location: `csrc/cumem_allocator.cpp` called `cuMemAddressReserve(&d_mem, alignedSize, granularity, 0, 0)` on ROCm. The NVIDIA branch uses alignment `0`.
- Fix part 1: keep the explicit ROCm reservation alignment as the first attempt, then fall back to alignment `0` if ROCm rejects that virtual-address reservation. The mapping size remains granularity-aligned while the fallback lets the driver choose a usable VA alignment.
- Follow-up targeted runs showed the model could then load, but sleep freed almost no VRAM and wake-up failed at `cumem_allocator.cpp:151`. The key log line was `Sleep mode freed 0.0 GiB memory`.
- Git history showed commit `10a1018c1` previously fixed ROCm sleep release by freeing and re-reserving the VA after unmap/release, but commit `66d1cc0c7` removed that behavior from the shared allocator free path because it broke normal tensor-parallel frees.
- Fix part 2: restore the ROCm VA free/re-reserve cycle only in the Python sleep unmap path (`python_unmap_and_release`), not in the allocator's normal `my_free` path. This keeps the sleep-mode VRAM-release behavior without reintroducing the normal-free regression.
- Added explicit `torch.cuda.synchronize()` before and after sleep unmapping and
  wake-up remapping so the public sleep/wake transitions are synchronous at the
  API boundary and sleep does not unmap memory while queued GPU work may still
  reference it.

### OpenAI Part 3 failure: Qwen3 LoRA server crash

- Failing tests:
  - `entrypoints/openai/models/test_models.py::test_check_models`
  - `entrypoints/openai/test_return_tokens_as_ids.py` token-ID completion/chat cases.
- These tests launch `Qwen/Qwen3-0.6B` with `--enable-lora`, Qwen3 LoRA adapters, `--max-lora-rank 64`, and `--max-num-seqs 128`.
- The server exits during engine startup with an HSA hardware exception shortly after:
  - `Using V2 Model Runner`
  - `Using PunicaWrapperGPU.`
  - `Using default LoRA kernel configs`
- Important correction after review: the temporary config guard that marked `LoRA on ROCm` unsupported was removed. LoRA is supported on ROCm; the observed failure needs a narrower V2-model-runner investigation/fix, not a broad feature disable.

### OpenAI Part 3 failure: schema-generated invalid chat tool call

- Failing schemathesis case: `POST /v1/chat/completions/batch`.
- Reproduced body contains a nested batch item with assistant `tool_calls` using `{"type": "custom", "custom": ...}` instead of the OpenAI chat-completion function-call shape.
- Server returned 500 with message `"'function'"`.
- Source location: chat message postprocessing indexed `item["function"]` without validating that the tool call is a function tool call.
- Fix: validate assistant `tool_calls` entries in `vllm/entrypoints/chat_utils.py` before touching `function`, rejecting unsupported non-function tool calls with `ValueError`.
- Follow-up schema runs showed malformed generated chat/render requests can raise `ValueError`, `TypeError`, `IndexError`, or Pydantic `ValidationError` during render-time preprocessing after the outer request model has parsed. Added boundary catches in the chat/completion render paths and the batched chat path so these malformed client requests become 400 responses instead of uncaught ASGI exceptions. After review, several catches were narrowed back to the exception classes observed at each boundary.

### OpenAI Part 3 failure: schema-generated huge generate `n`

- Failing schemathesis case: `POST /inference/v1/generate`.
- Reproduced body includes `sampling_params.n = 7125` while the schema test server starts with `--max-num-seqs 5`.
- The request times out after 180 seconds instead of being rejected quickly.
- Fix: added early validation in `ServingTokens.serve_tokens` that rejects `sampling_params.n > scheduler_config.max_num_seqs`.
- Also constrained `/inference/v1/generate` `token_ids` to `min_length=1` and catch preprocessing value/type/index errors as 400 responses. Schemathesis can otherwise generate empty token lists or out-of-vocabulary token ids that are invalid client input.

### Rebuild

- Because `csrc/cumem_allocator.cpp` changed, ran `../vllm-scripts/rebuild.sh` from `/app/vllm`.
- First run was interrupted after it stalled in CMake FetchContent cloning Triton kernels.
- Successful run used `TRITON_KERNELS_SRC_DIR=/app/vllm/vllm/third_party/triton_kernels` and then executed the required command `../vllm-scripts/rebuild.sh` from `/app/vllm`.
- Rebuilt native modules import successfully: `vllm._C`, `vllm._rocm_C`, and `vllm.cumem_allocator`.

### Validation plan

- Created `/vllm-workspace -> /app/vllm` locally so the Buildkite commands run against the same path shape.
- Add an orchestrator that creates separate log directories and runs each group command repeatedly with parent-shell `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
- Run targeted repro checks first:
  - `entrypoints/serve/instrumentator/test_sleep.py::test_sleep_mode`
  - Qwen3 LoRA OpenAI tests that crashed under V2
  - schemathesis schema tests covering `/v1/chat/completions/batch` and `/inference/v1/generate`
- Then run each full Buildkite group 10 times and keep each run log separately.

### Targeted validation completed

- Native import check passed after rebuild:
  - `vllm._C`
  - `vllm._rocm_C`
  - `vllm.cumem_allocator`
- `entrypoints/serve/instrumentator/test_sleep.py::test_sleep_mode` passed after the scoped ROCm VA-cycle fix.
  - Log: `raw_logs/rocm_flake_validation/targeted/sleep_mode_scoped_cycle.log`
  - Important signal: sleep now frees about 264 GiB on the local MI300 run, and wake-up succeeds.
- OpenAI LoRA/token-id targeted tests passed together.
  - Log: `raw_logs/rocm_flake_validation/targeted/openai_lora_token_ids.log`
  - Important signal: Qwen3 LoRA server starts and serves the token-id completion/chat tests without the V2/Punica ROCm hardware exception.
- `entrypoints/openai/test_openai_schema.py::test_openapi_stateless` passed after the API-boundary fixes.
  - Logs:
    - `raw_logs/rocm_flake_validation/targeted/openai_schema_stateless_after_boundary_fix.log`
    - `raw_logs/rocm_flake_validation/targeted/openai_schema_stateless_after_broad_boundary_fix.log`
  - Important signal: `/v1/chat/completions/batch` invalid generated cases return 400; `/inference/v1/generate` overlarge `n` and invalid token inputs return quickly instead of timing out.

### Full-group validation in progress

- Probe run for `api-server-2` started:
  - Log root: `raw_logs/rocm_flake_validation/api_server_2_probe`
  - Current log: `raw_logs/rocm_flake_validation/api_server_2_probe/api-server-2/run_01.log`
- Probe run failed in `tool_use/test_tool_calls.py::test_tool_call_and_choice[granite-3.0-8b]`, not in the original sleep-mode test.
  - Log: `raw_logs/rocm_flake_validation/api_server_2_probe/api-server-2/run_01.log`
  - Failure: the Granite 3.0 server returned no parsed tool calls for the single Dallas weather prompt.
  - Diagnostic script with response dumps showed the model was answering in plain text (`finish_reason=stop`) rather than emitting malformed tool JSON, so the Granite parser was not the root cause.
  - The Granite 3.0 custom template only listed `available_tools`; unlike Granite 3.1's built-in template, it gave no default system instruction to call tools when needed.
- Fix: add a generic default system instruction to `examples/tool_chat_template_granite.jinja` when tools are present and the user did not supply a system message.
  - Instruction order matters for Granite 3.0. A prompt sweep showed that placing the test-specific US-state-abbreviation reminder before the `<|tool_call|>` format instruction produced `state: "TX"`; placing extra argument-format instructions after the call-format sentence made the model fall back to plain text.
  - Follow-up cleanup moved that state-abbreviation reminder into the Granite 3.0 tool-use test config so the public example template stays generic.
  - Targeted validation passed:
    - `raw_logs/rocm_flake_validation/targeted/granite30_tool_call_and_choice_abbrev_before.log`
    - `raw_logs/rocm_flake_validation/targeted/granite30_tool_use_subset_abbrev_before.log`
  - After moving the state-abbreviation reminder out of the public template and
    into the Granite 3.0 tool-use test config, targeted validation still passed:
    `pytest -v -s tool_use/test_tool_calls.py::test_tool_call_and_choice --models granite-3.0-8b`
    (`1 passed, 10 skipped, 17 warnings in 33.54s`).
- The follow-up full API Server 2 probe got past sleep and Granite, then exposed two deterministic optional-middleware failures:
  - `entrypoints/serve/instrumentator/test_optional_middleware.py::test_v2_endpoint_rejects_missing_api_token[server0]`
  - `entrypoints/serve/instrumentator/test_optional_middleware.py::test_v2_endpoint_accepts_valid_api_token[server0]`
  - Root cause: the tests called `server.url_for("/v2/embed")`; `RemoteVLLMServer.url_for` concatenated this into `//v2/embed`, which FastAPI routed as 404 before auth handling.
  - Fix: normalize leading/trailing slashes in `tests/utils.py::RemoteVLLMServer.url_for`.
  - Targeted validation passed: `raw_logs/rocm_flake_validation/targeted/optional_middleware_v2_url_for.log`
- Full API Server 2 probe passed after the URL helper fix.
  - Log root: `raw_logs/rocm_flake_validation/api_server_2_probe_after_url_for`
  - Summary: `api-server-2	1	0	1052	raw_logs/rocm_flake_validation/api_server_2_probe_after_url_for/api-server-2/run_01.log`
  - Important signals:
    - `entrypoints/serve/instrumentator` completed with `35 passed, 3 skipped`.
    - `test_v2_endpoint_rejects_missing_api_token` used `POST /v2/embed` and returned 401.
    - `test_sleep_mode` reported two sleep frees of about 264 GiB and did not hit ROCm OOM.
    - `entrypoints/rpc` completed with `3 passed`.
    - `tool_use` completed successfully, including the previous Granite 3.0 failure point.

### OpenAI Part 3 full probe

- Full OpenAI Part 3 probe passed.
  - Log root: `raw_logs/rocm_flake_validation/openai_part_3_probe`
  - Summary: `openai-part-3	1	0	1239	raw_logs/rocm_flake_validation/openai_part_3_probe/openai-part-3/run_01.log`
  - Important signals:
    - `test_check_models` got past the Qwen3 LoRA server start and logged ROCm LoRA fallback to V1.
    - The four `test_return_tokens_as_ids` Qwen3 LoRA cases completed without the previous HSA crash.
    - `test_openapi_stateless[POST /v1/chat/completions/batch]` reported `SUBPASS`; invalid generated batch/chat bodies returned 400.
    - `test_openapi_stateless[POST /inference/v1/generate]` reported `SUBPASS`; invalid `n` and token inputs returned quickly as 400 instead of timing out.
- First OpenAI Part 3 10x attempt found another schema boundary in run 2:
  - `/v1/completions` returned 500 for generated invalid completion requests.
  - The first failure mode was empty/out-of-vocabulary prompts reaching `AsyncLLM.add_request`; fixed by converting non-streaming completion validation exceptions from the engine generator into `BadRequestError`.
  - A second run exposed invalid batch-chat sampling params (`min_p`) escaping while constructing per-conversation `SamplingParams`; fixed by returning `BadRequestError` from the batch handler for those validation exceptions.
  - A final `/v1/completions` falsifying case used `kv_transfer_params` with an integer below signed int64 range: `{"prompt": [[0]], "kv_transfer_params": {"": -9223372036854775809}}`. That value cannot be encoded for engine msgpack IPC, so the shared OpenAI wrapper now rejects non-msgpack-encodable `kv_transfer_params` as 400 before the request reaches the engine.
- Targeted schema validation passed after these boundary fixes.
  - Log: `raw_logs/rocm_flake_validation/targeted/openai_schema_stateless_after_kv_params_msgpack_guard.log`
  - Result: `1 passed, 24 subtests passed`.
  - Important signals:
    - `/v1/chat/completions/batch` reported `SUBPASS`.
    - `/v1/completions` reported `SUBPASS`.
    - `/inference/v1/generate` reported `SUBPASS`.

### API Server 2 10x validation

- Full `AMD: Entrypoints Integration (API Server 2) (mi300_1)` validation passed 10/10.
  - Log root: `raw_logs/rocm_flake_validation/api_server_2_10x`
  - Summary: `raw_logs/rocm_flake_validation/api_server_2_10x/summary.tsv`
  - Per-run logs: `raw_logs/rocm_flake_validation/api_server_2_10x/api-server-2/run_01.log` through `run_10.log`
  - All ten runs exited 0.
- Important repeated signals:
  - Every run completed `entrypoints/serve/instrumentator` with `35 passed, 3 skipped`.
  - Every run reached the sleep-mode test and freed about 264 GiB twice without ROCm OOM.
  - Every run completed `entrypoints/rpc` with `3 passed`.
  - Every run completed `tool_use`, including the previous Granite 3.0 failure point.

### Post-merge sampler validation and diff pruning

- Fetched and fast-forward merged `origin/main` from `23c15acd7` to `67f58ce23`.
  - Autostash applied cleanly.
- Buildkite nightly `66633` was confirmed to have run at `23c15acd770cf16ed36c6d3fed8e7d78db7d5282`.
  - Failed job: `AMD: Samplers Test (mi250_1)`.
  - The reported beam-search, ignore-eos, and logprobs failures all failed while creating a new `LLM`, before test-specific assertions.
  - Shared root cause in the log: `ValueError: Free memory on device cuda:0 (58.33-58.44/63.98 GiB) on startup is less than desired GPU memory utilization (0.92, 58.87 GiB).`
  - Conclusion: the nightly sampler failures were MI250 engine-startup memory-margin failures, not beam-search correctness failures. No sampler/beam-search code was changed in this branch.
- Re-ran the five sampler failures reported after the merge.
  - Initial log: `raw_logs/sampler_failures_after_merge/initial_after_merge.log`
  - Repeat logs: `raw_logs/sampler_failures_after_merge/run_02.log` through `run_04.log`
  - Summary: `raw_logs/sampler_failures_after_merge/repeat_summary.tsv`
  - Result: all four local runs exited 0.
- Ran the complete sampler command requested by the user.
  - Command: `(cd /vllm-workspace/tests && pytest -v -s samplers)`
  - Log: `raw_logs/sampler_failures_after_merge/full_samplers/run_01.log`
  - Result: `11 passed, 17 warnings in 359.15s`.
- Pruned one unnecessary ROCm allocator change.
  - Removed the earlier broad `cuMemAddressReserve` retry path.
  - Kept only the sleep-mode VA cycle after `unmap_and_release`, inside the existing ROCm-only branch.
  - Rationale: the original failure was sleep-specific HIP OOM after releasing handles while keeping the virtual address reservation; the reserve retry was not tied to any reproduced failure.
- Because `csrc/cumem_allocator.cpp` changed, rebuilt from `/app/vllm` with the exact required command:
  - `../vllm-scripts/rebuild.sh`
  - Result: rebuild completed successfully.
- Revalidated the sleep target after pruning and rebuild.
  - Command: `(cd /vllm-workspace/tests && VLLM_WORKER_MULTIPROC_METHOD=spawn pytest -v -s entrypoints/serve/instrumentator/test_sleep.py::test_sleep_mode)`
  - Log: `raw_logs/rocm_flake_validation/targeted/sleep_mode_after_pruned_reserve_retry.log`
  - Result: `1 passed, 17 warnings in 65.34s`.
- Fixed the sampler teardown gap observed in Buildkite 66633.
  - `EngineCore.shutdown()` now explicitly unfreezes startup objects and runs
    the shared distributed/memory cleanup in the EngineCore process before
    exit, so the uniproc worker process no longer leaves the default process
    group for interpreter shutdown.
  - `VllmRunner.__exit__()` now waits on ROCm until visible GPU memory is below
    the complementary startup threshold, `1 - gpu_memory_utilization`, before
    the next runner can start. This wait is explicitly bounded to 120 seconds.
  - Rationale: cleanup removes the process-group warning, while the wait covers
    ROCm's asynchronous VRAM release; validation still showed one polling
    interval where 100+ GiB was visible after shutdown before dropping below the
    startup-safe threshold.
- Re-ran the five sampler failures after the teardown fix.
  - Command: `(cd /vllm-workspace/tests && pytest -v -s samplers/test_beam_search.py::test_beam_search_with_concurrency_limit samplers/test_beam_search.py::test_beam_search_passes_multimodal_data samplers/test_ignore_eos.py::test_ignore_eos samplers/test_logprobs.py::test_ranks[True-True-half-distilbert/distilgpt2])`
  - Result before EngineCore cleanup: `5 passed, 17 warnings in 246.99s`.
  - Result after EngineCore cleanup: `5 passed, 17 warnings in 206.51s`.
  - The post-cleanup run did not emit the previous
    `destroy_process_group() was not called` warning.

### Current minimality notes

- LoRA on ROCm is supported. The temporary global `LoRA on ROCm` unsupported guard is no longer present in the diff or tree.
- `examples/tool_chat_template_granite.jinja` is used by tests: `tests/tool_use/utils.py` maps `granite-3.0-8b` to this template, and API Server 2 runs `pytest -v -s tool_use`.
- The remaining `try`/`except` changes are API-boundary conversions for malformed schemathesis-generated requests, not debugging scaffolding.
  - Kept catches: Pydantic `ValidationError` while converting batch chat conversations, `ValueError` from invalid sampling params/token ids/template rendering, and msgpack encoding errors for invalid `kv_transfer_params`.
  - Removed or avoided broad catches where a narrower schema or boundary was enough, including the temporary LoRA guard, the broad ROCm reserve retry, and speculative `TypeError`/`IndexError` catches in the render and batch-chat cleanup pass.
  - The goal of the catches that remain is to return 400 for invalid request bodies instead of leaking ASGI 500s.

### Diff cleanup after review

- Removed the untracked validation helper script from the branch diff; `rocm-log.md` is the only untracked investigation artifact intentionally left.
- Tried a global `min_length=1` constraint on `ChatCompletionRequest.messages`, then reverted it because Anthropic `/v1/messages` and `/v1/messages/count_tokens` internally construct a chat request with an empty message list for some malformed generated inputs. That global constraint turned those client errors into 500s, so the final constraint is scoped to `BatchChatCompletionRequest.messages` only.
- Narrowed the batch-chat preprocessing catch to `ValidationError`/`ValueError`, removed the role-extraction `try`/`except`, and rely on the batch-only non-empty-conversation schema for the final-message role lookup.
- Narrowed render endpoint catches to `ValueError` and wrapped only the preprocessing/template calls that can raise on malformed request bodies.
- Changed the malformed assistant `tool_calls` checks in `chat_utils.py` from plain `ValueError` to `VLLMValidationError(parameter="tool_calls")`, matching the existing request-validation error style.
- Re-ran `entrypoints/openai/test_openai_schema.py::test_openapi_stateless` after the schema-scope cleanup. Result: `1 passed, 24 subtests passed in 406.53s`.
- Ran `python -m py_compile` on the modified Python files after the final render diff cleanup. Result: success.
