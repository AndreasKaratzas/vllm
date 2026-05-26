# ROCm CI Stabilization Review Split

Date: 2026-05-25

Review branch: `wip-ci-fix-fin-proto`
Review base: `origin/main` at `d4004455d`
Patch shape in this workspace:

- `git status --short --untracked-files=all`: 139 entries.
- `git diff --name-only`: 109 tracked modified files.
- Untracked new files: 30 entries, including `PR_*.md`, issue notes, and new source files.
- Staged files: 0.

This document intentionally uses `git status --short --untracked-files=all` as
the source of truth because that is what matches the VS Code diff column. Plain
`git diff --name-only` undercounts the review surface because it ignores
untracked new files.

## Framing

The current branch is no longer a single bug fix. It is a collection of ROCm CI
repairs, test topology changes, runtime allocator fixes, distributed fixes,
quantization gates, model-specific fixes, and a few changes that look closer to
feature work than CI stabilization. The safe path is to split this into focused
PRs where each PR can answer:

- Which Buildkite test group or failure class does this fix?
- Which files are necessary for that fix?
- Which files are weakly justified and should move to a follow-up?
- Which exact pytest or Buildkite command validates it?

## Proposed PR Set

1. `PR_01.md` - AMD Buildkite topology, hardware gates, and sharding.
2. `PR_02.md` - Retired. The Docker/non-root image cleanup files were removed
   from the proto diff because they are not part of this ROCm CI contribution.
3. `PR_03.md` - Distributed, elastic EP, and collective test stability.
4. `PR_04.md` - cuMem sleep, wake, and ROCm process isolation.
5. `PR_05.md` - Compile/fullgraph and fusion pass gating.
6. `PR_06.md` - Quantization, ModelOpt, torchao, and model reload gates.
7. `PR_07.md` - Kernel and attention numeric fixes.
8. `PR_08.md` - DeepSeek V4 norm-router GEMM; likely split/drop from ROCm CI.
9. `PR_09.md` - DeepEP/OCP/GPT-OSS MoE harnesses.
10. `PR_10.md` - DeepSeek V4 AMD/NVIDIA split and ROCm MLA sparse attention.
11. `PR_11.md` - Spec decode and EAGLE acceptance behavior.
12. `PR_12.md` - KV connector, NIXL, Mooncake, and offloading scheduler fixes.
13. `PR_13.md` - Simple KV offload and filesystem I/O cleanup.
14. `PR_14.md` - Dataset, LM-eval, and Hugging Face compatibility.
15. `PR_15.md` - Tokenizer, processor, and registry hygiene.
16. `PR_16.md` - Model-specific ROCm fixes.
17. `PR_17.md` - Entrypoints, API protocol, tool parser, pooling, and speech tests.
18. `PR_18.md` - MHC TileLang and mixed-precision GEMM paths.
19. `PR_19.md` - Review artifacts and issue notes.
20. `PR_20.md` - Residual miscellaneous test baselines and follow-up bucket.

The split is intentionally conservative. Several groups can probably merge
together later, but the first pass should keep risky runtime changes away from
test-only gates.

## Test Group Map

### `Basic Correctness`

Primary files:

- `csrc/cumem_allocator.cpp`
- `vllm/device_allocator/cumem.py`
- `tests/basic_correctness/test_cumem.py`

Current hypothesis:

The sleep integration failure reports GPU memory back at `0.00 GB` before the
sleep test starts, then fails in the C++ allocator with ROCm/HIP OOM while the
device still reports almost all VRAM free. That points away from leaked physical
VRAM and toward ROCm virtual-memory reservation behavior in the cuMem allocator.

Proposed fix:

- Keep the original sleep test size; do not shrink `test_sleep.py`.
- In C++, first try ROCm VA reservation with explicit granularity alignment,
  then retry with default alignment if ROCm reports failure.
- After sleep releases ROCm physical handles, free and re-reserve the same VA so
  physical memory can actually return to the free pool while pointer identity is
  preserved for wake-up.
- On ROCm, use PyTorch's lower-level MemPool begin/end routing instead of
  `torch.cuda.memory.use_mem_pool()`. The public context manager releases the
  private pool on exit, and the tested HIP/PyTorch stack can later abort in
  `MemPool::~MemPool()` when that already-released pool is destroyed.
- Track mapped vs sleeping allocations so failed wake-up OOM paths tear down as
  VA-only allocations instead of trying to unmap stale physical handles.
- Synchronize before/after sleep and wake to prevent unmap/remap races.
- In `test_cumem.py`, allocate 60% of currently free memory on ROCm so one
  allocation fits and two live allocations should require about 120% of free
  memory. This expresses the intended wake-up OOM instead of relying on an
  architecture-specific fraction.
- Keep ROCm post-sleep memory assertions active. The C++ allocator fix should
  make the existing `mem_get_info` checks pass on ROCm too; if they fail again,
  treat that as source behavior to debug rather than a test waiver.

Validation observed locally:

- `pytest -v -s tests/basic_correctness/test_cumem.py::test_python_error tests/basic_correctness/test_cumem.py::test_basic_cumem tests/basic_correctness/test_cumem.py::test_cumem_with_cudagraph --tb=short`
- `pytest -v -s 'tests/basic_correctness/test_cumem.py::test_end_to_end[hmellor/tiny-random-LlamaForCausalLM]' --tb=short`
- `pytest -v -s tests/entrypoints/serve/instrumentator/test_sleep.py::test_sleep_mode`
- `pytest -v -s tests/entrypoints/serve/instrumentator/test_orca_metrics.py::test_single_completion[True] tests/entrypoints/serve/instrumentator/test_sleep.py::test_sleep_mode`
- `pytest -v -s tests/entrypoints/serve/instrumentator`

Weak confidence:

The C++ `clear_error_state()` helper is defensive rather than the main observed
failure. It is still reasonable because the allocator uses a sticky global
`error_code`, but it should be called out in PR review.

The ROCm MemPool begin/end path uses private PyTorch APIs. It is still the
cleaner option compared with fast-exiting subprocesses because it fixes the
specific teardown failure and lets pytest observe normal child-process exit.

The ROCm LLM kwargs in `test_cumem.py` are only test-scoping knobs. They keep
allocator-focused tests from reserving a huge MI300 KV cache for tiny prompts;
they are not a substitute for real API-server sleep-mode coverage.

### NCCL/HIP Invalid Argument In Multi-GPU Jobs

Relevant exact AMD labels:

- `V1 e2e (2 GPUs)`
- `RayExecutorV2 (4 GPUs)`
- `Distributed Torchrun + Examples (4 GPUs)`
- `Qwen3-30B-A3B-FP8-block Accuracy (4xH100-4xMI300)`
- `Qwen3-Next-80B-A3B-Instruct MTP Async EPLB Accuracy`

Primary files:

- `tests/distributed/test_comm_ops.py`
- `tests/distributed/test_pp_cudagraph.py`
- `vllm/distributed/elastic_ep/elastic_execute.py`
- `vllm/distributed/elastic_ep/elastic_state.py`
- `vllm/distributed/kv_transfer/kv_connector/utils.py`

Observation:

The nightly run is green for several of these while the WIP branch showed NCCL
`unhandled cuda error` and HIP `invalid argument`. Any fix here needs an A/B
comparison against main and the exact Buildkite final command. A passing local
single test is not enough if the whole group fails due to worker environment or
device visibility.

Weak confidence:

This remains one of the highest-risk areas. Avoid test-side workarounds that
hide a worker setup bug.

### `Quantization` And Model Reload

Primary files:

- `tests/quantization/test_modelopt.py`
- `tests/quantization/test_torchao.py`
- `tests/quantization/test_mixed_precision.py`
- `tests/model_executor/model_loader/test_reload.py`
- `vllm/model_executor/layers/quantization/modelopt.py`
- `vllm/model_executor/layers/quantization/torchao.py`
- `vllm/model_executor/layers/quantization/input_quant_fp8.py`
- `vllm/model_executor/layers/quantization/auto_gptq.py`

Purpose:

Keep unsupported FP8/FP4/NVFP4 paths off hardware that cannot execute them and
make ROCm dtype selection explicit. MI250 should not run FP8/FP4 paths that
require MI300/MI325-class support.

Weak confidence:

Skips must be based on ROCm architecture capability, not just
`current_platform.is_rocm()`. NVFP4 naming should be reviewed carefully.

### Kernel, Attention, And MoE Runtime Fixes

Primary files:

- `csrc/cache_kernels_fused.cu`
- `csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu`
- `tests/kernels/attention/*`
- `tests/kernels/core/*`
- `tests/kernels/quantization/test_rocm_skinny_gemms.py`
- `vllm/v1/attention/backends/rocm_attn.py`
- `vllm/v1/attention/backends/mla/triton_mla.py`
- `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`

Purpose:

Handle ROCm-specific numeric behavior and backend support gaps without widening
test tolerances unnecessarily.

Weak confidence:

Tolerance changes should include max-error or before/after evidence. Backend
selection changes should explain why the selected backend supports the model
feature, for example attention sinks in GPT-OSS/EAGLE tests.

### MoE Norm Router GEMM

Primary files:

- `CMakeLists.txt`
- `csrc/moe/*`
- `vllm/_custom_ops.py`
- `vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py`
- `vllm/model_executor/layers/fused_moe/*`
- `benchmarks/kernels/benchmark_norm_router_gemm.py`

Purpose:

Wire a norm-router GEMM path and related fused-MoE behavior.

Weak confidence:

This is a product/runtime addition, not a narrow CI repair. It should only ship
with a focused failing test group and direct kernel validation.

### DeepSeek V4

Primary files:

- `vllm/models/deepseek_v4/__init__.py`
- `vllm/models/deepseek_v4/amd/model.py`
- `vllm/models/deepseek_v4/amd/mtp.py`
- `vllm/models/deepseek_v4/nvidia/model.py`
- `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`

Purpose:

Make AMD DeepSeek V4 discoverable and separate AMD-specific behavior from the
NVIDIA path.

Weak confidence:

This is large. The AMD files should be reviewed for whether they need to be full
Python sources or can remain thinner wrappers. Avoid combining this with
unrelated CI gates.

### KV Connector And Offload

Primary files:

- `vllm/distributed/kv_transfer/kv_connector/v1/*`
- `tests/v1/kv_connector/*`
- `vllm/v1/simple_kv_offload/manager.py`
- `vllm/v1/kv_offload/tiering/fs/io.py`

Purpose:

Stabilize NIXL/Mooncake/offload scheduler behavior and simple KV offload
cleanup.

Weak confidence:

Connector behavior is user-visible. Changes to partial-chunk handling, cleanup,
or scheduling semantics need direct unit and multi-GPU validation.

### Entrypoints, Pooling, Speech, And Tool Parsers

Primary files:

- `vllm/entrypoints/anthropic/api_router.py`
- `vllm/entrypoints/openai/engine/protocol.py`
- `tests/entrypoints/openai/tool_parsers/test_granite4_tool_parser.py`
- `tests/tool_parsers/test_mistral_tool_parser.py`
- `tests/entrypoints/pooling/*`
- `tests/entrypoints/speech_to_text/correctness/test_transcription_api_correctness.py`

Purpose:

Address API/test failures around tool parser expectations, pooling, and speech
accuracy.

Weak confidence:

Speech WER baselines must be architecture-specific. MI300 and MI355 can differ,
so a generic ROCm relaxation is not acceptable without evidence.

## Issues To File Or Keep Separate

1. Llama4 Eagle MM input rendering and/or Triton compile hotspot.
   The V1 e2e 4-GPU group spent most of its wall time rendering conversations
   and compiling rather than validating distributed generation. This needs a
   profiling issue with top-20 timings, not only more Buildkite shards.

2. MI355 AITER inaccuracy.
   Keep this architecture-specific. Do not weaken MI300 expectations to paper
   over an MI355-only AITER accuracy problem.

3. NCCL/HIP invalid argument in Buildkite/Kubernetes.
   Because nightly main passed while the WIP branch failed, this needs an
   environment diff and exact final-command repro before runtime changes merge.

4. Product-like additions inside the CI branch.
   Norm-router GEMM and any new benchmark/kernel wiring should be reviewed as
   runtime PRs unless directly required by a failing AMD CI group.

## Weak-Confidence Files

- `.buildkite/test-amd.yaml`
- `tests/basic_correctness/test_cumem.py`
- `csrc/cumem_allocator.cpp`
- `tests/distributed/test_comm_ops.py`
- `vllm/distributed/elastic_ep/elastic_execute.py`
- `vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py`
- `csrc/moe/dsv4_norm_router_gemm*`
- `vllm/models/deepseek_v4/amd/model.py`
- `vllm/models/deepseek_v4/amd/mtp.py`
- `vllm/utils/hf_datasets.py`
- `tests/entrypoints/speech_to_text/correctness/test_transcription_api_correctness.py`

These are not necessarily wrong. They are the files where the review should ask
"is this needed here?" most aggressively.
