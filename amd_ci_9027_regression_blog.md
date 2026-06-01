# AMD CI 9027 Regression Blog

This is my working log for the Buildkite AMD CI 9027 failures, focused first on
MI300 and MI325 jobs. Logs were pulled locally under `/tmp/buildkite-9027/logs`.

## mi300_1: Basic Correctness

- test group name: `mi300_1: Basic Correctness`
- failure `tests/basic_correctness/test_cumem.py`
- problem: ROCm sleep-mode tests exposed two allocator lifetime bugs. First,
  the public `torch.cuda.memory.use_mem_pool` context released the PyTorch
  `MemPool` object at context exit while vLLM intentionally kept HIP virtual
  allocations alive across sleep/wake, which could segfault during teardown.
  Second, after `sleep()` unmapped and released memory handles, ROCm did not
  report the physical memory as free while the same virtual address range
  stayed reserved.
- proposed solution: use PyTorch's internal begin/end allocate-to-pool hooks on
  ROCm so exiting vLLM's allocator context does not destroy the pool object
  prematurely. Track whether each allocation is currently mapped, handle
  allocator frees of already-unmapped sleep allocations without returning a
  stale handle, and in the C++ HIP allocator cycle the virtual address
  reservation after Python releases the handle: free the VA range, immediately
  reserve the same range again, and keep the tensor pointer stable for wake-up.
- motivation behind a new functionality: sleep mode needs a split between
  virtual-address ownership and physical-memory ownership on ROCm. Keeping the
  virtual pointer stable lets captured graphs and tensors remain valid, while
  cycling the reservation gives the driver a chance to return VRAM to the free
  pool during sleep.
- personal notes: this was verified locally on the MI300 host with the full
  file: `VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONFAULTHANDLER=1 python3 -m
  pytest -q -s tests/basic_correctness/test_cumem.py` passed `8 passed, 17
  warnings in 329.55s`. The logs show sleep returning about 174-176 GiB and no
  teardown crash.
- q & a:
  - Q: Why touch C++ instead of only synchronizing in Python?
    A: Synchronization fixed ordering but not the ROCm accounting issue. The
    driver did not expose the physical memory as free while the original VA
    reservation remained active, so the allocator had to cycle the VA mapping
    after releasing the physical handle.
  - Q: Does this invalidate tensor pointers?
    A: No. The C++ path reserves the same virtual address immediately after
    freeing it, and wake-up remaps fresh physical memory at that address before
    the tensors are used again.
  - Q: Why is the `None` free callback path safe?
    A: It is only used for allocations that Python already unmapped and marked
    as released during sleep. The C++ free path then releases the remaining VA
    reservation without trying to release a physical handle twice.

## mi300_4: Distributed NixlConnector PD accuracy

- test group name: `mi300_4: Distributed NixlConnector PD accuracy`
- failure `tests/v1/kv_connector/nixl_integration/config_sweep_accuracy_test.sh`
- problem: the CI command forced `ROCM_ATTN=1`, which translated into
  `--attention-backend ROCM_ATTN`. Runtime validation correctly rejected that
  backend with `KV connector not supported`.
- proposed solution: stop forcing `ROCM_ATTN` for KV connector coverage in
  `.buildkite/test-amd.yaml`. Let ROCm backend auto-selection choose from the
  KV-connector-compatible candidates. Also teach
  `config_sweep_accuracy_test.sh` to accept a generic `ATTENTION_BACKEND`
  override for future explicit backend coverage without adding another
  one-off environment variable.
- motivation behind a new functionality: the script-level `ATTENTION_BACKEND`
  hook gives CI a single explicit override path while preserving automatic
  selection by default. This avoids baking backend-specific assumptions into
  tests whose purpose is NIXL/PD correctness.
- personal notes: this is not a threshold tweak or a skipped test. The failing
  backend is intentionally invalid for the feature under test.
- q & a:
  - Q: Why not hard-code `TRITON_ATTN`?
    A: The ROCm platform selector already knows which backends are compatible
    with KV connector metadata. Defaulting to auto keeps the test portable
    across MI250, MI300, and MI355 while still excluding `ROCM_ATTN`.
  - Q: Why edit `.buildkite/test-amd.yaml` instead of the Jinja template?
    A: In the vLLM repo this YAML is the test definition consumed by the
    external template. The template controls execution mechanics; the invalid
    backend was in this repo's test command list.
  - Q: What if both `ATTENTION_BACKEND` and `ROCM_ATTN` are set?
    A: `ATTENTION_BACKEND` wins. It is the generic override, and `ROCM_ATTN`
    remains a legacy compatibility knob.

## mi250/mi355: NixlConnector PD + Spec Decode acceptance

- test group name: `NixlConnector PD + Spec Decode acceptance (2 GPUs)`
- failure `tests/v1/kv_connector/nixl_integration/spec_decode_acceptance_test.sh`
- problem: the YAML forced `ATTENTION_BACKEND=ROCM_ATTN` for a KV connector
  plus speculative decode scenario. The script already defaults ROCm to
  `TRITON_ATTN`, which is the safer connector-compatible path.
- proposed solution: remove the forced `ROCM_ATTN` from the AMD CI YAML and
  let the script's platform-aware default select `TRITON_ATTN` on ROCm. Keep
  the script's explicit `ATTENTION_BACKEND` variable for targeted manual runs.
- motivation behind a new functionality: connector-plus-spec-decode coverage
  should exercise the combined feature path with a backend that can carry the
  metadata, while still leaving an explicit backend override available when the
  backend itself is under test.
- personal notes: this mirrors the MI300 NIXL correction and keeps the fix
  consistent across AMD generations.
- q & a:
  - Q: Why mention MI250/MI355 in a MI300-first branch?
    A: The same invalid override appeared in the shared AMD CI matrix. Leaving
    it behind would keep the same regression active on neighboring ROCm jobs.

## mi300_4: CrossLayer KV layout Distributed NixlConnector PD accuracy tests

- test group name: `mi300_4: CrossLayer KV layout Distributed NixlConnector PD accuracy tests`
- failure `tests/v1/kv_connector/nixl_integration/config_sweep_accuracy_test.sh`
- problem: same invalid `ROCM_ATTN` override as the base NIXL PD accuracy job,
  this time with `CROSS_LAYERS_BLOCKS=True`.
- proposed solution: preserve `CROSS_LAYERS_BLOCKS=True` and remove the
  invalid backend override so the backend is selected through ROCm's validated
  priority list.
- motivation behind a new functionality: cross-layer KV layout should validate
  the layout plus connector behavior, not whether an unrelated backend can be
  forced past its declared capability.
- personal notes: this shares the same root cause as the other NIXL failures,
  so I kept the fix centralized in CI command selection and script override
  handling.
- q & a:
  - Q: Are we losing coverage of `ROCM_ATTN`?
    A: No useful coverage is lost here because `ROCM_ATTN` advertises that it
    does not support KV connectors. Testing that combination only validates the
    guardrail.

## mi300_4: DP EP Distributed NixlConnector PD accuracy tests

- test group name: `mi300_4: DP EP Distributed NixlConnector PD accuracy tests`
- failure `tests/v1/kv_connector/nixl_integration/config_sweep_accuracy_test.sh`
- problem: `DP_EP=1 ROCM_ATTN=1` forced an invalid backend for the KV connector
  path and failed during engine startup.
- proposed solution: keep `DP_EP=1`, remove `ROCM_ATTN=1`, and allow backend
  auto-selection.
- motivation behind a new functionality: DP/EP coverage should be insensitive
  to backend availability unless the backend is the subject of the test.
- personal notes: the log showed multiple workers failing the same validation,
  so this is a configuration problem rather than a rank-local execution bug.
- q & a:
  - Q: Why not skip DP/EP on ROCm?
    A: The feature itself is valid on ROCm. Only the forced backend was invalid.

## mi300_4: Hybrid SSM NixlConnector PD accuracy tests

- test group name: `mi300_4: Hybrid SSM NixlConnector PD accuracy tests`
- failure `tests/v1/kv_connector/nixl_integration/config_sweep_accuracy_test.sh`
- problem: `HYBRID_SSM=1 ROCM_ATTN=1` forced the same unsupported KV connector
  backend.
- proposed solution: keep the Hybrid SSM mode and remove the invalid backend
  override. The selected backend must be one that supports KV connector layout.
- motivation behind a new functionality: Hybrid SSM should exercise its state
  transfer path with a supported attention implementation, not fail before the
  test reaches the state-management behavior.
- personal notes: this is the cleanest kind of CI fix: remove a stale override
  and let the runtime's capability checks do their job.
- q & a:
  - Q: Why is this not a test hack?
    A: The test still runs. It simply stops requesting a backend that the
    production configuration validator says cannot work.

## mi300_1: Spec Decode Eagle

- test group name: `mi300_1: Spec Decode Eagle`
- failure `tests/v1/e2e/spec_decode/test_spec_decode.py`
- problem: `ROCM_AITER_FA` hit
  `assert common_attn_metadata.seq_lens_cpu_upper_bound is not None` in
  `split_decodes_prefills_and_extends` for a decode-only batch.
- proposed solution: return the decode-only split result before reading
  `seq_lens_cpu_upper_bound`. That metadata is only needed to distinguish
  prefills from extends; all requests are decodes when
  `max_query_len <= decode_threshold`.
- motivation behind a new functionality: async speculative decode can avoid
  materializing CPU sequence-length upper bounds for pure decode batches. The
  splitter should not require prefill-only metadata on a path that never uses it.
- personal notes: I added a unit test that intentionally clears
  `seq_lens_cpu_upper_bound` for decode-only batches and covers both default
  and larger decode thresholds.
- q & a:
  - Q: Does this hide malformed metadata?
    A: No. It only skips metadata that is irrelevant once every row is known to
    be decode. Mixed or prefill batches still assert the field before using it.
  - Q: What about padded graph batches?
    A: The function returns `num_actual_tokens`, which is already the token
    count used by the existing decode-only branch.

## mi300_1: e2e Core (1 GPU)

- test group name: `mi300_1: e2e Core (1 GPU)`
- failure `tests/v1/e2e/general/test_cascade_attention.py`
- problem: the ROCm shard collected `test_cascade_attention[FLASH_ATTN]`, but
  the ROCm FlashAttention backend reached an engine-start assertion because the
  FlashAttention version is not detected for that backend path.
- proposed solution: parameterize the test by platform. CUDA keeps the existing
  `FLASH_ATTN` and `FLASHINFER` coverage. ROCm runs the same scenario with
  `backend="auto"`, allowing `rocm.py` to select a supported backend.
- motivation behind a new functionality: this preserves the workload shape on
  ROCm without pretending that CUDA FlashAttention is a ROCm backend. It also
  keeps the explicit CUDA cascade coverage intact.
- personal notes: I changed this away from a plain skip after review, because
  auto-selection gives ROCm useful coverage of the high-prefix batch pattern.
- q & a:
  - Q: Does ROCm still test cascade attention specifically?
    A: It tests the same high-prefix workload through the supported ROCm backend
    path. The FlashAttention-specific cascade backend remains CUDA coverage.
  - Q: Why not force `ROCM_ATTN`?
    A: Backend selection is exactly what broke here. The runtime should choose
    the valid ROCm backend for the model and platform.

## mi300_1: Kernels Attention Test 1

- test group name: `mi300_1: Kernels Attention Test 1`
- failure `tests/kernels/attention/test_attention_selector.py`
- problem: the test forced CUDA-only explicit backends while running on ROCm.
  `FLASHINFER` validation failed with `ImportError`, and local repro also showed
  the same problem for `FLASH_ATTN`.
- proposed solution: skip the explicit `FLASH_ATTN` case on ROCm and skip
  `FLASHINFER` on ROCm. CUDA still fails if backend coverage is expected but
  the dependency is missing.
- motivation behind a new functionality: the selector test should distinguish
  backend capability from package availability. ROCm unsupportedness should not
  masquerade as a selector logic regression.
- personal notes: I scoped the skips to ROCm after a reviewer challenge so CUDA
  dependency regressions remain visible.
- q & a:
  - Q: Could this hide a real CUDA failure?
    A: No. The skip is gated on `current_platform.is_rocm()`.

## mi300_1: Kernels Attention Test 2

- test group name: `mi300_1: Kernels Attention Test 2`
- failure `tests/kernels/attention/test_rocm_triton_attn_dsv4.py`
- problem: the ROCm sparse decode Triton kernel bitcast FNUZ FP8 cache bytes as
  `tl.float8e4b15`. AMD Triton reported that `fp8e4b15` is unsupported and
  listed `fp8e4b8` as the supported FNUZ E4M3 encoding.
- proposed solution: use `tl.float8e4b8` for the FNUZ path and keep
  `tl.float8e4nv` for the non-FNUZ path.
- motivation behind a new functionality: this aligns the kernel with Triton's
  AMD FP8 dtype mapping and the platform's `torch.float8_e4m3fnuz` cache dtype.
- personal notes: this is a source bug, not a test skip. It affects the decode
  path that reads packed DeepSeek V4 sparse attention cache.
- q & a:
  - Q: Why not change the test cache dtype instead?
    A: The test is already packing with `current_platform.fp8_dtype()`. The
    mismatch was in the kernel's Triton bitcast type.

## mi300_1: Language Models Tests (Standard)

- test group name: `mi300_1: Language Models Tests (Standard)`
- failure `tests/models/language/generation/test_common.py`
- problem: `openbmb/MiniCPM4.1-8B` imported a Transformers helper that no
  longer exists in the installed Transformers version.
- proposed solution: current main already carries a registry version guard for
  `MiniCPM4ForCausalLM`, so I am treating this as likely resolved by the
  merged main/PR stack unless local reproduction proves otherwise.
- motivation behind a new functionality: model registry guards are preferable
  to pinning the whole CI environment when a single upstream model repo is not
  yet compatible with a newer Transformers release.
- personal notes: pending targeted verification.
- q & a:
  - Q: Why not patch the model import?
    A: This is remote model code. A registry compatibility guard is safer and
    keeps the test suite honest about upstream support.

## mi300_1: Language Models Test (Extended Pooling)

- test group name: `mi300_1: Language Models Test (Extended Pooling)`
- failure `tests/models/language/pooling/test_token_classification.py`
- problem: ModernBERT token-classification output was numerically different
  from the expected tensor.
- proposed solution: PR stack includes ModernBERT changes, so this needs a
  targeted rerun before adding any new local change.
- motivation behind a new functionality: numeric model correctness should be
  fixed in model implementation logic, not by changing tolerances.
- personal notes: no threshold changes planned.
- q & a:
  - Q: Would increasing tolerance solve this?
    A: It would only hide the regression. The user explicitly asked not to do
    that, and I agree.

## mi325_1: V1 Spec Decode

- test group name: `mi325_1: V1 Spec Decode`
- failure `tests/v1/spec_decode/test_acceptance_length.py` and
  `tests/v1/spec_decode/test_eagle.py`
- problem: the log shows multiple symptoms: `ROCM_ATTN` rejected for attention
  sinks, missing `enable_multimodal_chat` on a test namespace, and unsupported
  FlashAttention metadata for multi-token speculative decode.
- proposed solution: do not increase acceptance thresholds. First verify the
  current merged branch because the namespace issue appears fixed on main. Then
  narrow backend parametrization so ROCm gpt-oss/spec-decode cases avoid
  backends that cannot support sinks or required metadata.
- motivation behind a new functionality: spec decode backend matrices need to
  express backend capabilities, not just enumerate every backend name.
- personal notes: this is next after the first MI300 patches and rebuild.
- q & a:
  - Q: Why not remove the gpt-oss case?
    A: The case is valuable. The right fix is capability-aware backend
    selection.
