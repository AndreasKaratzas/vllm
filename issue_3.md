# AITER MoE JIT compiles serially for minutes per variant on MI355

Target library: AITER JIT fused MoE kernels

## Environment

- GPU: AMD Instinct MI355, gfx950
- ROCm: Buildkite AMD CI image from `vllm/amd-ci` build 8730
- Test group: `mi355_1: Kernels MoE Test`
- Failing log: shard timed out after reaching an AITER modular MoE combination test

## Problem

The MoE kernel shard does not merely run slowly; it spends minutes compiling a single AITER JIT MoE module and then starts compiling another variant in the same pytest row. The Buildkite job is killed while the second AITER module is still building.

Relevant log excerpt:

```text
kernels/moe/test_modular_kernel_combinations.py::test_modular_kernel_combinations_singlegpu[2048-1024-32-dtype4-quant_config4-MoEPrepareAndFinalizeNoDPEPModular-AiterExperts-1]
[aiter] start build [module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_tensor_mulWeightStage2]
[aiter] finish build [module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_tensor_mulWeightStage2], cost 282.9s
[aiter] start build [module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_tensor_mulWeightStage1]
# Received cancellation signal, interrupting
```

The first AITER JIT module takes 282.9 seconds. The second variant is still compiling when Buildkite terminates the job around the global timeout.

## Repro Command

Run inside the ROCm CI image with AITER installed and a cold AITER JIT cache:

```bash
cd /vllm-workspace/tests
rm -rf /usr/local/lib/python3.12/dist-packages/aiter/jit/build/module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_tensor_mulWeightStage1
rm -rf /usr/local/lib/python3.12/dist-packages/aiter/jit/build/module_moe_ck2stages_f8_f8_preshuffle_off_b16_silu_per_tensor_mulWeightStage2
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
pytest -q -s \
  'kernels/moe/test_modular_kernel_combinations.py::test_modular_kernel_combinations_singlegpu[2048-1024-32-dtype4-quant_config4-MoEPrepareAndFinalizeNoDPEPModular-AiterExperts-1]' \
  --tb=short
```

## Expected Result

A single AITER MoE JIT variant should either compile quickly enough for normal CI first-use coverage, reuse a prebuilt artifact, or expose a prebuild/cache command that CI can run once before the test shard.

## Actual Result

The first-use compile takes several minutes per module variant. In a large sharded test run, this consumes the job budget before the shard finishes.

## Notes

- This is not a request to raise the Buildkite timeout. The timeout is a symptom of pathological first-use compile cost.
- vLLM should continue to test AITER MoE behavior, but CI needs either prebuilt AITER modules or a smaller set of cold-JIT variants if the library cannot reduce compile time.
