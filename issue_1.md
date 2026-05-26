# DeepEP low-latency FP8 MoE output mismatch on MI355/gfx950

Target library: DeepEP low-latency MoE kernels

## Environment

- GPU: AMD Instinct MI355, gfx950
- ROCm: Buildkite AMD CI image from `vllm/amd-ci` build 8730
- Test group: `mi355_2: Kernels FP8 MoE Test (2xH100-2xMI355)`
- Local note: the shell on the debug host does not have `deep_ep` installed, so the exact rows skip locally there. The failure reproduces in the Buildkite DeepEP-enabled image.

## Problem

The vLLM DeepEP low-latency FP8 MoE test compares the DeepEP path against the reference MoE output with `atol=rtol=0.06`. On MI355, several low-latency rows exceed that tolerance. Some failures are tiny one-element misses, while larger-token rows have broad mismatches.

Representative Buildkite failures:

```text
FAILED kernels/moe/test_deepep_moe.py::test_low_latency_deep_ep_moe[True-world_dp_size0-6-32-1-128-2560-dtype1]
FAILED kernels/moe/test_deepep_moe.py::test_low_latency_deep_ep_moe[True-world_dp_size0-6-32-3-1024-2560-dtype1]
FAILED kernels/moe/test_deepep_moe.py::test_low_latency_deep_ep_moe[True-world_dp_size0-6-32-32-128-2560-dtype1]
FAILED kernels/moe/test_deepep_moe.py::test_low_latency_deep_ep_moe[True-world_dp_size0-6-32-45-512-2560-dtype1]
FAILED kernels/moe/test_deepep_moe.py::test_low_latency_deep_ep_moe[True-world_dp_size0-6-32-64-1024-2560-dtype1]
FAILED kernels/moe/test_deepep_moe.py::test_low_latency_deep_ep_moe[True-world_dp_size0-6-32-222-1024-2560-dtype0]
FAILED kernels/moe/test_deepep_moe.py::test_low_latency_deep_ep_moe[True-world_dp_size0-6-32-222-1024-2560-dtype1]
FAILED kernels/moe/test_deepep_moe.py::test_low_latency_deep_ep_moe[False-world_dp_size0-6-32-222-1024-2560-dtype0]
FAILED kernels/moe/test_deepep_moe.py::test_low_latency_deep_ep_moe[False-world_dp_size0-6-32-222-1024-2560-dtype1]
```

Representative assertion output:

```text
Mismatched elements: 1 / 2560 (0.0%)
Greatest absolute difference: 0.0703125 at index (0, 2243) (up to 0.06 allowed)

Mismatched elements: 1072 / 7680 (14.0%)
Greatest absolute difference: 0.2890625 at index (1, 1525) (up to 0.06 allowed)

Mismatched elements: 16220 / 163840 (9.9%)
Greatest absolute difference: 0.421875 at index (47, 144) (up to 0.06 allowed)
```

## Repro Command

Run inside the ROCm CI image where DeepEP is installed:

```bash
cd /vllm-workspace/tests
HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1 \
pytest -q -s \
  'kernels/moe/test_deepep_moe.py::test_low_latency_deep_ep_moe[True-world_dp_size0-6-32-1-128-2560-dtype1]' \
  'kernels/moe/test_deepep_moe.py::test_low_latency_deep_ep_moe[True-world_dp_size0-6-32-3-1024-2560-dtype1]' \
  'kernels/moe/test_deepep_moe.py::test_low_latency_deep_ep_moe[False-world_dp_size0-6-32-222-1024-2560-dtype1]' \
  --tb=short
```

On the debug host without DeepEP, the same command skips with `Requires deep_ep kernels`; it must be run in the Buildkite DeepEP image.

## Expected Result

The DeepEP low-latency FP8 MoE output should be close to the test reference under the existing FP8 tolerance contract, or the library should document a different expected numerical envelope for gfx950 low-latency FP8 dispatch.

## Actual Result

The DeepEP low-latency FP8 path returns values outside the existing tolerance on MI355/gfx950, with broad mismatches on larger `M` rows.

## Notes

- The high-throughput DeepEP rows are not the failing family in this log.
- vLLM should not hide this by weakening every DeepEP assertion. If DeepEP's MI355 low-latency FP8 path has a different accumulation contract, the test reference or backend selection should be updated intentionally around that contract.
