# torchao dynamic FP8 quantization rejects MI355/gfx950

Target library: torchao 0.17 dynamic FP8 quantization

## Environment

- GPU: AMD Instinct MI355, gfx950
- ROCm: Buildkite AMD CI image from `vllm/amd-ci` build 8730
- torchao: `0.17.0`
- Test group: `mi355_1: Quantization`

## Problem

torchao dynamic FP8 activation plus FP8 weight quantization rejects MI355/gfx950 even though this platform supports OCP FP8 and torchao has an `is_MI350()` helper. The assertion only accepts CUDA SM 8.9+ or `is_MI300()`.

Buildkite failures:

```text
FAILED quantization/test_torchao.py::test_online_quant_config_dict_json
FAILED quantization/test_torchao.py::test_online_quant_config_file
FAILED quantization/test_torchao.py::test_reload_weights
```

Stack root:

```text
File ".../torchao/quantization/quant_api.py", line 1523, in _float8_dynamic_activation_float8_weight_transform
    assert is_sm_at_least_89() or is_MI300(), (
AssertionError: Float8 dynamic activation quantization is only supported on CUDA>=8.9 and MI300+
```

## Minimal Repro

```python
import torch
from torch import nn
from torchao.quantization import quantize_
from torchao.float8 import (
    Float8DynamicActivationFloat8WeightConfig,
    PerRow,
)

linear = nn.Linear(16, 16, bias=False, device="cuda", dtype=torch.bfloat16)
config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
quantize_(linear, config)
torch.cuda.synchronize()
print("quantized")
```

Run on MI355/gfx950 with torchao 0.17:

```bash
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python repro_torchao_gfx950_fp8.py
```

Equivalent vLLM test repro before the local skip:

```bash
cd /vllm-workspace/tests
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
pytest -q -s \
  'quantization/test_torchao.py::test_online_quant_config_dict_json' \
  --tb=short
```

## Expected Result

MI355/gfx950 should be accepted for dynamic FP8 quantization when the selected dtype is OCP `float8_e4m3fn`, or torchao should expose a clear capability query that vLLM can use to skip unsupported configurations.

## Actual Result

torchao raises the MI300-only assertion and rejects gfx950.

## Notes

The local vLLM branch skips only the three torchao dynamic FP8 tests on gfx950 while this is unresolved. That is a narrow test gate around an upstream capability check, not a product-code workaround.
