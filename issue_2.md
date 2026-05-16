# TorchAO 0.17.0 does not recognize gfx950 as MI300+ for FP8 dynamic quantization

## Summary

`torchao==0.17.0` rejects FP8 dynamic activation quantization on MI355
(`gfx950`) even though the error message says the path is supported on
`MI300+`. The local device reports `gcnArchName='gfx950:sramecc+:xnack-'`.

## Minimal Reproduction

```python
import torch
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    PerRow,
    quantize_,
)

linear = torch.nn.Linear(
    16, 16, bias=False, device="cuda", dtype=torch.bfloat16
)
config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
quantize_(linear, config)
```

On MI355 with `torchao==0.17.0`, this raises:

```text
AssertionError: Float8 dynamic activation quantization is only supported on CUDA>=8.9 and MI300+
```

## Why this is isolated to TorchAO detection

`torchao.utils.is_MI300()` and the duplicate helper imported into
`torchao.float8.inference` only match `gfx940`, `gfx941`, and `gfx942`.
They return `False` on `gfx950`, before vLLM model execution reaches a real
FP8 kernel correctness failure. Patching those helpers to return `True` for
`gfx950` allows vLLM's existing TorchAO online FP8 quantization tests to run
and pass.

## Local Validation

```bash
HIP_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python3 - <<'PY'
import torch
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow
from vllm.model_executor.layers.quantization.torchao import torchao_quantize_param_data

param = torch.nn.Parameter(
    torch.empty((16, 16), device="cuda", dtype=torch.bfloat16),
    requires_grad=False,
)
config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
weight = torchao_quantize_param_data(param, config)
print(type(weight), weight.device, weight.shape)
PY
```

Focused tests:

```bash
cd tests
HIP_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=3 PYTHONPATH=.. \
  VLLM_TEST_FORCE_LOAD_FORMAT=auto \
  pytest -q -s \
  quantization/test_torchao.py::test_online_quant_config_dict_json \
  quantization/test_torchao.py::test_online_quant_config_file \
  quantization/test_torchao.py::test_reload_weights
```

Result: `3 passed`.
