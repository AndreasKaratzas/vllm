# Llama-4 Scout MXFP4 two-layer fixture has invalid Transformers 5 config

## Affected Test Group

Kernels MoE Test

## Minimal Reproduction

```python
from transformers import AutoConfig

AutoConfig.from_pretrained(
    "fxmarty/Llama-4-Scout-17B-16E-Instruct-2-layers-mxfp4"
)
```

With current Transformers 5 in AMD CI, this fails while constructing the
Llama 4 text config:

```text
huggingface_hub.errors.StrictDataclassFieldValidationError:
Validation error for field 'attn_temperature_tuning':
    TypeError: Field 'attn_temperature_tuning' expected bool, got int (value: 4)
```

The downloaded `config.json` contains:

```text
text_config.attn_temperature_tuning = 4
text_config.no_rope_layer_interval = <missing>
```

## Why This Is External

The failure happens in `AutoConfig.from_pretrained()` before vLLM creates an
engine, loads weights, or applies `hf_overrides`. The test fixture config is
not type-valid for current Transformers strict dataclass validation: upstream
`Llama4TextConfig.attn_temperature_tuning` is a boolean field, but the fixture
stores the integer `4`.

The value appears to be the default `no_rope_layer_interval`, not a valid value
for `attn_temperature_tuning`. vLLM cannot reliably infer the intended upstream
config without silently changing user model metadata.

## Local vLLM Test Mitigation

The single invalid fixture parameter is marked `xfail(strict=True)` in
`tests/kernels/moe/test_ocp_mx_moe.py`. This keeps the MoE kernel group focused
on real vLLM kernel regressions while making the marker self-removing: if the
fixture config is fixed upstream, the strict XPASS will fail CI and force the
marker to be deleted.
