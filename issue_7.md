# HyperCLOVAX HF Remote Code Breaks Under Transformers 5.x

## Summary

The HF reference path for `naver-hyperclovax/HyperCLOVAX-SEED-Think-14B`
is not compatible with the current Transformers 5.x nightly used by AMD CI.
This makes `tests/models/language/generation/test_common.py` compare vLLM
against a broken HF baseline rather than testing a vLLM regression.

## Minimal Reproduction

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM

model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
```

With Transformers 5.x, model construction fails in the remote
`modeling_hyperclovax.py` RoPE setup because it indexes
`ROPE_INIT_FUNCTIONS["default"]`, but the `"default"` key no longer exists.

For completeness I also tried shimming a local `"default"` RoPE function. That
allows the HF model to construct, but the reference generation is still invalid:
one prompt produced repeated `!` tokens with `nan` top-logprob values, while
another produced degenerate repeated `L` / heading tokens. vLLM generated stable
text for the same prompts.

## Why This Is Not A vLLM Correctness Skip

The failing test is a cross-runner comparison. The vLLM path initializes and
generates; the HF reference path is the side that is incompatible with
Transformers 5.x. Keeping the row enabled with this HF version would only
measure remote-code breakage.

## Local Evidence

Focused command:

```bash
PYTHONPATH=. pytest -q -s \
  'tests/models/language/generation/test_common.py::test_models[True-False-5-32-naver-hyperclovax/HyperCLOVAX-SEED-Think-14B]' \
  'tests/models/language/generation/test_common.py::test_models[False-False-5-32-naver-hyperclovax/HyperCLOVAX-SEED-Think-14B]' \
  --tb=short -rs
```

Observed before adding the version guard:

- without a shim: `KeyError` at `ROPE_INIT_FUNCTIONS["default"]`
- with a shim: HF outputs invalid / degenerate tokens and `nan` logprobs while
  vLLM outputs stable text

## Temporary Test Action

The HF model registry entry now requires `transformers<=4.57` for HF-runner
tests, matching the compatibility boundary used for other broken remote-code
baselines. This should be removed once the upstream model code supports the
Transformers 5.x RoPE API and generation path.
