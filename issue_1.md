# MiniCPM4.1 HF Remote Code Breaks With Current Transformers

## Affected Test Group

Language Models Tests (Standard)

## Minimal Reproduction

```python
from transformers import AutoModelForCausalLM

AutoModelForCausalLM.from_pretrained(
    "openbmb/MiniCPM4.1-8B",
    trust_remote_code=True,
)
```

With current Transformers in AMD CI, this fails while importing the remote
`modeling_minicpm.py`:

```text
ImportError: cannot import name 'is_torch_fx_available' from
'transformers.utils.import_utils'
```

After locally providing an equivalent `is_torch_fx_available` helper, the next
failure is:

```text
ValueError: MiniCPMForCausalLM does not support Flash Attention 2 yet.
```

After requesting `_attn_implementation="eager"`, generation still fails in two
ways depending on the cached dynamic class state:

```text
AttributeError: 'MiniCPMForCausalLM' object has no attribute 'generate'
```

or, when `generate` is present:

```text
ValueError: Attention weights should be of size (1, 32, 1, 23), but is
torch.Size([1, 32, 23, 45])
```

## Why This Is External

The failing path is entirely inside the Hugging Face reference model load, before
vLLM initializes. The MiniCPM4.1 remote config sets
`_attn_implementation = "flash_attention_2"`, but the remote
`MiniCPMForCausalLM` class does not declare Flash Attention 2 support to
Transformers. The same remote file also imports a Transformers helper that no
longer exists.

## Local vLLM Test Mitigation

The model registry now marks this HF reference as compatible only through
Transformers 4.57 and points to this issue note. That prevents false vLLM
regressions from an HF baseline that cannot be loaded or trusted under
Transformers 5.x. The vLLM model path is not modified by this guard.
