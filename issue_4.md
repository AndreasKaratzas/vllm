# Skywork R1V HF Remote Code Breaks With Transformers 5

## Affected Test Group

Multi-Modal Models (Extended Generation 2)

## Minimal Reproduction

```python
from transformers import AutoModelForImageTextToText

AutoModelForImageTextToText.from_pretrained(
    "Skywork/Skywork-R1V-38B",
    trust_remote_code=True,
)
```

With the current AMD CI Transformers version, the HF reference model fails in
the remote model load path before vLLM output comparison is meaningful.

## Current Root Cause

The remote `SkyworkChatModel.__init__` does not call `self.post_init()`, so
`all_tied_weights_keys` is not initialized. Transformers 5 uses that attribute
while moving missing meta-device parameters to the real device.

## Why This Is External

The failing path is the Hugging Face reference model. vLLM has not initialized
an engine yet, and the failure is caused by a remote-code contract change with
Transformers 5.

## Local vLLM Test Mitigation

The registry marks this HF reference as requiring Transformers <= 4.57 and
points at this issue note. This prevents a broken HF baseline from reporting as
a vLLM ROCm generation regression.
