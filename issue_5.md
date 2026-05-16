# Aria HF Processor Metadata Is Missing for Current Transformers

## Affected Test Group

Multi-Modal Models (Extended Generation 3)

## Minimal Reproduction

```python
from vllm.model_executor.models.aria import AriaProcessor

AriaProcessor.from_pretrained("rhymes-ai/Aria")
```

The current model repository does not provide `vision_processor.py`, and the
processor load fails with:

```text
OSError: rhymes-ai/Aria does not appear to have a file named vision_processor.py
```

## Why This Is External

The failure happens while loading the model processor metadata from the Hugging
Face repository. It is not a ROCm kernel, scheduler, or vLLM model-runner
failure.

## Local vLLM Test Mitigation

The Aria extended-generation test is marked skipped with a direct issue note
until the upstream model repository publishes the processor file required by
the current Transformers/vLLM processor load path.
