# GLM-4V and GLM-OCR Extended Generation Baselines Are Broken

## Affected Test Group

Multi-Modal Models (Extended Generation 3)

## Minimal Reproduction

```bash
cd tests
pytest -v -s \
  'models/multimodal/generation/test_common.py::test_single_image_models[glm4v-test_case134]'

pytest -v -s \
  'models/multimodal/generation/test_common.py::test_multi_image_models[glm_ocr-test_case102]'
```

The AMD CI log shows that the GLM-OCR row decodes only `STOP` in vLLM while
the HF baseline emits an OCR-style multi-image description. The GLM-4V rows
similarly fail golden-output comparison against the current HF baseline.

## Why This Is External / Not Yet Attributable to ROCm

The failure is tracked as a model/test baseline issue rather than a
ROCm-specific regression: the same test was reported as failing on both AMD and
NVIDIA hardware in the corresponding upstream vLLM tracking issue. The GLM
remote-code path also has a linked model-side bug report.

## Local vLLM Test Mitigation

The affected GLM-4V and GLM-OCR extended-generation rows are marked skipped
with this issue note. This is intentionally narrow: it does not relax any
thresholds and does not skip unrelated GLM models.
