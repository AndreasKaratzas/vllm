# OpenAPI Boundary PR

Commit message: `Return 400 for invalid schema inputs`

Files:

- `vllm/entrypoints/chat_utils.py`
- `vllm/entrypoints/openai/chat_completion/batch_serving.py`
- `vllm/entrypoints/openai/chat_completion/protocol.py`
- `vllm/entrypoints/openai/completion/serving.py`
- `vllm/entrypoints/openai/engine/serving.py`
- `vllm/entrypoints/serve/disagg/protocol.py`
- `vllm/entrypoints/serve/disagg/serving.py`
- `vllm/entrypoints/serve/render/serving.py`

PR title: Return client errors for invalid generated API inputs

PR body:

## Summary

- Reject malformed assistant `tool_calls` with `VLLMValidationError`.
- Require non-empty conversations only for batch chat requests.
- Convert post-Pydantic render, sampling, token, and msgpack validation failures
  into 400 `ErrorResponse`s.
- Reject `/inference/v1/generate` requests with empty `token_ids` or `n` above
  the server `max_num_seqs`.

## Why

Schemathesis generated malformed but parseable requests that reached
post-Pydantic code paths and escaped as 500s or timeouts. These are client input
errors, not server crashes. The changes keep static shape checks in protocol
models and handle model/tokenizer/runtime-dependent validation at serving
boundaries.

## Validation

- `pytest -v -s entrypoints/openai/test_openai_schema.py::test_openapi_stateless`
  passed with 24 subtests.
- OpenAI Part 3 probe passed after these boundary fixes.
- `python -m py_compile` passed for the modified Python serving files.
- `git diff --check` passed.

## Reviewer Q&A

Q: Why not rely on the global FastAPI exception handler?
A: Known invalid client input should return a normal endpoint `ErrorResponse`.
The global handler is a last resort and would log these as uncaught exceptions.

Q: Why not put all of this in Pydantic validators?
A: Some checks need server state or later processing: tokenizer/template
rendering, `max_num_seqs`, per-conversation batch conversion, sampling params,
and msgpack IPC encoding.

Q: Why scope `min_length=1` to batch chat instead of all chat requests?
A: A global `ChatCompletionRequest.messages` constraint broke Anthropic
malformed-request conversion paths. Batch chat is the endpoint that indexes the
last message for each nested conversation, so the constraint belongs there.

Q: Why catch `ValueError` in completion generation?
A: Invalid token prompts can surface while consuming the non-streaming engine
generator. Returning 400 there matches the existing serving pattern for known
client validation failures after scheduling.
