# PR 4: Granite Tool Calls

Commit message: `Guide Granite tool-call formatting`

## Summary

Add a Granite 3.0-specific system prompt and apply server-config system prompts to the tool-call tests. This nudges Granite to emit the expected `<|tool_call|>` JSON format instead of prose about how to call a tool.

## Files

| File | Diff |
|---|---|
| `tests/tool_use/utils.py` | Add `system_prompt` for `granite-3.0-8b`. |
| `tests/tool_use/test_tool_calls.py` | Pass `server_config` into tool-call tests and wrap messages with `ensure_system_prompt(...)`. |

## Buildkite Test Groups

| Group | Hardware section | Command |
|---|---|---|
| `Entrypoints Integration (API Server 2)` | MI355 and MI300 | `pytest -v -s tool_use` |

## Pytest Commands

```bash
cd /app/vllm/tests

CUDA_VISIBLE_DEVICES=2 VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONPATH=.. pytest -q -s \
  tool_use/test_tool_calls.py \
  --models granite-3.0-8b

CUDA_VISIBLE_DEVICES=2 VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONPATH=.. pytest -q -s \
  tool_use/test_parallel_tool_calls.py \
  --models granite-3.0-8b
```
