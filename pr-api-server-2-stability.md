# API Server 2 Stability PR

Commit message: `Stabilize API Server 2 tests`

Files:

- `examples/tool_chat_template_granite.jinja`
- `tests/tool_use/test_tool_calls.py`
- `tests/tool_use/utils.py`
- `tests/utils.py`

PR title: Stabilize Granite tool-use and test URL construction

PR body:

## Summary

- Add a generic Granite 3.0 tool-use system instruction when tools are present
  and no system message is provided.
- Keep the Dallas weather test's state-abbreviation steering in the tool-use
  test config instead of the public example template.
- Normalize `RemoteVLLMServer.url_for` path segments so callers can pass
  `"v2/embed"` or `"/v2/embed"` and get the same URL.

## Why

After the ROCm sleep fix, API Server 2 exposed deterministic failures in
the same Buildkite group. Granite 3.0 sometimes answered in plain text because
the custom template listed tools without instructing the model to call them.
The test-specific Dallas weather argument hint now stays in the test harness,
not the reusable example template. The optional middleware tests also built
`//v2/embed`, which missed the
registered `/v2/embed` route and returned 404 before auth was tested.

## Validation

- Granite 3.0 tool-use targeted tests passed.
- After moving the state-abbreviation hint out of the template:
  `pytest -v -s tool_use/test_tool_calls.py::test_tool_call_and_choice --models granite-3.0-8b`
  passed.
- Optional middleware `/v2/embed` targeted tests passed.
- Full API Server 2 validation passed 10/10.

## Reviewer Q&A

Q: Is `tool_chat_template_granite.jinja` used by tests?
A: Yes. `tests/tool_use/utils.py` maps `granite-3.0-8b` to this template, and
API Server 2 runs `pytest -v -s tool_use`.

Q: Why remove the state-abbreviation instruction from the Granite template?
A: That instruction belongs to the Dallas weather test's expected arguments,
not to a reusable example template. The public template now only provides the
generic Granite tool-call format instruction.

Q: Why change `url_for` instead of the two test call sites?
A: Existing tests use both leading-slash and no-leading-slash styles. The helper
should canonicalize path segments so all callers hit the intended route.

Q: Was the URL bug flaky?
A: The bad URL construction was deterministic for leading-slash inputs. The
visibility varied by caller and routing normalization. The fix makes the URL
canonical before the request is sent.
