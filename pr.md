# Proposed PR Split

This branch is easier to review as four small PRs. Each PR below has its own
commit message, file list, title/body, and reviewer Q&A.

- `pr-rocm-sleep.md`: ROCm sleep-mode memory release.
- `pr-samplers-rocm-memory.md`: EngineCore teardown and sampler runner memory
  guard.
- `pr-openapi-boundary.md`: invalid OpenAPI/schema inputs return client errors.
- `pr-api-server-2-stability.md`: Granite tool-use and test URL helper fixes.

`rocm-log.md` is an investigation artifact. Include it only if reviewers want
the validation trail in the PR; otherwise leave it out of the code commits.
