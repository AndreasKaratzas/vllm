# Sampler ROCm Teardown PR

Commit message: `Fix ROCm runner teardown`

Files:

- `vllm/v1/engine/core.py`
- `tests/conftest.py`

PR title: Fix ROCm runner teardown between sampler tests

PR body:

## Summary

- Unfreeze EngineCore startup objects and run shared distributed/memory cleanup
  when an EngineCore process shuts down.
- On ROCm, wait after `VllmRunner` exit until GPU memory is below the next
  runner's startup threshold.

## Why

The MI250 sampler nightly failures were not beam-search assertion failures. The
first engine passed, then the next engines failed while constructing `LLM`
because free memory was just below the V1 startup guard:
`58.33-58.44/63.98 GiB` free versus `58.87 GiB` required. The same logs also
showed process-group teardown warnings after EngineCore shutdown.

This keeps teardown in the engine process itself and prevents consecutive
`VllmRunner` tests from racing ROCm's delayed VRAM release.

## Validation

- Targeted sampler failures passed before the EngineCore cleanup:
  `5 passed, 17 warnings in 246.99s`.
- Targeted sampler failures passed after the EngineCore cleanup:
  `5 passed, 17 warnings in 206.51s`.
- The post-cleanup run no longer emitted the
  `destroy_process_group() was not called` warning.

## Reviewer Q&A

Q: Why keep the explicit `gc.unfreeze()` if cleanup also unfreezes?
A: EngineCore startup freezes GC-managed objects, so the matching unfreeze
belongs in `EngineCore.shutdown()`. The shared cleanup still unfreezes
defensively before collecting and emptying caches; calling it twice is harmless.

Q: Why not just rely on the existing cleanup fixture?
A: The failed Buildkite command did not enable `VLLM_TEST_CLEAN_GPU_MEMORY`.
`VllmRunner` already owns engine shutdown, so it is the narrowest place to make
sequential runner tests wait for startup-safe memory.

Q: Why use `1 - gpu_memory_utilization` as the wait threshold?
A: V1 startup requires `free_memory >= total * gpu_memory_utilization`. Waiting
until used memory is at most the complementary ratio directly matches that
guard.

Q: Is the wait hiding a leak?
A: No. The run shows memory dropping after one polling interval. If memory does
not drop, the helper still fails with a clear timeout instead of letting the next
test fail with a misleading startup error.

Q: Can cleanup wait indefinitely?
A: No. The ROCm memory wait is explicitly bounded to 120 seconds.
