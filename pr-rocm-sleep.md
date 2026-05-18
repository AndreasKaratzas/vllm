# ROCm Sleep PR

Commit message: `Fix ROCm sleep memory release`

Files:

- `csrc/cumem_allocator.cpp`
- `vllm/device_allocator/cumem.py`

PR title: Fix ROCm sleep-mode memory release

PR body:

## Summary

- Free and re-reserve the ROCm virtual address only in the sleep-mode unmap path.
- Keep wake-up compatible by re-reserving the same virtual address.
- Synchronize before and after sleep unmapping and wake-up remapping.

## Why

`test_sleep_mode` intermittently failed on MI300 with a HIP OOM while the GPU
reported almost all VRAM free. ROCm was releasing physical handles, but the
reserved virtual address range kept the memory from becoming usable. Cycling
the VA reservation after sleep unmap makes the freed memory visible while still
preserving the address needed by `wake_up`.

## Validation

- Rebuilt native code with `../vllm-scripts/rebuild.sh` from `/app/vllm`.
- `entrypoints/serve/instrumentator/test_sleep.py::test_sleep_mode` passed after
  pruning the broader reserve retry.
- Full API Server 2 validation passed 10/10 with this sleep fix.

## Reviewer Q&A

Q: Why is the VA cycle only in `python_unmap_and_release`?
A: The failure is sleep-specific. A broader allocator free-path change risks
normal free behavior, including tensor-parallel paths. This keeps the workaround
at the sleep API boundary.

Q: Why re-reserve the same virtual address?
A: The allocator stores pointers and `wake_up` remaps handles back to those
addresses. Reserving a different VA would break wake-up.

Q: Why add synchronizes in Python?
A: Sleep and wake are user-visible state transitions. The sync before sleep
ensures queued GPU work is done before we copy and unmap allocations. The syncs
after sleep and wake ensure release/remap completes before the API reports
success.
