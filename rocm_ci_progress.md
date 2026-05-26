# ROCm CI Progress Index

Date: 2026-05-25

Current review branch: `wip-ci-fix-fin-proto`
Current base: `origin/main` at `d4004455d`

This note is the lightweight index for the proto branch. The more detailed
split rationale lives in `ROCM_CI_DECISIVE_BLOG.md`, and the proposed PR bodies
live in `PR_01.md` and `PR_03.md` through `PR_20.md`. `PR_02.md` was retired
after the Docker/non-root image cleanup files were removed from the proto diff.

## Current Patch Shape

- `git status --short --untracked-files=all`: 139 entries.
- `git diff --name-only`: 109 tracked modified files.
- Untracked new files: 30 entries.
- Staged files: 0.

The VS Code diff column should be understood as the status view, not plain
`git diff`, because plain `git diff` does not count untracked new files.

## Review Principle

Do not present this branch as one coherent PR. It mixes several kinds of work:

- Buildkite topology and hardware gating.
- ROCm allocator/runtime fixes.
- Distributed and KV connector stability work.
- Quantization capability gates.
- Kernel and attention numeric fixes.
- Model-specific ROCm compatibility fixes.
- A few new source files that look closer to feature work than CI repair.

Each proposed PR should answer:

- Which exact Buildkite group or pytest failure is addressed?
- Which changed files are required for that group?
- Which files are weakly justified and should be challenged?
- Which validation command should be run before posting?

## High-Risk Areas

### cuMem Sleep/Wake

Tracked in `PR_04.md`.

The strongest hypothesis is that the sleep test failure is not ordinary leaked
VRAM. The preceding server teardown reports memory back at zero, and the
failure happens inside the cuMem C++ allocator while MI300 still reports almost
all physical memory free. That suggests ROCm virtual-address reservation or
custom allocator state is the suspect.

The proposed fix keeps the real sleep test size and changes the allocator path
instead of shrinking the test.

The ROCm test-side assertion is no longer a blanket waiver. The C++ allocator
fix should make the existing post-sleep memory assertions pass on ROCm too; if
they fail again, that should be treated as source behavior to debug.

### NCCL/HIP Invalid Argument

Tracked in `PR_03.md`.

This remains weak-confidence unless validated with an A/B against `origin/main`
using the exact Buildkite final commands. Several affected groups are green in
nightly main but failed on this branch, so test-only workarounds are especially
suspicious here.

### Llama4 Eagle MM Runtime

Tracked as an issue in `ROCM_CI_DECISIVE_BLOG.md`.

The Llama4 Eagle MM heavy rows spend a large amount of time rendering
conversations and compiling Triton paths. This should become a profiling issue,
not an unbounded Buildkite timeout increase.

### MI355 AITER Accuracy

Tracked as an issue in `ROCM_CI_DECISIVE_BLOG.md`.

The MI355-specific AITER inaccuracy should stay architecture-specific. Do not
generalize that tolerance or backend gate to MI300/MI325 unless the same
failure is proven there.

## Immediate Review Order

1. `PR_04.md`: cuMem allocator and sleep-mode failure.
2. `PR_03.md`: distributed/NCCL failures, but only with exact-command A/B.
3. `PR_01.md`: Buildkite topology, sharding, and hardware gates.
4. `PR_06.md`: quantization gates, especially MI250/MI300 capability splits.
5. `PR_11.md`: spec decode/EAGLE acceptance behavior.

Everything else should be kept as a candidate split until the owning test group
has a clean failure-to-fix story.
