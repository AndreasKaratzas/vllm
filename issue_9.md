# Superseded: Ray RLHF NCCL visibility mismatch

This note was originally drafted as an external RCCL direct-P2P issue for
`examples/rl/rlhf_nccl.py`. The deeper local investigation changed the
diagnosis.

## Updated Finding

The trainer actor does not need to reserve two GPUs, and direct RCCL P2P is not
the validated root cause on the MI355 node.

The actual local failure came from process visibility state:

- The driver imports vLLM before `ray.init()`.
- vLLM's ROCm platform import mirrors broad `HIP_VISIBLE_DEVICES` into
  `CUDA_VISIBLE_DEVICES`.
- Ray later narrows `HIP_VISIBLE_DEVICES` inside GPU actors, but the actor can
  still inherit the driver's broad `CUDA_VISIBLE_DEVICES`.
- A vLLM import inside the Ray worker then sees mismatched visibility variables
  and fails before the intended NCCL path is reached.

## Local Resolution

`vllm/platforms/rocm.py` now recognizes Ray worker processes via
`RAY_JOB_ID` and `RAY_RAYLET_PID`. If a Ray ROCm worker has both visibility
variables set but they differ, vLLM mirrors Ray's narrowed
`HIP_VISIBLE_DEVICES` into `CUDA_VISIBLE_DEVICES`.

With that fix, the full RLHF NCCL example passes with:

```bash
env -u CUDA_VISIBLE_DEVICES -u NCCL_P2P_DISABLE \
  HIP_VISIBLE_DEVICES=0,1,2,3 ROCR_VISIBLE_DEVICES=0,1,2,3 \
  PYTHONPATH=/app/vllm RAY_DEDUP_LOGS=0 NCCL_DEBUG=INFO \
  VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  python3 examples/rl/rlhf_nccl.py
```

The passing run uses direct RCCL P2P/IPC for the three-rank weight-transfer
communicator.
