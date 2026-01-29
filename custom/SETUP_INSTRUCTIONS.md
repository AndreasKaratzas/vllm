# Setup Instructions

Quick reference for setting up this branch on a new system.

## Add Remote and Fetch Branch

```bash
# Navigate to vLLM repository
cd /path/to/vllm

# Add Andreas's fork as remote
git remote add andreas https://github.com/AndreasKaratzas/vllm.git

# Fetch all branches from the fork
git fetch andreas

# Checkout the fix branch
git checkout fix-gfx950-prefix-cache-determinism

# Or create a local tracking branch
git checkout -b fix-gfx950-prefix-cache-determinism andreas/fix-gfx950-prefix-cache-determinism
```

## Alternative: Clone Directly

```bash
# Clone Andreas's fork
git clone https://github.com/AndreasKaratzas/vllm.git
cd vllm

# Checkout the fix branch
git checkout fix-gfx950-prefix-cache-determinism
```

## Verify Setup

```bash
# Check current branch
git branch

# Check remote configuration
git remote -v

# View recent commits
git log --oneline -5
```

## Test the Fixes

### Test TRITON Backend (Fixed)
```bash
cd /path/to/vllm
VLLM_ROCM_USE_AITER=0 python3 custom/test_simple_compare.py
# Expected: All runs produce identical tokens ✅
```

### Test AITER Backend (Known Issue)
```bash
cd /path/to/vllm
VLLM_ROCM_USE_AITER=1 python3 custom/debug_aiter_cache.py
# Expected: Run 1 differs from Runs 2-3 ❌
```

## Branch Information

- **Branch Name**: `fix-gfx950-prefix-cache-determinism`
- **Remote**: `andreas` (https://github.com/AndreasKaratzas/vllm.git)
- **Base**: vLLM v0.15.0rc2.dev44+gf210f0b7b
- **Target GPU**: AMD MI355X (gfx950)

## Files Modified

### Core Fixes:
- `vllm/v1/attention/ops/triton_unified_attention.py` - BF16 precision fix
- `vllm/v1/core/block_pool.py` - Deterministic block selection
- `vllm/v1/core/single_type_kv_cache_manager.py` - GPU sync for gfx950

### Debugging/Logging:
- `vllm/v1/attention/backends/rocm_aiter_fa.py` - Attention tensor logging
- `vllm/v1/sample/sampler.py` - Logits and sampling logging
- `vllm/model_executor/models/qwen3.py` - Layer-wise activation logging

### Custom Directory:
- `custom/` - All test scripts, debug tools, and documentation

## Key Documentation

- `custom/AITER_NUMERICAL_ANALYSIS.md` - Complete numerical analysis
- `custom/AITER_FINAL_FINDINGS.md` - Bug summary and findings
- `custom/BF16_PREFIX_CACHE_FIX_SUMMARY.md` - TRITON fix explanation
