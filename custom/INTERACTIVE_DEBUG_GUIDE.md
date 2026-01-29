# Interactive Debugging Guide: MI355X Prefix Caching Bug

Since this happens with **all attention backends**, the issue is likely in:
1. How the KV cache is written after the first request
2. How the KV cache is read on subsequent requests
3. Some state/configuration that changes between requests
4. Platform-specific (gfx950) cache behavior

Let's debug this systematically with checkpoints.

---

## 🔍 Checkpoint 1: Verify Cache Hit/Miss

**Question**: Is the prefix cache actually being used?

**Run**:
```bash
cd /app
python debug_step1_cache_hit.py
```

**What to look for**:
- Does Run 1 differ from Run 2?
- Does the server log show cache hits?

**Expected on MI355X (with bug)**:
```
✗ Tokens DIFFER (bug present)
  Run 1: 'The European Union consists of 27 member states'
  Run 2: 'The European Union (EU) consists of **2'
```

**If they match**: Bug doesn't reproduce in this simpler setup → investigate test differences

**If they differ**: Bug confirmed → proceed to Checkpoint 2

---

## 🔍 Checkpoint 2: Test Without Prefix Caching

**Question**: Is this a prefix caching issue or something more general?

**Run**:
```bash
cd /app
python debug_step2_same_input.py
```

**What to look for**:
- Do multiple requests WITHOUT prefix caching produce identical results?

**Expected (no bug)**:
```
✓ ALL REQUESTS IDENTICAL
```

**If they differ**: Something else is broken (not prefix caching) → investigate non-determinism

**If they match**: Confirms issue is specific to prefix caching → proceed to Checkpoint 3

---

## 🔍 Checkpoint 3: Check Cache Block Allocation

**Question**: Are cache blocks being allocated correctly?

Let's add logging to see what's happening with cache blocks.

**Modify the test** to enable debug logging:
```python
base_args = [
    "--dtype", "float16",
    "--max-model-len", "512",
    "--enforce-eager",
    "--generation-config", "vllm",
    "--max_num_seqs", "1",
]

# Before starting server, set environment:
import os
os.environ['VLLM_LOGGING_LEVEL'] = 'DEBUG'
```

**Run the test** and look for:
- "cache hit" messages
- "cache miss" messages
- Block allocation logs
- "reusing cached blocks" messages

**Save the logs**:
```bash
python debug_step1_cache_hit.py 2>&1 | tee debug_checkpoint3.log
```

---

## 🔍 Checkpoint 4: Compare KV Cache Values

**Question**: Are the KV cache values themselves different?

This requires instrumenting the code to dump KV cache values.

**Add prints** in the cache write/read locations:
1. Where KV is written after first request
2. Where KV is read on second request

**Files to check**:
- `/app/vllm/vllm/v1/worker/gpu_model_runner.py`
- `/app/vllm/vllm/v1/core/single_type_kv_cache_manager.py`

Would you like me to add instrumentation for this?

---

## 🔍 Checkpoint 5: Check Platform-Specific Behavior

**Question**: Is there gfx950-specific cache handling?

**Search for**:
```bash
cd /app/vllm
grep -r "gfx950\|MI355" --include="*.py" --include="*.cpp" --include="*.cu"
```

**Look for**:
- Special cache size calculations
- gfx950-specific block allocation
- MI355X-specific workarounds

---

## 🔍 Checkpoint 6: Compare with MI325X

**Question**: What's different between MI325X (working) and MI355X (broken)?

**From your environment info**:
- MI355X: `gfx950`
- MI325X: `gfx942`

**Check**:
1. Does `on_gfx950()` return True?
2. Does `on_gfx942()` return True?
3. Are there conditional code paths for gfx950?

**Run this**:
```python
import torch
GPU_ARCH = torch.cuda.get_device_properties("cuda").gcnArchName
print(f"GPU Architecture: {GPU_ARCH}")

from vllm.platforms.rocm import on_gfx950, on_gfx942
print(f"on_gfx950(): {on_gfx950()}")
print(f"on_gfx942(): {on_gfx942()}")
```

---

## 🔍 Checkpoint 7: Check Cache Hash/Lookup

**Question**: Is the prefix cache lookup working correctly?

The prefix cache uses hashes of token sequences to find matching prefixes.

**Hypothesis**: Maybe the hash is not matching on MI355X?

**Check**:
- `/app/vllm/vllm/v1/core/kv_cache_utils.py` - `BlockHashList` class
- How are cache blocks indexed and looked up?
- Is there floating-point comparison involved? (could fail on gfx950)

---

## 📋 Current Status

Based on your test output:
```
--- VLLM WITH PREFIX CACHING ---
  Deterministic: ✗ First run differs (PREFIX CACHE BUG!)
```

- ✅ Bug reproduced
- ✅ Happens with all attention backends
- ❓ Cache hit/miss behavior?
- ❓ Block allocation correct?
- ❓ KV values correct?
- ❓ gfx950-specific issue?

---

## 🎯 Next Steps

**Let's start with Checkpoint 1 and 2**:

1. Run `python debug_step1_cache_hit.py`
2. Run `python debug_step2_same_input.py`
3. Share the outputs

**Then based on results**, we'll:
- Add instrumentation to see cache block behavior
- Check for gfx950-specific code paths
- Compare KV cache values between runs

---

**Ready to start?** Run the first two debug scripts and share the output!
