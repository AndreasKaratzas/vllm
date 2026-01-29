# Quick Fix Guide: MI355X Prefix Caching Bug

## TL;DR

**Problem**: First vLLM request produces different output than subsequent requests on MI355X with prefix caching enabled.

**Root Cause**: Missing `input_precision` parameter in decode kernel causes numerical divergence on gfx950.

**Fix**: Add precision parameter to `kernel_paged_attention_2d()` to match prefill kernel behavior.

---

## Apply the Fix

### Method 1: Apply Patch (Recommended)

```bash
cd /app/vllm
patch -p1 < /app/potential_fix.patch
```

### Method 2: Manual Edit

Edit `/app/vllm/vllm/v1/attention/ops/chunked_prefill_paged_decode.py`:

#### Step 1: Add parameter to kernel function (line ~68)
```python
@triton.jit
def kernel_paged_attention_2d(
    # ... existing parameters ...
    USE_SINKS: tl.constexpr,
    USE_FP8: tl.constexpr,
    IN_PRECISION: tl.constexpr,  # ← ADD THIS LINE
    FP8_MIN: tl.constexpr = float8_info.min,
```

#### Step 2: Update QK computation (line ~193)
```python
# OLD:
qk = scale * tl.dot(Q, K)

# NEW:
qk = scale * tl.dot(Q, K, input_precision=IN_PRECISION)
```

#### Step 3: Update accumulation (line ~227)
```python
# OLD:
acc += tl.dot(p.to(V.dtype), V)

# NEW:
acc += tl.dot(p.to(V.dtype), V, input_precision=IN_PRECISION)
```

#### Step 4: Compute precision mode (line ~270, in `chunked_prefill_paged_decode()` function)
```python
def chunked_prefill_paged_decode(
    # ... parameters ...
):
    # ← ADD THIS BLOCK HERE (after function definition)
    # Match precision mode with prefix_prefill for consistency
    q_dtype_is_f32 = query.dtype is torch.float32
    IS_TURING = current_platform.get_device_capability() == (7, 5)
    IN_PRECISION = "ieee" if IS_TURING and q_dtype_is_f32 else None

    # Rest of function...
    if sm_scale is None:
        sm_scale = 1.0 / (query.shape[2] ** 0.5)
```

#### Step 5: Pass precision to kernel (line ~460)
```python
kernel_paged_attention_2d[
    (num_seqs, num_kv_heads)
](
    # ... existing arguments ...
    USE_SINKS=sinks is not None,
    USE_FP8=output_scale is not None,
    IN_PRECISION=IN_PRECISION,  # ← ADD THIS LINE
)
```

---

## Verify the Fix

### 1. Quick Test
```bash
cd /app/vllm
pytest -s -v tests/entrypoints/openai/test_prefix_cache_debug.py::test_side_by_side
```

**Look for**:
```
--- VLLM WITH PREFIX CACHING ---
  Deterministic: ✓ All 5 runs identical  # ← Should say this!
```

### 2. Full Test Suite
```bash
pytest -s -v tests/entrypoints/openai/test_prefix_cache_debug.py
```

### 3. Visual Verification
Check the test output for:
```
vLLM+PC Run 1: 'The European Union (EU) consists of **2'
vLLM+PC Run 2: 'The European Union (EU) consists of **2'  # ← Should match Run 1!
vLLM+PC Run 3: 'The European Union (EU) consists of **2'
vLLM+PC Run 4: 'The European Union (EU) consists of **2'
vLLM+PC Run 5: 'The European Union (EU) consists of **2'
```

All runs should be **identical** ✓

---

## Rollback (if needed)

```bash
cd /app/vllm
git checkout vllm/v1/attention/ops/chunked_prefill_paged_decode.py
```

---

## Performance Check

After applying fix, verify no significant regression:

```bash
# Simple performance test
python -c "
from vllm import LLM, SamplingParams

llm = LLM(
    model='Qwen/Qwen3-0.6B',
    dtype='bfloat16',
    enforce_eager=True,
    enable_prefix_caching=True,
)

prompts = ['How many countries are in the EU?'] * 10
sampling_params = SamplingParams(temperature=0, max_tokens=10)

import time
start = time.time()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.time() - start

print(f'Generated {len(outputs)} sequences in {elapsed:.2f}s')
print(f'Throughput: {len(outputs)/elapsed:.2f} req/s')
"
```

**Expected**: <5% difference from baseline

---

## Troubleshooting

### Issue: Patch fails
**Solution**: File might have changed. Apply manual edits from Method 2.

### Issue: Import errors after fix
**Solution**: Rebuild vLLM:
```bash
cd /app/vllm
pip install -e .
```

### Issue: Tests still fail
**Possible causes**:
1. Precision parameter not passed correctly
2. Triton version incompatibility
3. Different issue on your specific hardware

**Debug**:
```bash
# Check if precision is being used
python -c "
import inspect
from vllm.v1.attention.ops.chunked_prefill_paged_decode import kernel_paged_attention_2d
sig = inspect.signature(kernel_paged_attention_2d.fn)
print('Parameters:', list(sig.parameters.keys()))
print('IN_PRECISION present:', 'IN_PRECISION' in sig.parameters)
"
```

Should print: `IN_PRECISION present: True`

---

## Need Help?

1. **Check logs**: `/app/BUG_ANALYSIS_SUMMARY.md` for detailed analysis
2. **Run diagnostics**: `python /app/debug_attention_divergence.py`
3. **Compare outputs**: Run test with `-s` flag to see full output

---

## Success Criteria

After applying fix, you should see:

✅ All 5 vLLM+PC runs produce identical output
✅ vLLM+PC matches vLLM-PC (without prefix caching)
✅ Token sequences are deterministic
✅ Logprobs are consistent across runs
✅ No performance regression >5%

---

**Last Updated**: 2026-01-28
**Applies To**: vLLM 0.14.0rc2+ on AMD MI355X (gfx950)
