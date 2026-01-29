# MI355X BFloat16 Prefix Caching Bug - Debugging Guide

## What We Know

### ✅ NOT the Problem
1. **PyTorch operations are deterministic** (test_torch_bf16_consistency.py passed)
   - Matmul and attention ops give identical results across runs
2. **KV cache roundtrip is exact** (test_cache_roundtrip.py passed)
   - Writing and reading from cache preserves values perfectly
3. **The attention kernel precision fixes didn't help**
   - Modified triton_decode_attention.py, triton_prefill_attention.py, triton_unified_attention.py
   - Bug persists with both TRITON and AITER backends

### ❌ The Actual Bug
- **First request (cache miss)**: logprob = -0.750660, output = "27 member states"
- **Second+ requests (cache hit)**: logprob = -0.804742, output = "(EU) consists of **2"
- **Divergence magnitude**: 0.054082 in logprob space (significant!)
- **Only affects bfloat16**, float16 works fine
- **Backend-agnostic**: Happens with both Triton and AITER
- **Hardware-specific**: MI355X (gfx950) only

## Root Cause Hypothesis

Since:
- Cache storage works correctly ✓
- PyTorch ops are deterministic ✓
- But vLLM output differs ✗

The bug MUST be in one of these areas:

### Hypothesis 1: Model Layer Computation (MOST LIKELY)
**Problem**: The model's QKV projection or normalization layers produce slightly different values due to:
- Non-deterministic RMSNorm/LayerNorm on gfx950 with BF16
- Matrix multiplication in linear layers using different precision
- Residual connections accumulating errors

**How to test**:
```bash
# Run with detailed logging
python test_debug_logits.py

# Add instrumentation
python patch_add_checksums.py
python test_bfloat16_fix.py  # Look for checksum mismatches
```

**Expected finding**: Checksums will differ in the model forward pass BEFORE attention

### Hypothesis 2: Prefix Cache Block Selection
**Problem**: vLLM incorrectly decides which KV blocks to reuse, causing:
- Wrong blocks being read from cache
- Cache hits when there should be cache miss
- Mixing blocks from different requests

**How to test**:
```python
# Add logging to kv_cache_manager.py
import logging
logger = logging.getLogger(__name__)

# In get_computed_blocks():
logger.warning(f"Prefix cache: found {len(computed_blocks)} blocks to reuse")
logger.warning(f"Block hashes: {[b.block_hash for b in computed_blocks[:3]]}")
```

**Expected finding**: Block hashes or counts differ between runs

### Hypothesis 3: Attention Metadata Mismatch
**Problem**: Different metadata passed to attention for cache-hit vs cache-miss:
- Different num_prefill_tokens / num_decode_tokens
- Wrong slot_mapping
- Incorrect block_table

**How to test**:
```python
# In flash_attn.py forward():
print(f"Attention metadata:")
print(f"  num_prefill_tokens: {attn_metadata.num_prefill_tokens}")
print(f"  num_decode_tokens: {attn_metadata.num_decode_tokens}")
print(f"  slot_mapping: {attn_metadata.slot_mapping}")
```

**Expected finding**: Metadata differs between first and second request

## Recommended Debugging Steps

### Step 1: Confirm the Bug Location
```bash
# Run the logit comparison test
python test_compare_logits_direct.py
```

This will show EXACTLY where logprobs first diverge.

### Step 2: Add Checksum Logging
```bash
# Instrument vLLM to log tensor checksums
python patch_add_checksums.py

# Run test and capture output
python test_bfloat16_fix.py 2>&1 | tee debug_output.log

# Look for where checksums first differ
grep "\\[ATTN" debug_output.log
```

### Step 3: Compare Model Layers
If checksums show model outputs differ:

```python
# Add this to vllm/v1/worker/gpu_model_runner.py in execute_model()
for layer_idx, layer_output in enumerate(hidden_states):
    checksum = hashlib.md5(layer_output.cpu().numpy().tobytes()).hexdigest()[:8]
    print(f"Layer {layer_idx} output checksum: {checksum}")
```

### Step 4: Test with Disabled Prefix Caching
```bash
# Does the bug go away without prefix caching?
python test_bfloat16_fix.py --disable-prefix-caching
```

If yes → Bug is in prefix caching logic
If no → Bug is in model computation itself

### Step 5: Bisect the Model
Find which layer first produces different outputs:

```python
# Save hidden states at each layer
hidden_states_run1 = []
hidden_states_run2 = []

# Compare layer by layer
for i in range(num_layers):
    if not torch.allclose(hidden_states_run1[i], hidden_states_run2[i]):
        print(f"Divergence starts at layer {i}")
        break
```

## Advanced Debugging with GDB

If you want to use GDB to trace execution:

```bash
# Find the vLLM server process
ps aux | grep vllm

# Attach GDB
gdb -p <PID>

# Set breakpoint in attention
break flash_attn_varlen_func

# Continue and capture variables
c
# When breakpoint hits:
print query.mean()
print key_cache.mean()
```

## Quick Tests You Can Run Now

### Test 1: Disable Prefix Caching Entirely
```python
# Modify test_bfloat16_fix.py to add:
"--disable-prefix-caching"
```

### Test 2: Force Cache Miss Every Time
```python
# In kv_cache_manager.py, modify get_computed_blocks():
def get_computed_blocks(self, request):
    # Force cache miss
    return self.empty_kv_cache_blocks, 0
```

### Test 3: Compare RMSNorm Outputs
```python
# RMSNorm is often the culprit for BF16 issues
# Add logging in vllm/model_executor/layers/layernorm.py
```

## Files to Focus On

Based on the symptoms, these are the most likely locations:

1. **vllm/v1/core/kv_cache_manager.py** (lines 164-204)
   - `get_computed_blocks()` - decides what to cache

2. **vllm/model_executor/layers/layernorm.py**
   - RMSNorm might have BF16 precision issues on gfx950

3. **vllm/v1/worker/gpu_model_runner.py**
   - `execute_model()` - runs the forward pass

4. **vllm/v1/attention/backends/flash_attn.py** (lines 607-814)
   - How attention uses cached vs fresh KV

## Expected Solution

Once you find WHERE the divergence happens, the fix will likely be:

**If model layers**:
- Force FP32 accumulation in RMSNorm/LayerNorm for BF16 on gfx950
- Use higher precision for residual connections

**If prefix caching**:
- Fix block hash computation or selection logic
- Ensure deterministic cache key generation

**If attention metadata**:
- Fix how metadata is constructed for decode vs prefill

## Contact for Help

If stuck, provide:
1. Output of `test_compare_logits_direct.py`
2. Output of `grep "\\[ATTN" debug_output.log` after running with checksums
3. Which layer first shows different checksums

This will pinpoint the exact location of the bug.
