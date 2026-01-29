# vLLM V1 Dataflow on ROCm with Prefix Caching

## Executive Summary

**Default Backend on ROCm**: `TRITON_ATTN` (line 338-339 in `/app/vllm/vllm/platforms/rocm.py`)

**The Bug**:
- Run 1 (cache miss): Generates token 17167 at position 3
- Runs 2+ (cache hit): Generate token 320 at position 3
- **Both use the SAME cached KV from run 1**, but produce different outputs!

## Complete Request Flow

### 1. Request Entry Point
```
User HTTP Request
  ↓
/app/vllm/vllm/entrypoints/openai/api_server.py
  → POST /inference/v1/generate
  ↓
/app/vllm/vllm/v1/engine/async_llm.py
  → AsyncLLM.generate()
```

### 2. Engine Core (Scheduling & Execution)
```
/app/vllm/vllm/v1/engine/core.py
  → EngineCore.step()
    ├─> Prefix Cache Manager decides what to reuse
    │   /app/vllm/vllm/v1/core/kv_cache_manager.py
    │     → get_computed_blocks()  ⚠️ CRITICAL: Decides which KV blocks to reuse
    │
    ├─> Build execution plan
    │   → create_scheduler_output()
    │
    └─> Execute model forward pass
        /app/vllm/vllm/v1/worker/gpu_model_runner.py
          → execute_model()
```

### 3. Model Forward Pass
```
/app/vllm/vllm/v1/worker/gpu_model_runner.py::execute_model()
  ↓
Model layers (e.g., Qwen2ForCausalLM)
  ↓
For each transformer layer:
  ├─> RMSNorm (input normalization) ⚠️ Possible BF16 issue location
  │   /app/vllm/model_executor/layers/layernorm.py
  │
  ├─> QKV Projection (Linear layer) ⚠️ Possible BF16 issue location
  │   /app/vllm/model_executor/layers/linear.py
  │
  ├─> Attention
  │   /app/vllm/v1/attention/backends/triton_attn.py  ← **THIS IS THE ACTUAL BACKEND**
  │     → forward()
  │       ├─> For PREFILL (cache miss - Run 1):
  │       │   Uses: triton_prefill_attention.py::context_attention_fwd()
  │       │   AND:  triton_reshape_and_cache_flash.py  (writes K/V to cache)
  │       │
  │       └─> For DECODE (cache hit - Runs 2+):
  │           Uses: triton_unified_attention.py::unified_attention()
  │           Reads cached K/V from previous run
  │
  ├─> MLP (feed-forward)
  │
  └─> RMSNorm (post-attention) ⚠️ Possible BF16 issue location
```

### 4. Attention Backend Details (TRITON_ATTN)

**File**: `/app/vllm/vllm/v1/attention/backends/triton_attn.py`

```python
class TritonAttentionBackend(AttentionBackend):
    def forward(...):
        # Check request type
        if is_prefill:
            # RUN 1: Cache miss path
            # Compute fresh attention with Q, K, V from model
            context_attention_fwd(q, k, v, ...)  # triton_prefill_attention.py

            # Write K, V to cache for future use
            triton_reshape_and_cache_flash(k, v, kv_cache, ...)

        else:  # is_decode
            # RUNS 2+: Cache hit path
            # Read K, V from cache (written in run 1)
            # Compute attention with fresh Q but cached K, V
            unified_attention(q, k_cache, v_cache, ...)  # triton_unified_attention.py
```

## The Bug Location

Based on your test output:
- **Divergence at position 3** (4th generated token)
- **Run 1 vs Runs 2+** differ

### Critical Code Paths to Investigate:

#### 1. Prefix Cache Block Selection ⭐ **MOST LIKELY**
**File**: `/app/vllm/vllm/v1/core/kv_cache_manager.py`
**Method**: `get_computed_blocks()` (lines 164-204)

**Why**: This decides WHICH cached blocks to use. If it:
- Returns wrong blocks
- Has off-by-one error
- Mixes blocks incorrectly

Then runs 2+ would read different KV than what run 1 computed with.

**Add logging here**:
```python
def get_computed_blocks(self, request: Request):
    computed_blocks, num_computed = self.coordinator.find_longest_cache_hit(...)

    # DEBUG
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"[PREFIX_CACHE] request_id={request.request_id}, "
                  f"num_blocks_reused={len(computed_blocks)}, "
                  f"block_hashes={[b.block_hash for b in computed_blocks[:3]]}")

    return ...
```

#### 2. Attention Metadata Construction
**File**: `/app/vllm/vllm/v1/attention/backends/triton_attn.py`
**Method**: `build_metadata()`

**Why**: If metadata (slot_mapping, block_table, etc.) is wrong for cache-hit path, attention will read from wrong cache locations.

**Add logging here**:
```python
def build_metadata(...):
    # ... build metadata ...

    # DEBUG
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"[ATTN_METADATA] num_prefill={num_prefill_tokens}, "
                  f"num_decode={num_decode_tokens}, "
                  f"common_prefix_len={common_prefix_len}")
```

#### 3. Model Layer Computation (RMSNorm/Linear)
**File**: `/app/vllm/model_executor/layers/layernorm.py`
**Method**: `RMSNorm.forward()`

**Why**: RMSNorm with BF16 on gfx950 might have non-deterministic behavior or precision issues.

**Add logging here**:
```python
def forward(self, x):
    out = self._forward(x)

    # DEBUG
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"[RMSNORM] input_mean={x.float().mean().item():.6f}, "
                  f"output_mean={out.float().mean().item():.6f}")

    return out
```

## Files to Add Logging To (In Priority Order)

### Priority 1: Prefix Cache Logic ⭐
1. `/app/vllm/vllm/v1/core/kv_cache_manager.py::get_computed_blocks()`
2. `/app/vllm/vllm/v1/core/kv_cache_coordinator.py::find_longest_cache_hit()`

### Priority 2: Attention Metadata
3. `/app/vllm/vllm/v1/attention/backends/triton_attn.py::build_metadata()`
4. `/app/vllm/vllm/v1/attention/backends/triton_attn.py::forward()`

### Priority 3: Model Layers
5. `/app/vllm/model_executor/layers/layernorm.py::RMSNorm.forward()`
6. `/app/vllm/model_executor/layers/linear.py::Linear.forward()`

## Key Questions to Answer

1. **Does run 1 write correct KV to cache?**
   - Log K/V statistics when writing to cache

2. **Do runs 2+ read the SAME KV that run 1 wrote?**
   - Log K/V statistics when reading from cache
   - Compare checksums

3. **Is the model computation (Q projection) identical?**
   - Log Q statistics for run 1 vs runs 2+
   - Should be identical if using same input

4. **Is prefix cache selecting correct blocks?**
   - Log which blocks are selected for reuse
   - Check if block_hashes match

## Next Step

I'll add targeted logging to these specific files (NOT flash_attn.py which was wrong).
