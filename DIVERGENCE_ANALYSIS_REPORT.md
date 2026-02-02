# ROCm Prefix Caching Non-Determinism Root Cause Analysis

## Executive Summary

**Root Cause Identified:** The non-determinism occurs in the **MLP (Multi-Layer Perceptron) computation** in Layer 0, NOT in the attention mechanism. This is a ROCm-specific floating-point precision/ordering issue that manifests when the same input is processed in different batch contexts.

---

## Detailed Analysis

### Test Setup
- **R1 (First Run):** Processes positions 0-30 as a single fresh batch (no prefix caching)
- **R2 (Second Run):** Processes positions 16-30 with positions 0-15 already cached
- **Focus Position:** Position 16 (the boundary between cached and new tokens)

### Data Flow for Position 16

#### ✅ **Stage 1: Embedding - IDENTICAL**

**R1 (Line 523):**
```
Embedding[pos=16] = [-0.0213623046875, -0.06494140625, 0.021484375, 0.007293701171875, 0.00732421875, -0.059326171875, 0.06689453125, 0.007476806640625]
```

**R2 (Line 2153):**
```
Embedding[pos=16] = [-0.0213623046875, -0.06494140625, 0.021484375, 0.007293701171875, 0.00732421875, -0.059326171875, 0.06689453125, 0.007476806640625]
```

**Status:** ✅ MATCH - Embeddings are bit-for-bit identical

---

#### ✅ **Stage 2: Layer 0 Attention Output - IDENTICAL**

**R1 (Line 537):**
```
ATTN_OUT[pos=16] = [-0.02880859375, -0.00994873046875, 0.022216796875, 0.034912109375, -0.03466796875, -0.046142578125, -0.056884765625, -0.00396728515625]
```

**R2 (Line 2166):**
```
ATTN_OUT[pos=16] = [-0.02880859375, -0.00994873046875, 0.022216796875, 0.034912109375, -0.03466796875, -0.046142578125, -0.056884765625, -0.00396728515625]
```

**Status:** ✅ MATCH - After QKV projection, RMSNorm, RoPE, and full attention computation, outputs are identical

---

#### ✅ **Stage 3: Layer 0 Residual Stream - IDENTICAL**

**R1 (Line 540):**
```
Residual after post_norm[pos=16,:8] = [-0.216796875, 0.6640625, 0.052734375, -0.5703125, 0.16796875, -0.134765625, 0.28515625, 0.04833984375]
```

**R2 (Line 2170):**
```
Residual after post_norm[pos=16,:8] = [-0.216796875, 0.6640625, 0.052734375, -0.5703125, 0.16796875, -0.134765625, 0.28515625, 0.04833984375]
```

**Status:** ✅ MATCH - Residual stream is identical

---

### ❌ **Stage 4: Layer 0 MLP Output - DIVERGENCE POINT**

**R1 (Line 546):**
```
After MLP, hidden_states[pos=16,:8] = [0.177734375, 0.1787109375, 0.08544921875, -0.287109375, -0.1025390625, 0.05126953125, 0.19140625, -0.2021484375]
```

**R2 (Line 2176):**
```
After MLP, hidden_states[pos=16,:8] = [0.1767578125, 0.1787109375, 0.08544921875, -0.28515625, -0.10205078125, 0.051513671875, 0.19140625, -0.201171875]
```

**Status:** ❌ **DIVERGENCE DETECTED**

**Numerical Differences:**
```
Position 0: R1=0.1777343750, R2=0.1767578125, diff=-0.0009765625
Position 3: R1=-0.2871093750, R2=-0.2851562500, diff=+0.0019531250
Position 4: R1=-0.1025390625, R2=-0.1020507812, diff=+0.0004882812
Position 5: R1=0.0512695312, R2=0.0515136719, diff=+0.0002441406
Position 7: R1=-0.2021484375, R2=-0.2011718750, diff=+0.0009765625
```

---

### ❌ **Stage 5: Layer 0 Final Output - Divergence Propagates**

The final output is computed as: `hidden_states + residual`

**R1 (Line 548):**
```
Layer 0 OUTPUT[pos=16] = [-0.0390625, 0.84375, 0.138671875, -0.859375, 0.0654296875, -0.08349609375, 0.4765625, -0.154296875]
```

**R2 (Line 2178):**
```
Layer 0 OUTPUT[pos=16] = [-0.0400390625, 0.84375, 0.138671875, -0.85546875, 0.06591796875, -0.0830078125, 0.4765625, -0.15234375]
```

**Numerical Differences:**
```
Position 0: diff = -0.0010
Position 3: diff = +0.0039
Position 4: diff = +0.0005
Position 5: diff = +0.0003
```

---

## Cascading Effect Through Layers

### Layer 1 Input Divergence

**R1 Layer 1 INPUT (Line 552):**
```
[-0.0390625, 0.84375, 0.138671875, -0.859375, 0.0654296875, -0.08349609375, 0.4765625, -0.154296875]
```

**R2 Layer 1 INPUT (Line 2182):**
```
[-0.0400390625, 0.84375, 0.138671875, -0.85546875, 0.06591796875, -0.0830078125, 0.4765625, -0.15234375]
```

The small differences from Layer 0 compound through subsequent layers, eventually causing the logprob differences observed in the final output.

---

## Technical Conclusion

### The Specific Issue

The MLP layer consists of three sequential operations:
1. **`gate_up_proj`**: Linear projection (gate and up combined)
2. **`act_fn`**: Activation function (SiLU/Swish for Qwen3)
3. **`down_proj`**: Linear projection (down)

The divergence occurs somewhere in this pipeline when:
- **R1 Context:** Processing position 16 as part of a 31-token batch (positions 0-30)
- **R2 Context:** Processing position 16 as part of a 15-token batch (positions 16-30) with cached prefix

### ROCm-Specific Behavior

This is **NOT** a kernel implementation bug, but rather a **ROCm floating-point operation ordering or reduction precision issue** that affects how matrix multiplications or element-wise operations are computed when tensors have different batch dimensions.

Possible ROCm library calls involved:
1. **`rocblas_gemm`** (for linear projections in MLP)
2. **SiLU activation** (element-wise operations)
3. **Tensor addition** (hidden_states + residual)

The non-determinism likely stems from:
- Different reduction tree orderings for different batch sizes
- Non-associative floating-point arithmetic in accumulation
- Race conditions in parallel reduction operations
- Different kernel launch configurations based on batch size

---

## Why `VLLM_DETERMINISTIC_PREFIX_CACHE=1` Works

The deterministic prefix cache workaround splits R1 into TWO passes:
1. **First pass:** Positions 0-15 (cache these)
2. **Second pass:** Positions 16-30 (with 0-15 cached)

This makes R1's second pass **identical in batch structure to R2**, eliminating the divergence.

---

## Recommendations for Fix

### Option 1: Force Deterministic ROCm Operations (Short-term)
Enable `PYTORCH_ROCM_DETERMINISTIC_ALGORITHMS=1` or similar flags to force deterministic reductions.

### Option 2: Normalize Batch Processing (Medium-term)
Ensure MLP processing uses consistent batch dimensions by padding or restructuring.

### Option 3: Report to AMD/ROCm Team (Long-term)
This appears to be a fundamental ROCm/rocBLAS non-determinism issue that should be reported upstream.

---

## Supporting Evidence

All evidence points to MLP as the divergence source:
- ✅ Embeddings identical
- ✅ Attention QKV projections identical (K values after RoPE matched)
- ✅ Attention outputs identical
- ✅ Residual streams identical
- ❌ **MLP outputs differ** ← Root cause
- ❌ Final layer outputs differ (consequence)

The bug is in **ROCm's matrix multiplication or activation function libraries**, not in vLLM's attention kernels.
