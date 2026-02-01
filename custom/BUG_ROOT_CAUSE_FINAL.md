# Prefix Cache Bug: TRUE ROOT CAUSE

**Date**: January 31, 2026  
**Status**: ❌ **BUG STILL PRESENT** (Despite IN_PRECISION="ieee")

---

## Critical Finding

The bug persists even with `IN_PRECISION="ieee"` applied to ALL `tl.dot` operations because:

### Request Flow Differences

**R01 (Cache MISS)**:
```
Query shape: [31, 16, 128]  ← ALL 31 tokens processed
Context: 0 tokens (no cache)
Computation: Fresh 31x31 causal attention
```

**R02-R05 (Cache HIT)**:
```
Query shape: [15, 16, 128]  ← Only 15 NEW tokens processed  
Context: 16 tokens (FROM CACHE)
Computation: 15x31 attention (15 queries attend to 16 cached + 15 fresh K/V)
```

### The Problem

Even with `IN_PRECISION="ieee"`:
1. **R01**: Computes all 31 tokens together in one kernel call
2. **R02**: Reads 16 tokens from cache + computes 15 new tokens
3. The kernel **tiles and accumulates** in different order!

**Result**: Different accumulation order → Different numerical results

---

## Test Evidence

From `/app/vllm/test_with_request_tracking.log`:

```
[R01_CACHE_LOOKUP] num_tokens=31 cached_tokens=0 cache_hit=NO
[UNIFIED_ATTN_FIX] IN_PRECISION='ieee' set for gfx950+bf16 (q_shape=torch.Size([31, 16, 128]))

[R02_CACHE_LOOKUP] num_tokens=31 cached_tokens=16 cache_hit=YES
[UNIFIED_ATTN_FIX] IN_PRECISION='ieee' set for gfx950+bf16 (q_shape=torch.Size([15, 16, 128]))
```

**Logprob Differences** (vLLM+PC R1-R2 column):
```
Pos 0: +0.014736  (R1=-0.1028, R2=-0.1175)
Pos 5: -0.056620  (R1=-0.6332, R2=-0.5766) ← HUGE!
Pos 6: +0.008953  (R1=-0.0700, R2=-0.0790)
```

---

## Why IN_PRECISION="ieee" Isn't Enough

`input_precision="ieee"` **ONLY** affects `tl.dot` operations. It does NOT affect:

| Operation | Affected by IN_PRECISION? | Non-deterministic on gfx950? |
|-----------|---------------------------|------------------------------|
| `tl.dot(Q, K)` | ✅ YES | ✅ FIXED |
| `tl.dot(P, V)` | ✅ YES | ✅ FIXED |
| `tl.max(S, axis=1)` | ❌ NO | ⚠️ MAYBE |
| `tl.sum(P, axis=1)` | ❌ NO | ⚠️ MAYBE |
| `tl.exp(S - m_j)` | ❌ NO | ⚠️ MAYBE |
| `L = L * alpha + l_j` | ❌ NO | ⚠️ MAYBE |
| `acc = acc * alpha` | ❌ NO | ⚠️ MAYBE |

### The Real Issue

When the kernel processes cached vs fresh K/V:
1. **Tile iteration order** differs (context_len=0 vs context_len=16)
2. **Online softmax updates** accumulate in different order
3. **Block table indexing** uses different physical blocks

Even if each individual operation is deterministic, the **ORDER** of operations changes!

---

## Example of Accumulation Order Difference

### R01 (31 tokens, no cache):
```
Tile 0: seq_offset=[0-15]   → Compute Q[0-15] @ K[0-15]
Tile 1: seq_offset=[16-31]  → Compute Q[16-31] @ K[16-31]

Online softmax accumulation order:
  acc_0 = attention(Q[0-15])
  acc_1 = merge(acc_0, attention(Q[16-31]))
```

### R02 (15 new + 16 cached):
```
context_len = 16 (cached)

Tile 0: seq_offset=[0-15]   → READ from cache (K[0-15], V[0-15])
Tile 1: seq_offset=[16-31]  → Fresh K[16-31], V[16-31]

Online softmax accumulation order:
  acc_0 = attention_with_cached_kv(Q[0-14], K_cached[0-15])
  acc_1 = merge(acc_0, attention(Q[0-14], K_fresh[16-31]))
```

**Different tile boundaries → Different accumulation order → Different results!**

---

## Why This Is a Real Bug

You are absolutely correct that this is NOT acceptable:

1. **0.056620 difference** is HUGE in logprob space
2. This can lead to **different token selection** in edge cases  
3. The test only checks token identity, not logprob identity
4. **Prefix caching should be transparent** - same input should give same output

---

## Attempted Fixes (All Failed)

| Fix | Status | Result |
|-----|--------|--------|
| `IN_PRECISION="ieee"` on P*V dot | ❌ NOT SUFFICIENT | Still differs |
| `IN_PRECISION="ieee"` on Q*K dot | ❌ NOT SUFFICIENT | Still differs |
| `HSA_HIGH_PRECISION_MODE=1` | ❌ NOT SUFFICIENT | Still differs |
| `torch.cuda.synchronize()` (already in code) | ❌ NOT SUFFICIENT | Still differs |

---

## Possible Solutions

### Option 1: Force Identical Tile Processing (HARD)
Ensure tile iteration order is identical regardless of cache hit/miss. This would require:
- Always process full sequence length in kernel
- Mark cached tiles with a flag but process them anyway
- **Problem**: Defeats purpose of prefix caching (no speedup)

### Option 2: FP32 Accumulation (MODERATE COST)
Use FP32 for all accumulator operations, not just for IN_PRECISION:
- Change `acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)`
- Keep M, L in FP32
- **Problem**: Already using FP32 for accumulators!

### Option 3: Recompute Cached Portion (DEFEATS PURPOSE)
Don't trust cached K/V for numerical computation, recompute:
- Use cache only for memory efficiency
- Always recompute attention even if K/V is cached
- **Problem**: Defeats the performance benefit

### Option 4: Accept Small Variations (Current State)
Document that prefix caching may have small numerical variations:
- Token selection is deterministic (test passes)
- Logprobs may vary slightly (~0.05)
- **Problem**: You correctly identified this as a bug!

---

## The Real Root Cause

**It's not just floating-point precision - it's the fundamental algorithmic difference between**:
1. Computing attention for N tokens at once
2. Computing attention for M new tokens with (N-M) cached tokens

The online softmax algorithm processes tiles sequentially, and the tile boundaries differ between these two scenarios, leading to different accumulation patterns.

---

## Next Steps

We need to either:
1. **Redesign the kernel** to ensure identical tile processing order
2. **Force full recomputation** when using cached K/V (defeats purpose)
3. **Switch to a different attention algorithm** that doesn't use online softmax
4. **Use FP64** (extremely slow, impractical)
5. **Wait for AMD hardware fix** (`HSA_HIGH_PRECISION_MODE` in future ROCm)

---

## Summary for GitHub Issue

**Status**: The bug is MORE COMPLEX than initially thought.

- `IN_PRECISION="ieee"` fixes individual operations ✅
- But tile processing order differs between cache hit/miss ❌  
- This causes different accumulation patterns ❌
- Logprob differences up to 0.056620 persist ❌

**The fix you implemented helps, but doesn't fully solve the problem.**

---

**Generated**: 2026-01-31  
**Issue**: https://github.com/vllm-project/vllm/issues/33123  
**Status**: ❌ **UNRESOLVED** - Requires kernel redesign or algorithmic change
