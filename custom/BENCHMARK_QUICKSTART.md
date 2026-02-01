# Benchmark Quickstart: Deterministic Prefix Caching

Quick guide to benchmark the performance impact of deterministic prefix caching.

## TL;DR

```bash
# 1. Start server in normal mode
vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000

# 2. Run benchmark (in another terminal)
cd /app/vllm
pytest -s tests/entrypoints/openai/test_prefix_cache_benchmark.py

# 3. Stop server (Ctrl+C), restart in deterministic mode
VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
    vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000

# 4. Run benchmark again
VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
    pytest -s tests/entrypoints/openai/test_prefix_cache_benchmark.py

# 5. Compare results
python scripts/compare_benchmark_results.py \
    benchmark_normal_*.json \
    benchmark_deterministic_*.json
```

---

## Step-by-Step Guide

### Step 1: Benchmark Normal Mode

**Terminal 1** - Start server:
```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --enable-prefix-caching \
    --port 8000
```

Wait for: `Application startup complete`

**Terminal 2** - Run benchmark:
```bash
cd /app/vllm
pytest -s tests/entrypoints/openai/test_prefix_cache_benchmark.py
```

**Expected output**:
```
PREFIX CACHE PERFORMANCE BENCHMARK - Normal Mode
================================================================================
Model: Qwen/Qwen2.5-0.5B-Instruct
Server: http://localhost:8000/v1
Deterministic: False
Configurations: 4

  Running short_16_tokens...
    Requests: 20
    Progress: 1/20 requests (last: 98.5ms)
    Progress: 5/20 requests (last: 56.2ms)
    ...

short_16_tokens:
  Prompt tokens: 16
  R1 latency: 102.3 ms
  R2+ latency (avg): 58.1 ms
  Throughput: 17.2 req/s
  R1 overhead: +76.1%

Results saved to: benchmark_normal_20260201_123456.json
```

**Save the JSON filename** - you'll need it for comparison!

---

### Step 2: Benchmark Deterministic Mode

**Terminal 1** - Stop server (Ctrl+C), restart with deterministic mode:
```bash
VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
    vllm serve Qwen/Qwen2.5-0.5B-Instruct \
        --enable-prefix-caching \
        --port 8000
```

**Terminal 2** - Run benchmark again:
```bash
VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
    pytest -s tests/entrypoints/openai/test_prefix_cache_benchmark.py
```

**Expected output**:
```
PREFIX CACHE PERFORMANCE BENCHMARK - Deterministic Mode
================================================================================
Model: Qwen/Qwen2.5-0.5B-Instruct
Server: http://localhost:8000/v1
Deterministic: True
Configurations: 4

  Running short_16_tokens...
    Requests: 20
    Progress: 1/20 requests (last: 163.7ms)  ← Slower R1!
    Progress: 5/20 requests (last: 58.3ms)   ← Similar R2+
    ...

short_16_tokens:
  Prompt tokens: 16
  R1 latency: 165.2 ms       ← +60% vs normal
  R2+ latency (avg): 58.4 ms ← Same as normal
  Throughput: 16.1 req/s
  R1 overhead: +182.9%

Results saved to: benchmark_deterministic_20260201_124567.json
```

---

### Step 3: Compare Results

```bash
python scripts/compare_benchmark_results.py \
    benchmark_normal_20260201_123456.json \
    benchmark_deterministic_20260201_124567.json
```

**Expected output**:
```
PERFORMANCE COMPARISON: Normal vs Deterministic Prefix Caching
================================================================================

────────────────────────────────────────────────────────────────────────────────
Configuration: short_16_tokens
  Prompt tokens: 16
  Total requests: 20
────────────────────────────────────────────────────────────────────────────────

First Request (R1) - Cache Miss:
  Normal:           102.3 ms
  Deterministic:    165.2 ms
  Overhead:         +61.5 %

Subsequent Requests (R2+) - Cache Hit:
  Normal:            58.1 ms (avg)
  Deterministic:     58.4 ms (avg)
  Overhead:          +0.5 %

Amortized (All 20 requests):
  Normal:          1205.2 ms (total)
  Deterministic:   1271.9 ms (total)
  Overhead:         +5.5 %

Throughput:
  Normal:          16.59 req/s
  Deterministic:   15.72 req/s
  Change:          -5.2 %

================================================================================
SUMMARY
================================================================================

Average R1 Overhead:        +62.3 %
Average Amortized Overhead:  +6.8 %

Key Findings:
  • R1 (first request) pays the cost of two-pass computation
  • R2+ (subsequent requests) show minimal/no overhead
  • Overhead decreases significantly when amortized over multiple requests
  • Trade-off: ~62% slower R1 for perfect determinism

Report saved to: benchmark_comparison.txt
```

---

## Quick Microbenchmark

For a faster test without JSON comparison:

```bash
# Start server
vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000

# Run microbenchmark (another terminal)
pytest -s tests/entrypoints/openai/test_prefix_cache_microbenchmark.py::test_two_pass_overhead

# Restart with deterministic mode
VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
    vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000

# Run again
VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
    pytest -s tests/entrypoints/openai/test_prefix_cache_microbenchmark.py::test_two_pass_overhead
```

This gives you a quick overhead percentage without needing comparison scripts.

---

## Interpreting Results

### Key Metrics

| Metric | What it means | Good value |
|--------|---------------|------------|
| **R1 Overhead** | How much slower first request is | ~60% expected |
| **R2+ Overhead** | How much slower subsequent requests are | <5% (should be minimal) |
| **Amortized Overhead** | Real-world impact over many requests | <10% for 10+ requests |

### Decision Guide

**Enable deterministic mode if**:
- ✅ Running 10+ requests per prompt
- ✅ Reproducibility > Speed
- ✅ Research/validation work
- ✅ Can tolerate 60% slower first request

**Don't enable if**:
- ❌ First-request latency critical
- ❌ Single-request workloads
- ❌ Maximum throughput needed

### Expected Overheads

| # Requests | Amortized Overhead |
|------------|-------------------|
| 1 | +62% |
| 5 | +19% |
| 10 | +9% |
| 50 | +2% |
| 100 | +1% |

---

## Troubleshooting

### "Server not available"

**Problem**: Test can't connect to server

**Solution**:
```bash
# Check if server is running
curl http://localhost:8000/v1/models

# If not, start it:
vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000
```

### Different results each run

**Problem**: Results vary significantly

**Causes**:
- Server not warmed up (run a few requests first)
- System under load
- GPU memory contention

**Solution**:
```bash
# Warm up server before benchmarking
for i in {1..5}; do
    curl -X POST http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"test"}],"max_tokens":1}'
done

# Then run benchmark
pytest -s tests/entrypoints/openai/test_prefix_cache_benchmark.py
```

### JSON files have same mode

**Problem**: Comparison shows both files are same mode

**Solution**: Make sure to set `VLLM_DETERMINISTIC_PREFIX_CACHE=1` for second run:
```bash
# Second benchmark - set the env var!
VLLM_DETERMINISTIC_PREFIX_CACHE=1 pytest -s tests/...
```

---

## What You Get

After running all benchmarks, you'll have:

1. **JSON files**: Detailed results for each mode
   - `benchmark_normal_TIMESTAMP.json`
   - `benchmark_deterministic_TIMESTAMP.json`

2. **Comparison report**: `benchmark_comparison.txt`
   - Side-by-side metrics
   - Overhead calculations
   - Decision guidance

3. **Terminal output**: Real-time progress and results

---

## Next Steps

Once you have results:

1. **Review overhead**: Is amortized overhead acceptable for your workload?
2. **Check accuracy**: Does deterministic mode fix your reproducibility issues?
3. **Make decision**: Enable or disable based on your priorities
4. **Document**: Save results for future reference

See `BENCHMARK_GUIDE.md` for more details and advanced usage.
