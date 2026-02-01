# Benchmark Guide: Deterministic Prefix Caching

This guide explains how to benchmark the performance impact of deterministic prefix caching.

## Quick Start

### Option 1: Quick Comparison (Recommended - 5 minutes)

See `BENCHMARK_QUICKSTART.md` for the fastest path to results.

```bash
# 1. Normal mode
vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000
pytest -s tests/entrypoints/openai/test_prefix_cache_benchmark.py

# 2. Deterministic mode (restart server)
VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
    vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000
VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
    pytest -s tests/entrypoints/openai/test_prefix_cache_benchmark.py

# 3. Compare
python scripts/compare_benchmark_results.py benchmark_normal_*.json benchmark_deterministic_*.json
```

### Option 2: Microbenchmarks (Fast, less detailed)

```bash
# Start server in normal mode
vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000

# In another terminal, run microbenchmarks
pytest -s tests/entrypoints/openai/test_prefix_cache_microbenchmark.py

# Stop server, then start in deterministic mode
VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
    vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000

# Run microbenchmarks again
VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
    pytest -s tests/entrypoints/openai/test_prefix_cache_microbenchmark.py
```

---

## Benchmark Tests

### 1. Microbenchmark (`test_prefix_cache_microbenchmark.py`)

**Purpose**: Measure specific performance characteristics

**Tests**:
- `test_two_pass_overhead`: Measures R1 vs R2+ latency
- `test_cache_effectiveness`: Tests cache hit/miss performance
- `test_prompt_length_scaling`: How performance scales with prompt length

**Run**:
```bash
# All tests
pytest -s tests/entrypoints/openai/test_prefix_cache_microbenchmark.py

# Specific test
pytest -s tests/entrypoints/openai/test_prefix_cache_microbenchmark.py::test_two_pass_overhead
```

**Output**:
```
TWO-PASS COMPUTATION OVERHEAD
Deterministic mode: True
================================================================================

16 tokens:
  R1 latency: 162.45 ms (16 tokens)
  R2-R6 latency: 98.23 ms (avg of 5)
  R1 overhead: +65.3%

32 tokens:
  R1 latency: 185.67 ms (32 tokens)
  R2-R6 latency: 102.34 ms (avg of 5)
  R1 overhead: +81.4%

SUMMARY
================================================================================
Average R1 overhead: +73.4%
```

### 2. Full Benchmark (`test_prefix_cache_benchmark.py`)

**Purpose**: Comprehensive performance measurement with detailed JSON output

**Features**:
- Tests multiple prompt lengths (16, 32, 64, 128 tokens)
- Measures R1 (cache miss) and R2+ (cache hit) separately
- Calculates throughput and amortized overhead
- Saves results to JSON for comparison

**Requirements**: Server must be pre-started on port 8000

**Run**:
```bash
# 1. Start server in normal mode
vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000

# 2. Run benchmark (in another terminal)
pytest -s tests/entrypoints/openai/test_prefix_cache_benchmark.py

# 3. Restart server in deterministic mode
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

**Output** (from comparison script):
```
PERFORMANCE COMPARISON: Normal vs Deterministic Prefix Caching
================================================================================

────────────────────────────────────────────────────────────────────────────────
Configuration: medium_32_tokens
  Prompt tokens: 32
  Total requests: 15
────────────────────────────────────────────────────────────────────────────────

First Request (R1) - Cache Miss:
  Normal:           100.5 ms
  Deterministic:    162.3 ms
  Overhead:         +61.5 %

Subsequent Requests (R2+) - Cache Hit:
  Normal:            58.2 ms (avg)
  Deterministic:     58.4 ms (avg)
  Overhead:          +0.3 %

Amortized (All 15 requests):
  Normal:          914.3 ms (total)
  Deterministic:   979.9 ms (total)
  Overhead:         +7.2 %

Throughput:
  Normal:          16.41 req/s
  Deterministic:   15.31 req/s
  Change:          -6.7 %

================================================================================
SUMMARY
================================================================================

Average R1 Overhead:        +62.3 %
Average Amortized Overhead:  +8.9 %

Key Findings:
  • R1 (first request) pays the cost of two-pass computation
  • R2+ (subsequent requests) show minimal/no overhead
  • Overhead decreases significantly when amortized over multiple requests
  • Trade-off: ~62% slower R1 for perfect determinism

Report saved to: benchmark_comparison.txt
```

---

## Understanding the Metrics

### Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **R1 Latency** | First request latency (cache miss) | Shows two-pass overhead in deterministic mode |
| **R2+ Latency** | Subsequent request latency (cache hit) | Should be similar in both modes |
| **Amortized Overhead** | Total time difference / Total requests | Real-world impact over multiple requests |
| **Throughput** | Requests per second | Overall system performance |

### Expected Results

**Normal Mode (Baseline)**:
```
R1 (cache miss):  ~100ms
R2+ (cache hit):  ~58ms
Amortization:     Minimal difference between R1 and R2+
```

**Deterministic Mode**:
```
R1 (cache miss):  ~162ms (+60% overhead)
R2+ (cache hit):  ~58ms  (no overhead)
Amortization:     ~9% overhead over 10 requests
```

### Why R1 is Slower

In deterministic mode, R1 executes in two passes:

```
Normal R1:         [────────────────] 100ms (1 pass)

Deterministic R1:  [──────][──────] 162ms (2 passes)
                   Pass 1  Pass 2
                   (prefix)(suffix)
```

**Pass 1**: Process prefix, cache it (~85ms)
**Pass 2**: Process suffix with cached prefix (~77ms)
**Total**: ~162ms vs 100ms normal = +62% overhead

### Why R2+ is Unchanged

R2+ already uses the cached prefix naturally:

```
Normal R2:         [──────] 58ms (cached prefix)

Deterministic R2:  [──────] 58ms (cached prefix)
                   Same!
```

No additional overhead because it's already doing what deterministic mode forces R1 to do.

---

## Interpreting Results for Your Workload

### When Deterministic Mode is Worth It

✅ **Good fit if**:
- Running batch workloads (10+ requests)
- Reproducibility > Speed
- Research/validation work
- Can tolerate 60% slower first request

❌ **Not ideal if**:
- First-request latency is critical
- Single-request workloads
- Maximum throughput needed
- <10 requests per prompt

### Cost-Benefit Analysis

Use this table to decide:

| # Requests | Normal | Deterministic | Overhead | Worth it? |
|------------|--------|---------------|----------|-----------|
| 1 | 100ms | 162ms | +62% | ❌ Probably not |
| 5 | 332ms | 394ms | +19% | ⚠️ Depends |
| 10 | 682ms | 744ms | +9% | ✅ Likely yes |
| 50 | 3022ms | 3084ms | +2% | ✅ Definitely |

---

## Custom Benchmarks

### Test Your Own Prompts

Create a custom test:

```python
import pytest
from openai import OpenAI

@pytest.mark.asyncio
async def test_my_workload():
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )
    
    # Your actual prompts
    my_prompts = [
        "Your system prompt + question 1",
        "Your system prompt + question 2",
        # ... more prompts
    ]
    
    # Measure
    import time
    latencies = []
    
    for prompt in my_prompts:
        start = time.perf_counter()
        response = client.chat.completions.create(
            model="your-model",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
        end = time.perf_counter()
        latencies.append(end - start)
    
    # Analyze
    r1 = latencies[0]
    r2_plus = latencies[1:]
    print(f"R1: {r1*1000:.2f}ms")
    print(f"R2+ avg: {sum(r2_plus)/len(r2_plus)*1000:.2f}ms")
```

### Measure Specific Operations

Add timing to vLLM code:

```python
# In scheduler.py
import time

# Before two-pass logic
start = time.perf_counter()

# ... two-pass computation ...

end = time.perf_counter()
logger.info(f"[TIMING] Two-pass took {(end-start)*1000:.2f}ms")
```

---

## Troubleshooting

### Server Won't Start

**Error**: `Address already in use`

**Solution**:
```bash
# Find and kill existing vLLM process
ps aux | grep vllm
kill <pid>

# Or use different port
vllm serve ... --port 8001
```

### Benchmark Takes Too Long

**Solution**: Use quick mode
```bash
./scripts/benchmark_deterministic_cache.sh quick
```

Or test fewer requests:
```python
# In test file, reduce num_requests
BENCHMARK_CONFIGS = [
    {
        "name": "short_test",
        "prompt": "Hello world!",
        "num_requests": 5,  # Reduced from 20
    },
]
```

### Results Don't Match Documentation

**Possible causes**:
1. Different GPU (results are for MI355X)
2. Different model size
3. Different prompt length
4. System load

**Solution**: Compare relative differences (%) not absolute times (ms)

---

## Advanced: Profiling

### Using PyTorch Profiler

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    # Run your benchmark
    pass

prof.export_chrome_trace("trace.json")
# View in chrome://tracing
```

### Using ROCm Profiler (AMD)

```bash
rocprof --stats python -m vllm.entrypoints.openai.api_server ...
```

---

## Benchmark Checklist

Before running benchmarks:

- [ ] GPU is not under load from other processes
- [ ] Deterministic mode is set correctly (`VLLM_DETERMINISTIC_PREFIX_CACHE`)
- [ ] Server has warmed up (run a few requests first)
- [ ] Using consistent model and config
- [ ] Recording environment details (GPU, model, settings)

After running benchmarks:

- [ ] Save results with timestamp
- [ ] Document any anomalies
- [ ] Compare with expected ranges
- [ ] Calculate cost-benefit for your workload
- [ ] Make informed decision about enabling deterministic mode

---

## Example Results

### AMD MI355X (gfx950), Qwen2.5-0.5B, bfloat16

```
Prompt Length: 31 tokens
Max Tokens: 10
Num Requests: 10

Mode          R1      R2-R10   Total   Amortized
            (ms)    (ms avg)   (ms)    Overhead
────────────────────────────────────────────────
Normal      100.5     58.2    682.3      -
Deterministic 162.3     58.4    743.9    +9.0%

Conclusion: 9% overhead acceptable for determinism
```

---

## Summary

**To benchmark deterministic prefix caching**:

1. **Quick test**: `./scripts/benchmark_deterministic_cache.sh quick`
2. **Full test**: `./scripts/benchmark_deterministic_cache.sh full`
3. **Custom test**: Modify test files for your workload
4. **Analyze**: Compare R1 overhead vs amortized cost
5. **Decide**: Enable if amortized overhead acceptable

**Expected trade-offs**:
- R1: +60% slower
- R2+: No change
- Amortized (10 reqs): +9%
- Benefit: Perfect determinism (0.000000 difference)

---

For questions or issues, see:
- `TECHNICAL_REPORT.md` - Detailed analysis
- `PR_DESCRIPTION.md` - Implementation overview
- `QUICKSTART_DETERMINISTIC_CACHE.md` - Usage guide
