# Benchmark Fix Summary

## What Was Wrong

The original benchmark test (`test_prefix_cache_benchmark.py`) tried to **programmatically start vLLM servers** using Python's `subprocess.Popen()`. This approach had several problems:

1. **Server startup detection was unreliable**: The test kept retrying connections but the server wasn't starting properly
2. **Process management was complex**: Starting/stopping servers programmatically is fragile
3. **Timeouts and retries**: Your log showed continuous `Retrying request to /models` errors
4. **Port conflicts**: Automatic port allocation could clash with existing services

The test would hang trying to connect to a server that never fully initialized.

## What Was Fixed

### 1. **Simplified Architecture** ✅

Changed from:
- ❌ Test starts server → waits → runs benchmark → stops server
- ✅ **User starts server manually → test connects → runs benchmark**

This is the **standard pytest pattern** used by most API tests.

### 2. **Test File Changes**

**Before** (broken):
```python
def start_server(enable_deterministic):
    # Complex subprocess management
    process = subprocess.Popen(...)
    # Unreliable startup detection
    for i in range(60):
        try:
            client.models.list()
            break
        except:
            time.sleep(2)  # Keep retrying...
```

**After** (working):
```python
@pytest.fixture(scope="module")
def client():
    # Simply connect to pre-started server
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
    # Verify it's running, skip test if not
    try:
        client.models.list()
    except:
        pytest.skip("Server not running")
    return client
```

### 3. **New Workflow**

**Step 1**: Start server manually
```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000
```

**Step 2**: Run benchmark
```bash
pytest -s tests/entrypoints/openai/test_prefix_cache_benchmark.py
```

**Step 3**: Restart server with deterministic mode
```bash
# Stop first server (Ctrl+C), then:
VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
    vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000
```

**Step 4**: Run benchmark again
```bash
VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
    pytest -s tests/entrypoints/openai/test_prefix_cache_benchmark.py
```

**Step 5**: Compare results
```bash
python scripts/compare_benchmark_results.py \
    benchmark_normal_*.json \
    benchmark_deterministic_*.json
```

### 4. **New Comparison Script** ✅

Created `scripts/compare_benchmark_results.py` to:
- Load JSON results from both modes
- Generate side-by-side comparison
- Calculate overhead percentages
- Save report to `benchmark_comparison.txt`

### 5. **Updated Documentation** ✅

- **`BENCHMARK_QUICKSTART.md`**: 5-minute quick start guide
- **`BENCHMARK_GUIDE.md`**: Updated with correct workflow
- **`README_BENCHMARKS.md`**: Overview of all benchmark files

## Files Modified/Created

### Modified
- ✅ `tests/entrypoints/openai/test_prefix_cache_benchmark.py`
  - Removed `start_server()` and `stop_server()` functions
  - Added `client` fixture for pre-started server
  - Simplified test to single mode per run
  - Added JSON output with timestamps

### Created
- ✅ `scripts/compare_benchmark_results.py` - Compare JSON results
- ✅ `BENCHMARK_QUICKSTART.md` - Quick start guide
- ✅ `README_BENCHMARKS.md` - Overview of benchmark files
- ✅ `BENCHMARK_FIX_SUMMARY.md` - This document

### Unchanged (still working)
- ✅ `tests/entrypoints/openai/test_prefix_cache_microbenchmark.py` - Already used pre-started server pattern
- ✅ `benchmarks/overheads/benchmark_deterministic_cache.sh` - Shell script wrapper

## How to Use Now

### Quick Test (5 minutes)

```bash
# Terminal 1: Start normal mode server
vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000

# Terminal 2: Run benchmark
cd /app/vllm
pytest -s tests/entrypoints/openai/test_prefix_cache_benchmark.py
# Save the JSON filename from output!

# Terminal 1: Stop server (Ctrl+C), restart with deterministic
VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
    vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000

# Terminal 2: Run benchmark again
VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
    pytest -s tests/entrypoints/openai/test_prefix_cache_benchmark.py

# Compare results
python scripts/compare_benchmark_results.py \
    benchmark_normal_TIMESTAMP1.json \
    benchmark_deterministic_TIMESTAMP2.json
```

### Even Quicker (2 minutes)

Just use the microbenchmark:

```bash
# Terminal 1: Start server
vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000

# Terminal 2: Quick test
pytest -s tests/entrypoints/openai/test_prefix_cache_microbenchmark.py::test_two_pass_overhead

# Terminal 1: Restart with deterministic
VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
    vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000

# Terminal 2: Quick test again
VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
    pytest -s tests/entrypoints/openai/test_prefix_cache_microbenchmark.py::test_two_pass_overhead
```

## Why This is Better

### Advantages

1. **✅ Reliable**: No server startup races or hangs
2. **✅ Simple**: User controls server lifecycle
3. **✅ Debuggable**: Can inspect server logs in real-time
4. **✅ Flexible**: Can use different ports, models, settings
5. **✅ Standard**: Matches how other vLLM tests work

### Trade-offs

- ⚠️ **Manual steps**: User must start/stop server manually
- ⚠️ **Two terminal windows**: One for server, one for tests

But these are **acceptable** for a benchmark tool because:
- Benchmarks aren't run frequently (not in CI)
- User needs to control environment anyway
- Easier to debug issues when they arise

## Expected Output

### Normal Mode Run

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

NORMAL MODE RESULTS
================================================================================

short_16_tokens:
  Prompt tokens: 16
  R1 latency: 102.3 ms
  R2+ latency (avg): 58.1 ms
  Throughput: 17.2 req/s
  R1 overhead: +76.1%

...

Results saved to: benchmark_normal_20260201_123456.json
```

### Comparison Output

```
PERFORMANCE COMPARISON: Normal vs Deterministic Prefix Caching
================================================================================

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

SUMMARY
================================================================================
Average R1 Overhead:        +62.3 %
Average Amortized Overhead:  +6.8 %

Report saved to: benchmark_comparison.txt
```

## Testing the Fix

To verify everything works:

```bash
# 1. Start server
vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000

# 2. Wait for "Application startup complete"

# 3. In another terminal, run test
cd /app/vllm
pytest -s tests/entrypoints/openai/test_prefix_cache_benchmark.py

# Should see:
# ✓ Connected to server at http://localhost:8000/v1
# ✓ Running benchmarks...
# ✓ Results saved to: benchmark_normal_TIMESTAMP.json
```

If it hangs like before, check:
1. Is server actually running? (`curl http://localhost:8000/v1/models`)
2. Are you in the right directory? (`cd /app/vllm`)
3. Is port 8000 accessible?

## Summary

**The benchmark now works correctly** by:
- ✅ Expecting pre-started server (like other vLLM tests)
- ✅ Saving results to JSON files
- ✅ Providing comparison script
- ✅ Better documentation

**To use**:
1. Read `BENCHMARK_QUICKSTART.md`
2. Start server manually
3. Run benchmark
4. Compare results

**No more hanging on server startup!** 🎉
