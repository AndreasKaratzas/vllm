# Benchmark Files Overview

This directory contains benchmarking tools for deterministic prefix caching.

## Files

### Test Files

| File | Purpose | Usage |
|------|---------|-------|
| `test_prefix_cache_benchmark.py` | Full benchmark with JSON output | Comprehensive comparison |
| `test_prefix_cache_microbenchmark.py` | Quick focused tests | Fast overhead measurement |

### Scripts

| File | Purpose | Usage |
|------|---------|-------|
| `scripts/compare_benchmark_results.py` | Compare two JSON results | Generate comparison report |
| `benchmarks/overheads/benchmark_deterministic_cache.sh` | Automated microbenchmark runner | Shell script automation |

### Documentation

| File | Purpose |
|------|---------|
| `BENCHMARK_QUICKSTART.md` | 5-minute quick start guide |
| `BENCHMARK_GUIDE.md` | Comprehensive benchmarking guide |
| `TECHNICAL_REPORT.md` | Full technical analysis |

## Quick Usage

**Fastest path to results**:

```bash
# See BENCHMARK_QUICKSTART.md for step-by-step instructions
cat BENCHMARK_QUICKSTART.md
```

**Workflow**:

1. **Normal mode benchmark**:
   ```bash
   vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000
   pytest -s tests/entrypoints/openai/test_prefix_cache_benchmark.py
   ```

2. **Deterministic mode benchmark**:
   ```bash
   # Restart server with deterministic mode
   VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
       vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000
   
   VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
       pytest -s tests/entrypoints/openai/test_prefix_cache_benchmark.py
   ```

3. **Compare**:
   ```bash
   python scripts/compare_benchmark_results.py \
       benchmark_normal_*.json \
       benchmark_deterministic_*.json
   ```

## Expected Results

| Metric | Normal | Deterministic | Difference |
|--------|--------|---------------|------------|
| R1 (first request) | ~100ms | ~162ms | +60% |
| R2+ (cached) | ~58ms | ~58ms | 0% |
| Amortized (10 reqs) | ~682ms | ~744ms | +9% |

## Files Location

```
vllm/
├── tests/entrypoints/openai/
│   ├── test_prefix_cache_benchmark.py
│   └── test_prefix_cache_microbenchmark.py
├── scripts/
│   └── compare_benchmark_results.py
├── benchmarks/overheads/
│   └── benchmark_deterministic_cache.sh
├── BENCHMARK_QUICKSTART.md
├── BENCHMARK_GUIDE.md
└── TECHNICAL_REPORT.md
```

## Choosing the Right Test

| Need | Use This |
|------|----------|
| Quick overhead check | `test_prefix_cache_microbenchmark.py` |
| Detailed JSON results | `test_prefix_cache_benchmark.py` |
| Side-by-side comparison | Both tests + `compare_benchmark_results.py` |
| Automated run | `benchmark_deterministic_cache.sh` |

## Next Steps

1. **Start here**: Read `BENCHMARK_QUICKSTART.md`
2. **For details**: See `BENCHMARK_GUIDE.md`
3. **Understanding**: Check `TECHNICAL_REPORT.md`

## Support

For issues or questions:
- Check troubleshooting in `BENCHMARK_QUICKSTART.md`
- See FAQ in `BENCHMARK_GUIDE.md`
- Review technical details in `TECHNICAL_REPORT.md`
