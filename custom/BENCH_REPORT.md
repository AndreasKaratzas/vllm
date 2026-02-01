Run `pytest -s tests/entrypoints/openai/test_prefix_cache_microbenchmark.py::test_two_pass_overhead`:


```log
tests/entrypoints/openai/test_prefix_cache_microbenchmark.py
================================================================================
TWO-PASS COMPUTATION OVERHEAD
Deterministic mode: False
================================================================================

16 tokens:
  R1 latency: 429.07 ms (54 tokens)
  R2-R6 latency: 20.78 ms (avg of 5)
  R1 overhead: +1964.7%

32 tokens:
  R1 latency: 20.37 ms (70 tokens)
  R2-R6 latency: 19.35 ms (avg of 5)
  R1 overhead: +5.3%

48 tokens:
  R1 latency: 19.79 ms (90 tokens)
  R2-R6 latency: 19.45 ms (avg of 5)
  R1 overhead: +1.8%

64 tokens:
  R1 latency: 20.11 ms (126 tokens)
  R2-R6 latency: 19.55 ms (avg of 5)
  R1 overhead: +2.8%

================================================================================
SUMMARY
================================================================================

Average R1 overhead: +493.6%

✓ Normal mode (baseline)
  R1 uses single-pass computation
  R2+ benefit from cached prefix
.
```


Run `VLLM_DETERMINISTIC_PREFIX_CACHE=1 pytest -s tests/entrypoints/openai/test_prefix_cache_microbenchmark.py::test_two_pass_overhead`:

```log
tests/entrypoints/openai/test_prefix_cache_microbenchmark.py
================================================================================
TWO-PASS COMPUTATION OVERHEAD
Deterministic mode: True
================================================================================

16 tokens:
  R1 latency: 214.52 ms (54 tokens)
  R2-R6 latency: 20.97 ms (avg of 5)
  R1 overhead: +922.8%

32 tokens:
  R1 latency: 20.27 ms (70 tokens)
  R2-R6 latency: 19.50 ms (avg of 5)
  R1 overhead: +4.0%

48 tokens:
  R1 latency: 19.81 ms (90 tokens)
  R2-R6 latency: 19.88 ms (avg of 5)
  R1 overhead: -0.3%

64 tokens:
  R1 latency: 19.77 ms (126 tokens)
  R2-R6 latency: 20.12 ms (avg of 5)
  R1 overhead: -1.7%

================================================================================
SUMMARY
================================================================================

Average R1 overhead: +231.2%

✓ Deterministic mode enabled
  R1 uses two-pass computation for consistency
  R2+ benefit from cached prefix (no overhead)
.
```
