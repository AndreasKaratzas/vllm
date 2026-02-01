"""
Comprehensive prefix caching benchmark and validation suite.

This consolidates all prefix caching tests into one file:
- Debug/validation tests (from test_prefix_cache_debug.py)
- Microbenchmarks (from test_prefix_cache_microbenchmark.py)
- Full benchmarks (from test_prefix_cache_benchmark.py)

Uses RemoteOpenAIServer for automatic server lifecycle management.
No manual server start/stop needed - just run pytest!

Usage:
    # Run all tests
    HIP_VISIBLE_DEVICES=4,5,6,7 pytest -s test_prefix_cache_comprehensive.py
    
    # Run specific test
    pytest -s test_prefix_cache_comprehensive.py::test_determinism_validation
    
    # Run benchmarks only
    pytest -s test_prefix_cache_comprehensive.py -k benchmark
"""

import json
import os
import time
from typing import Dict, List, Any
import pytest
import httpx
from openai import OpenAI

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# ============================================================================
# FIXTURES - Server Management
# ============================================================================

@pytest.fixture(scope="module")
def server_normal():
    """Normal prefix caching server (baseline)."""
    with RemoteOpenAIServer(
        MODEL_NAME,
        [
            "--enable-prefix-caching",
            "--max-model-len", "2048",
            "--gpu-memory-utilization", "0.8",
        ],
    ) as server:
        yield server


@pytest.fixture(scope="module")
def server_deterministic():
    """Deterministic prefix caching server."""
    with RemoteOpenAIServer(
        MODEL_NAME,
        [
            "--enable-prefix-caching",
            "--max-model-len", "2048",
            "--gpu-memory-utilization", "0.8",
        ],
        env_dict={"VLLM_DETERMINISTIC_PREFIX_CACHE": "1"},
    ) as server:
        yield server


@pytest.fixture
def client_normal(server_normal):
    """OpenAI client for normal mode."""
    return server_normal.get_client()


@pytest.fixture
def client_deterministic(server_deterministic):
    """OpenAI client for deterministic mode."""
    return server_deterministic.get_client()


# ============================================================================
# TEST 1: Determinism Validation (from test_prefix_cache_debug.py)
# ============================================================================

def test_determinism_validation(server_normal, server_deterministic):
    """
    Validate that deterministic mode produces identical results across runs.
    
    This is the main correctness test from test_prefix_cache_debug.py.
    """
    print(f"\n{'='*80}")
    print("DETERMINISM VALIDATION TEST")
    print(f"{'='*80}")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How many countries are in the European Union?"},
    ]
    
    # Test normal mode (may have differences)
    print(f"\n{'-'*80}")
    print("Testing Normal Mode (R1 vs R2 may differ)")
    print(f"{'-'*80}")
    
    client_normal = server_normal.get_client()
    normal_logprobs = []
    
    for run in range(5):
        response = client_normal.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=10,
            temperature=0.0,
            logprobs=True,
            top_logprobs=5,
        )
        
        logprobs = [choice.logprobs.content[i].logprob 
                   for choice in response.choices 
                   for i in range(len(choice.logprobs.content))]
        normal_logprobs.append(logprobs)
        
        print(f"  Run {run+1}: {len(logprobs)} tokens, first logprob: {logprobs[0]:.6f}")
    
    # Check normal mode differences
    r1_r2_diff = max(abs(normal_logprobs[0][i] - normal_logprobs[1][i]) 
                     for i in range(len(normal_logprobs[0])))
    r2_r3_diff = max(abs(normal_logprobs[1][i] - normal_logprobs[2][i]) 
                     for i in range(len(normal_logprobs[1])))
    
    print(f"\n  Normal Mode:")
    print(f"    R1-R2 max diff: {r1_r2_diff:.6f} (may be non-zero)")
    print(f"    R2-R3 max diff: {r2_r3_diff:.6f} (should be zero)")
    
    # Test deterministic mode (should be identical)
    print(f"\n{'-'*80}")
    print("Testing Deterministic Mode (All runs should match)")
    print(f"{'-'*80}")
    
    client_det = server_deterministic.get_client()
    det_logprobs = []
    
    for run in range(5):
        response = client_det.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=10,
            temperature=0.0,
            logprobs=True,
            top_logprobs=5,
        )
        
        logprobs = [choice.logprobs.content[i].logprob 
                   for choice in response.choices 
                   for i in range(len(choice.logprobs.content))]
        det_logprobs.append(logprobs)
        
        print(f"  Run {run+1}: {len(logprobs)} tokens, first logprob: {logprobs[0]:.6f}")
    
    # Check deterministic mode - all should be identical
    max_diff = 0.0
    for i in range(1, 5):
        run_diff = max(abs(det_logprobs[0][j] - det_logprobs[i][j]) 
                      for j in range(len(det_logprobs[0])))
        max_diff = max(max_diff, run_diff)
    
    print(f"\n  Deterministic Mode:")
    print(f"    Max diff across all runs: {max_diff:.6f}")
    
    # Assertion
    assert max_diff == 0.0, f"Deterministic mode should have zero diff, got {max_diff}"
    
    print(f"\n{'='*80}")
    print("✅ DETERMINISM VALIDATION PASSED")
    print(f"{'='*80}\n")


# ============================================================================
# TEST 2: Two-Pass Overhead Microbenchmark
# ============================================================================

def test_two_pass_overhead(client_deterministic):
    """
    Measure overhead of two-pass computation in deterministic mode.
    
    From test_prefix_cache_microbenchmark.py::test_two_pass_overhead
    """
    print(f"\n{'='*80}")
    print("TWO-PASS OVERHEAD MICROBENCHMARK")
    print(f"{'='*80}")
    
    test_cases = [
        ("16 tokens", "Hello world! " * 8),
        ("32 tokens", "The quick brown fox. " * 8),
        ("48 tokens", "AI is transforming technology. " * 12),
    ]
    
    results = []
    
    for name, prompt in test_cases:
        print(f"\n{name}:")
        
        # First request (potential two-pass)
        start = time.perf_counter()
        r1 = client_deterministic.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        r1_latency = time.perf_counter() - start
        
        print(f"  R1 latency: {r1_latency*1000:.2f} ms ({r1.usage.prompt_tokens} tokens)")
        
        # Subsequent requests (should use cache)
        r2_latencies = []
        for _ in range(5):
            start = time.perf_counter()
            client_deterministic.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            r2_latencies.append(time.perf_counter() - start)
        
        r2_avg = sum(r2_latencies) / len(r2_latencies)
        overhead_pct = ((r1_latency - r2_avg) / r2_avg) * 100
        
        print(f"  R2-R6 latency: {r2_avg*1000:.2f} ms (avg)")
        print(f"  R1 overhead: {overhead_pct:+.1f}%")
        
        results.append({
            'name': name,
            'r1': r1_latency,
            'r2_avg': r2_avg,
            'overhead_pct': overhead_pct,
        })
    
    avg_overhead = sum(r['overhead_pct'] for r in results) / len(results)
    
    print(f"\n{'='*80}")
    print(f"Average R1 overhead: {avg_overhead:+.1f}%")
    print(f"{'='*80}\n")


# ============================================================================
# TEST 3: Full Benchmark Suite
# ============================================================================

def test_full_benchmark_normal(client_normal):
    """
    Comprehensive benchmark with multiple prompt lengths (normal mode).
    
    From test_prefix_cache_benchmark.py
    """
    print(f"\n{'='*80}")
    print("FULL BENCHMARK - Normal Mode")
    print(f"{'='*80}")
    
    configs = [
        {"name": "short_16", "prompt": "Hello world! " * 8, "requests": 10},
        {"name": "medium_32", "prompt": "The quick brown fox. " * 8, "requests": 10},
        {"name": "long_64", "prompt": "AI is transforming technology. " * 16, "requests": 8},
    ]
    
    results = _run_benchmark_suite(client_normal, configs, "Normal")
    
    # Save results
    output = {
        'mode': 'normal',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': results,
    }
    
    filename = f"benchmark_normal_{int(time.time())}.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to: {filename}")


def test_full_benchmark_deterministic(client_deterministic):
    """
    Comprehensive benchmark with multiple prompt lengths (deterministic mode).
    """
    print(f"\n{'='*80}")
    print("FULL BENCHMARK - Deterministic Mode")
    print(f"{'='*80}")
    
    configs = [
        {"name": "short_16", "prompt": "Hello world! " * 8, "requests": 10},
        {"name": "medium_32", "prompt": "The quick brown fox. " * 8, "requests": 10},
        {"name": "long_64", "prompt": "AI is transforming technology. " * 16, "requests": 8},
    ]
    
    results = _run_benchmark_suite(client_deterministic, configs, "Deterministic")
    
    # Save results
    output = {
        'mode': 'deterministic',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': results,
    }
    
    filename = f"benchmark_deterministic_{int(time.time())}.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to: {filename}")


def _run_benchmark_suite(client: OpenAI, configs: List[Dict], mode: str) -> List[Dict]:
    """Helper to run benchmark suite."""
    results = []
    
    for config in configs:
        print(f"\n  Running {config['name']}...")
        
        latencies = []
        prompt_tokens = 0
        
        for i in range(config['requests']):
            start = time.perf_counter()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": config['prompt']}],
                max_tokens=10,
                temperature=0.0,
            )
            latencies.append(time.perf_counter() - start)
            prompt_tokens = response.usage.prompt_tokens
            
            if (i + 1) % 5 == 0 or i == 0:
                print(f"    Progress: {i+1}/{config['requests']} "
                      f"(last: {latencies[-1]*1000:.1f}ms)")
        
        r1 = latencies[0]
        r2_plus = latencies[1:]
        avg_r2_plus = sum(r2_plus) / len(r2_plus) if r2_plus else 0
        
        result = {
            'config_name': config['name'],
            'num_requests': config['requests'],
            'prompt_tokens': prompt_tokens,
            'r1_latency_ms': r1 * 1000,
            'r2_plus_avg_ms': avg_r2_plus * 1000,
            'total_time_ms': sum(latencies) * 1000,
            'throughput': config['requests'] / sum(latencies),
        }
        
        results.append(result)
        
        print(f"    R1: {result['r1_latency_ms']:.1f}ms, "
              f"R2+ avg: {result['r2_plus_avg_ms']:.1f}ms, "
              f"Throughput: {result['throughput']:.2f} req/s")
    
    return results


# ============================================================================
# TEST 4: Cache Effectiveness
# ============================================================================

def test_cache_effectiveness(client_normal):
    """Test cache effectiveness with shared prefixes."""
    print(f"\n{'='*80}")
    print("CACHE EFFECTIVENESS TEST")
    print(f"{'='*80}")
    
    prefix = "You are a helpful AI assistant. Please answer: "
    suffixes = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
        "What is Go?",
    ]
    
    latencies = []
    
    print("\nTesting shared prefix...")
    for i, suffix in enumerate(suffixes):
        start = time.perf_counter()
        client_normal.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prefix + suffix}],
            max_tokens=10,
            temperature=0.0,
        )
        latency = time.perf_counter() - start
        latencies.append(latency)
        
        print(f"  Request {i+1}: {latency*1000:.2f} ms")
    
    r1 = latencies[0]
    r2_plus_avg = sum(latencies[1:]) / len(latencies[1:])
    speedup = r1 / r2_plus_avg
    
    print(f"\nR1 (no cache): {r1*1000:.2f} ms")
    print(f"R2+ (cached): {r2_plus_avg*1000:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    
    assert speedup > 1.2, f"Expected >1.2x speedup from cache, got {speedup:.2f}x"


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    print("\nRun with pytest:")
    print("  pytest -s test_prefix_cache_comprehensive.py")
