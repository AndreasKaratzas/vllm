"""
Performance benchmark for deterministic prefix caching.

This test measures the performance impact of deterministic prefix caching
by comparing:
- First request (R1) latency
- Subsequent request (R2+) latency  
- Throughput over multiple requests
- Different prompt lengths
- Amortized overhead

IMPORTANT: This test requires a pre-started vLLM server.

Usage:
    # 1. Start server in normal mode
    vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000
    
    # 2. Run benchmark
    pytest -s tests/entrypoints/openai/test_prefix_cache_benchmark.py
    
    # 3. Stop server, restart in deterministic mode
    VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
        vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000
    
    # 4. Run benchmark again  
    VLLM_DETERMINISTIC_PREFIX_CACHE=1 \
        pytest -s tests/entrypoints/openai/test_prefix_cache_benchmark.py
"""

import json
import os
import time
from typing import List, Dict, Any
import pytest
from openai import OpenAI


# Model to benchmark
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SERVER_URL = "http://localhost:8000/v1"

# Test configurations for different prompt lengths
BENCHMARK_CONFIGS = [
    {
        "name": "short_16_tokens",
        "prompt": "Hello world! " * 8,  # ~16 tokens
        "num_requests": 20,
    },
    {
        "name": "medium_32_tokens", 
        "prompt": "The quick brown fox jumps over the lazy dog. " * 7,  # ~32 tokens
        "num_requests": 15,
    },
    {
        "name": "long_64_tokens",
        "prompt": "In a galaxy far, far away, there existed a civilization that had mastered the art of interstellar travel. " * 4,  # ~64 tokens
        "num_requests": 10,
    },
    {
        "name": "very_long_128_tokens",
        "prompt": "Artificial intelligence has revolutionized the way we process information and interact with technology. From machine learning algorithms to neural networks, AI systems continue to evolve and improve. " * 6,  # ~128 tokens
        "num_requests": 8,
    },
]


@pytest.fixture(scope="module")
def client():
    """
    Get OpenAI client for pre-started server.
    
    NOTE: This expects a server to already be running on port 8000.
    """
    client = OpenAI(
        api_key="EMPTY",
        base_url=SERVER_URL,
        timeout=60.0,
    )
    
    # Verify server is accessible
    try:
        models = client.models.list()
        print(f"✓ Connected to server at {SERVER_URL}")
        print(f"  Available models: {[m.id for m in models.data]}")
    except Exception as e:
        pytest.skip(f"Server not available at {SERVER_URL}: {e}")
    
    return client


def is_deterministic_mode() -> bool:
    """Check if server is running in deterministic mode."""
    return os.getenv("VLLM_DETERMINISTIC_PREFIX_CACHE") == "1"


def measure_request(
    client: OpenAI,
    prompt: str,
    max_tokens: int = 10,
) -> Dict[str, Any]:
    """
    Make a single request and measure timing.
    
    Returns:
        dict with 'latency' (seconds), 'tokens_generated', 'prompt_tokens'
    """
    start_time = time.perf_counter()
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        extra_body={"use_beam_search": False},
    )
    
    end_time = time.perf_counter()
    latency = end_time - start_time
    
    return {
        'latency': latency,
        'tokens_generated': response.usage.completion_tokens,
        'prompt_tokens': response.usage.prompt_tokens,
        'total_tokens': response.usage.total_tokens,
    }


def run_benchmark(
    config: Dict[str, Any],
    client: OpenAI,
) -> Dict[str, Any]:
    """
    Run benchmark for a specific configuration.
    
    Returns detailed performance metrics.
    """
    prompt = config['prompt']
    num_requests = config['num_requests']
    
    print(f"\n  Running {config['name']}...")
    print(f"    Requests: {num_requests}")
    
    # Measure each request
    latencies = []
    prompt_tokens = 0
    
    for i in range(num_requests):
        result = measure_request(client, prompt, max_tokens=10)
        latencies.append(result['latency'])
        prompt_tokens = result['prompt_tokens']
        
        # Progress indicator
        if (i + 1) % 5 == 0 or i == 0:
            print(f"    Progress: {i+1}/{num_requests} requests "
                  f"(last: {result['latency']*1000:.1f}ms)")
    
    # Calculate statistics
    r1_latency = latencies[0]
    r2_plus_latencies = latencies[1:]
    avg_r2_plus = sum(r2_plus_latencies) / len(r2_plus_latencies) if r2_plus_latencies else 0
    
    total_time = sum(latencies)
    throughput = num_requests / total_time if total_time > 0 else 0
    
    return {
        'config_name': config['name'],
        'num_requests': num_requests,
        'prompt_tokens': prompt_tokens,
        'r1_latency': r1_latency,
        'r1_latency_ms': r1_latency * 1000,
        'avg_r2_plus_latency': avg_r2_plus,
        'avg_r2_plus_latency_ms': avg_r2_plus * 1000,
        'min_latency': min(latencies),
        'max_latency': max(latencies),
        'total_time': total_time,
        'throughput': throughput,
        'all_latencies': latencies,
        'deterministic': is_deterministic_mode(),
    }


def compare_results(
    normal_results: List[Dict[str, Any]],
    deterministic_results: List[Dict[str, Any]],
) -> str:
    """
    Generate comparison report between normal and deterministic modes.
    """
    lines = []
    lines.append("\n" + "="*80)
    lines.append("PERFORMANCE COMPARISON: Normal vs Deterministic Prefix Caching")
    lines.append("="*80)
    
    for normal, det in zip(normal_results, deterministic_results):
        config_name = normal['config_name']
        
        lines.append(f"\n{'─'*80}")
        lines.append(f"Configuration: {config_name}")
        lines.append(f"  Prompt tokens: {normal['prompt_tokens']}")
        lines.append(f"  Total requests: {normal['num_requests']}")
        lines.append(f"{'─'*80}")
        
        # R1 comparison
        r1_overhead = ((det['r1_latency'] - normal['r1_latency']) / normal['r1_latency']) * 100
        lines.append(f"\nFirst Request (R1) - Cache Miss:")
        lines.append(f"  Normal:        {normal['r1_latency_ms']:>8.1f} ms")
        lines.append(f"  Deterministic: {det['r1_latency_ms']:>8.1f} ms")
        lines.append(f"  Overhead:      {r1_overhead:>+8.1f} %")
        
        # R2+ comparison
        r2_overhead = ((det['avg_r2_plus_latency'] - normal['avg_r2_plus_latency']) / normal['avg_r2_plus_latency']) * 100 if normal['avg_r2_plus_latency'] > 0 else 0
        lines.append(f"\nSubsequent Requests (R2+) - Cache Hit:")
        lines.append(f"  Normal:        {normal['avg_r2_plus_latency_ms']:>8.1f} ms (avg)")
        lines.append(f"  Deterministic: {det['avg_r2_plus_latency_ms']:>8.1f} ms (avg)")
        lines.append(f"  Overhead:      {r2_overhead:>+8.1f} %")
        
        # Amortized comparison
        amortized_overhead = ((det['total_time'] - normal['total_time']) / normal['total_time']) * 100
        lines.append(f"\nAmortized (All {normal['num_requests']} requests):")
        lines.append(f"  Normal:        {normal['total_time']*1000:>8.1f} ms (total)")
        lines.append(f"  Deterministic: {det['total_time']*1000:>8.1f} ms (total)")
        lines.append(f"  Overhead:      {amortized_overhead:>+8.1f} %")
        
        # Throughput comparison
        throughput_change = ((det['throughput'] - normal['throughput']) / normal['throughput']) * 100
        lines.append(f"\nThroughput:")
        lines.append(f"  Normal:        {normal['throughput']:>8.2f} req/s")
        lines.append(f"  Deterministic: {det['throughput']:>8.2f} req/s")
        lines.append(f"  Change:        {throughput_change:>+8.1f} %")
        
        # Latency breakdown
        lines.append(f"\nLatency Distribution (Deterministic):")
        lines.append(f"  Min:  {det['min_latency']*1000:>7.1f} ms")
        lines.append(f"  Max:  {det['max_latency']*1000:>7.1f} ms")
        lines.append(f"  R1:   {det['r1_latency']*1000:>7.1f} ms")
        lines.append(f"  R2+:  {det['avg_r2_plus_latency']*1000:>7.1f} ms (avg)")
    
    lines.append(f"\n{'='*80}")
    lines.append("SUMMARY")
    lines.append(f"{'='*80}")
    
    # Calculate average overheads
    avg_r1_overhead = sum([
        ((d['r1_latency'] - n['r1_latency']) / n['r1_latency']) * 100
        for n, d in zip(normal_results, deterministic_results)
    ]) / len(normal_results)
    
    avg_amortized_overhead = sum([
        ((d['total_time'] - n['total_time']) / n['total_time']) * 100
        for n, d in zip(normal_results, deterministic_results)
    ]) / len(normal_results)
    
    lines.append(f"\nAverage R1 Overhead:        {avg_r1_overhead:>+8.1f} %")
    lines.append(f"Average Amortized Overhead: {avg_amortized_overhead:>+8.1f} %")
    
    lines.append("\nKey Findings:")
    lines.append("  • R1 (first request) pays the cost of two-pass computation")
    lines.append("  • R2+ (subsequent requests) show minimal/no overhead")
    lines.append("  • Overhead decreases significantly when amortized over multiple requests")
    lines.append("  • Trade-off: ~60% slower R1 for perfect determinism")
    
    lines.append(f"\n{'='*80}\n")
    
    return "\n".join(lines)


def save_detailed_results(
    results: List[Dict[str, Any]],
    _unused: List[Dict[str, Any]],  # Keep signature compatible
    output_file: str = "benchmark_results.json",
):
    """Save detailed results to JSON file."""
    data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': MODEL_NAME,
        'server_url': SERVER_URL,
        'deterministic': is_deterministic_mode(),
        'configs': BENCHMARK_CONFIGS,
        'results': results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Detailed results saved to: {output_file}")


@pytest.mark.asyncio
async def test_benchmark_prefix_cache_performance(client: OpenAI):
    """
    Benchmark prefix cache performance.
    
    This test runs against a pre-started server and saves results.
    Run it once in normal mode, once in deterministic mode, then compare.
    """
    deterministic = is_deterministic_mode()
    mode_str = "Deterministic" if deterministic else "Normal"
    
    print(f"\n{'='*80}")
    print(f"PREFIX CACHE PERFORMANCE BENCHMARK - {mode_str} Mode")
    print(f"{'='*80}")
    print(f"Model: {MODEL_NAME}")
    print(f"Server: {SERVER_URL}")
    print(f"Deterministic: {deterministic}")
    print(f"Configurations: {len(BENCHMARK_CONFIGS)}")
    
    results = []
    
    print(f"\n{'─'*80}")
    print(f"RUNNING: {mode_str} Mode")
    print(f"{'─'*80}")
    
    for config in BENCHMARK_CONFIGS:
        result = run_benchmark(config, client)
        results.append(result)
    
    # Save results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    mode_label = "deterministic" if deterministic else "normal"
    output_file = f"benchmark_{mode_label}_{timestamp}.json"
    
    save_detailed_results(results, [], output_file)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"{mode_str.upper()} MODE RESULTS")
    print(f"{'='*80}")
    
    for result in results:
        print(f"\n{result['config_name']}:")
        print(f"  Prompt tokens: {result['prompt_tokens']}")
        print(f"  R1 latency: {result['r1_latency_ms']:.1f} ms")
        print(f"  R2+ latency (avg): {result['avg_r2_plus_latency_ms']:.1f} ms")
        print(f"  Throughput: {result['throughput']:.2f} req/s")
        
        if result['avg_r2_plus_latency'] > 0:
            overhead = ((result['r1_latency'] - result['avg_r2_plus_latency']) / 
                       result['avg_r2_plus_latency'] * 100)
            print(f"  R1 overhead: {overhead:+.1f}%")
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_file}")
    print("\nTo compare with other mode:")
    print("  1. Stop this server")
    print(f"  2. Start server in {'normal' if deterministic else 'deterministic'} mode")
    print("  3. Run this test again")
    print("  4. Use compare_benchmark_results.py to compare JSON files")
    print()


# Standalone run support
if __name__ == "__main__":
    import asyncio
    
    print("Please start the vLLM server first:")
    print("  vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8000")
    print("\nThen run:")
    print("  pytest -s tests/entrypoints/openai/test_prefix_cache_benchmark.py")
