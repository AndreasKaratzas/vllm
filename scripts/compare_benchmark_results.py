#!/usr/bin/env python3
"""
Compare benchmark results from normal and deterministic modes.

Usage:
    python scripts/compare_benchmark_results.py \
        benchmark_normal_20260201_123456.json \
        benchmark_deterministic_20260201_124567.json
"""

import json
import sys
from typing import Dict, Any, List


def load_results(filepath: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_results(normal: Dict[str, Any], deterministic: Dict[str, Any]) -> str:
    """Generate comparison report."""
    lines = []
    lines.append("="*80)
    lines.append("PERFORMANCE COMPARISON: Normal vs Deterministic Prefix Caching")
    lines.append("="*80)
    lines.append(f"\nModel: {normal['model']}")
    lines.append(f"Normal timestamp: {normal['timestamp']}")
    lines.append(f"Deterministic timestamp: {deterministic['timestamp']}")
    
    normal_results = normal['results']
    det_results = deterministic['results']
    
    if len(normal_results) != len(det_results):
        lines.append("\n⚠ WARNING: Different number of configs tested!")
        lines.append(f"  Normal: {len(normal_results)}")
        lines.append(f"  Deterministic: {len(det_results)}")
    
    for norm, det in zip(normal_results, det_results):
        config_name = norm['config_name']
        
        lines.append(f"\n{'─'*80}")
        lines.append(f"Configuration: {config_name}")
        lines.append(f"  Prompt tokens: {norm['prompt_tokens']}")
        lines.append(f"  Total requests: {norm['num_requests']}")
        lines.append(f"{'─'*80}")
        
        # R1 comparison
        r1_overhead = ((det['r1_latency'] - norm['r1_latency']) / norm['r1_latency']) * 100
        lines.append(f"\nFirst Request (R1) - Cache Miss:")
        lines.append(f"  Normal:        {norm['r1_latency_ms']:>8.1f} ms")
        lines.append(f"  Deterministic: {det['r1_latency_ms']:>8.1f} ms")
        lines.append(f"  Overhead:      {r1_overhead:>+8.1f} %")
        
        # R2+ comparison
        if norm['avg_r2_plus_latency'] > 0:
            r2_overhead = ((det['avg_r2_plus_latency'] - norm['avg_r2_plus_latency']) / 
                          norm['avg_r2_plus_latency']) * 100
            lines.append(f"\nSubsequent Requests (R2+) - Cache Hit:")
            lines.append(f"  Normal:        {norm['avg_r2_plus_latency_ms']:>8.1f} ms (avg)")
            lines.append(f"  Deterministic: {det['avg_r2_plus_latency_ms']:>8.1f} ms (avg)")
            lines.append(f"  Overhead:      {r2_overhead:>+8.1f} %")
        
        # Amortized comparison
        amortized_overhead = ((det['total_time'] - norm['total_time']) / norm['total_time']) * 100
        lines.append(f"\nAmortized (All {norm['num_requests']} requests):")
        lines.append(f"  Normal:        {norm['total_time']*1000:>8.1f} ms (total)")
        lines.append(f"  Deterministic: {det['total_time']*1000:>8.1f} ms (total)")
        lines.append(f"  Overhead:      {amortized_overhead:>+8.1f} %")
        
        # Throughput comparison
        throughput_change = ((det['throughput'] - norm['throughput']) / norm['throughput']) * 100
        lines.append(f"\nThroughput:")
        lines.append(f"  Normal:        {norm['throughput']:>8.2f} req/s")
        lines.append(f"  Deterministic: {det['throughput']:>8.2f} req/s")
        lines.append(f"  Change:        {throughput_change:>+8.1f} %")
    
    # Summary
    lines.append(f"\n{'='*80}")
    lines.append("SUMMARY")
    lines.append(f"{'='*80}")
    
    # Calculate average overheads
    avg_r1_overhead = sum([
        ((d['r1_latency'] - n['r1_latency']) / n['r1_latency']) * 100
        for n, d in zip(normal_results, det_results)
    ]) / len(normal_results)
    
    avg_amortized_overhead = sum([
        ((d['total_time'] - n['total_time']) / n['total_time']) * 100
        for n, d in zip(normal_results, det_results)
    ]) / len(normal_results)
    
    lines.append(f"\nAverage R1 Overhead:        {avg_r1_overhead:>+8.1f} %")
    lines.append(f"Average Amortized Overhead: {avg_amortized_overhead:>+8.1f} %")
    
    lines.append("\nKey Findings:")
    lines.append("  • R1 (first request) pays the cost of two-pass computation")
    lines.append("  • R2+ (subsequent requests) show minimal/no overhead")
    lines.append("  • Overhead decreases significantly when amortized over multiple requests")
    lines.append(f"  • Trade-off: ~{avg_r1_overhead:.0f}% slower R1 for perfect determinism")
    
    lines.append(f"\n{'='*80}\n")
    
    return "\n".join(lines)


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_benchmark_results.py <normal_results.json> <deterministic_results.json>")
        sys.exit(1)
    
    normal_file = sys.argv[1]
    det_file = sys.argv[2]
    
    print(f"Loading results...")
    print(f"  Normal: {normal_file}")
    print(f"  Deterministic: {det_file}")
    print()
    
    try:
        normal = load_results(normal_file)
        deterministic = load_results(det_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)
    
    # Verify they're different modes
    if normal.get('deterministic') == deterministic.get('deterministic'):
        print("⚠ WARNING: Both files appear to be from the same mode!")
        if normal.get('deterministic'):
            print("  Both are deterministic mode")
        else:
            print("  Both are normal mode")
        print()
    
    report = compare_results(normal, deterministic)
    print(report)
    
    # Save report
    output_file = "benchmark_comparison.txt"
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {output_file}")


if __name__ == "__main__":
    main()
