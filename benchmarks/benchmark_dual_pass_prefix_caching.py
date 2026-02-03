#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Performance benchmark for dual-pass prefix caching on ROCm.

This script measures the performance overhead of dual-pass prefix caching mode,
which provides deterministic outputs on ROCm by splitting prefill at block
boundaries. The benchmark focuses on throughput, latency, and performance
characteristics under various workloads.

For determinism testing, see: tests/v1/core/test_dual_pass_prefix_cache_determinism.py
For the bug report, see: https://github.com/vllm-project/vllm/issues/33123

Usage:
    # Basic benchmark with synthetic prompts
    python benchmark_dual_pass_prefix_caching.py \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --num-prompts 10 \\
        --repeat-count 5 \\
        --input-length-range 128:256 \\
        --output-len 50 \\
        --num-warmup-runs 2

    # Benchmark with real dataset
    python benchmark_dual_pass_prefix_caching.py \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \\
        --num-prompts 20 \\
        --repeat-count 3 \\
        --input-length-range 128:256 \\
        --output-len 50

    # Run on specific ROCm GPUs
    HIP_VISIBLE_DEVICES=0,1 python benchmark_dual_pass_prefix_caching.py \\
        --model Qwen/Qwen3-0.6B \\
        --num-prompts 5 \\
        --repeat-count 10 \\
        --input-length-range 64:128 \\
        --output-len 32 \\
        --dtype bfloat16
"""

import argparse
import dataclasses
import json
import random
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np
from transformers import PreTrainedTokenizerBase

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

try:
    from vllm.tokenizers import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer


@dataclasses.dataclass
class Request:
    """A single benchmark request."""

    prompt: str
    prompt_len: int
    output_len: int


@dataclasses.dataclass
class BenchmarkResults:
    """Results from a single benchmark run."""

    config_name: str
    total_time: float
    num_requests: int
    num_tokens_generated: int
    throughput_tokens_per_sec: float
    latency_per_request_ms: float
    per_request_latencies: list[float]  # Individual request latencies
    outputs: list[str]
    token_ids: list[list[int]]
    
    @property
    def latency_p50(self) -> float:
        """50th percentile latency."""
        return float(np.percentile(self.per_request_latencies, 50))
    
    @property
    def latency_p90(self) -> float:
        """90th percentile latency."""
        return float(np.percentile(self.per_request_latencies, 90))
    
    @property
    def latency_p95(self) -> float:
        """95th percentile latency."""
        return float(np.percentile(self.per_request_latencies, 95))
    
    @property
    def latency_p99(self) -> float:
        """99th percentile latency."""
        return float(np.percentile(self.per_request_latencies, 99))
    
    @property
    def latency_std(self) -> float:
        """Standard deviation of latencies."""
        return float(np.std(self.per_request_latencies))


def sample_tokens(tokenizer: PreTrainedTokenizerBase, length: int) -> list[int]:
    """Sample random tokens from the vocabulary, excluding special tokens."""
    vocab = tokenizer.get_vocab()
    all_special_ids = set(tokenizer.all_special_ids)
    return random.choices(
        [v for v in vocab.values() if v not in all_special_ids],
        k=length,
    )


def sample_requests_from_dataset(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    input_length_range: tuple[int, int],
    fixed_output_len: int,
) -> list[Request]:
    """Sample requests from ShareGPT-style dataset."""
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Filter conversations with at least 2 turns
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    random.shuffle(dataset)

    min_len, max_len = input_length_range
    filtered_requests: list[Request] = []

    for i in range(len(dataset)):
        if len(filtered_requests) == num_requests:
            break

        prompt_token_ids = tokenizer(dataset[i][0]).input_ids
        prompt = tokenizer.decode(prompt_token_ids)
        prompt_len = len(prompt_token_ids)

        if min_len <= prompt_len <= max_len:
            filtered_requests.append(Request(prompt, prompt_len, fixed_output_len))

    return filtered_requests


def sample_requests_from_random(
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    input_length_range: tuple[int, int],
    fixed_output_len: int,
    prefix_len: int,
) -> list[Request]:
    """Generate random requests with a common prefix."""
    requests = []
    prefix_token_ids = sample_tokens(tokenizer, prefix_len)
    min_len, max_len = input_length_range

    for _ in range(num_requests):
        unique_part_len = random.randint(min_len - prefix_len, max_len - prefix_len)
        unique_part_token_ids = sample_tokens(tokenizer, unique_part_len)
        prompt_token_ids = prefix_token_ids + unique_part_token_ids
        prompt = tokenizer.decode(prompt_token_ids)
        prompt_len = len(prompt_token_ids)
        requests.append(Request(prompt, prompt_len, fixed_output_len))

    return requests


def run_benchmark(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    config_name: str,
    is_warmup: bool = False,
) -> BenchmarkResults:
    """Run a single benchmark and collect results."""
    if not is_warmup:
        print(f"\n{'='*80}")
        print(f"Running: {config_name}")
        print(f"{'='*80}")

    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    num_requests = len(outputs)
    num_tokens_generated = sum(len(output.outputs[0].token_ids) for output in outputs)
    throughput = num_tokens_generated / total_time
    
    # Calculate per-request latencies (approximation)
    per_request_latencies = []
    for output in outputs:
        tokens_in_request = len(output.outputs[0].token_ids)
        # Approximate latency: proportional to tokens generated
        approx_latency = (tokens_in_request / num_tokens_generated) * total_time * 1000
        per_request_latencies.append(approx_latency)
    
    avg_latency = np.mean(per_request_latencies)

    output_texts = [output.outputs[0].text for output in outputs]
    output_token_ids = [output.outputs[0].token_ids for output in outputs]

    if not is_warmup:
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Requests: {num_requests}")
        print(f"  Tokens generated: {num_tokens_generated}")
        print(f"  Throughput: {throughput:.2f} tokens/sec")
        print(f"  Avg latency: {avg_latency:.2f}ms")
        print(f"  P50 latency: {np.percentile(per_request_latencies, 50):.2f}ms")
        print(f"  P90 latency: {np.percentile(per_request_latencies, 90):.2f}ms")
        print(f"  P99 latency: {np.percentile(per_request_latencies, 99):.2f}ms")

    return BenchmarkResults(
        config_name=config_name,
        total_time=total_time,
        num_requests=num_requests,
        num_tokens_generated=num_tokens_generated,
        throughput_tokens_per_sec=throughput,
        latency_per_request_ms=avg_latency,
        per_request_latencies=per_request_latencies,
        outputs=output_texts,
        token_ids=output_token_ids,
    )




def print_summary_table(
    baseline_results: list[BenchmarkResults],
    dualpass_results: list[BenchmarkResults],
):
    """Print comprehensive performance summary with detailed statistics."""
    print(f"\n{'='*120}")
    print("PERFORMANCE BENCHMARK RESULTS")
    print(f"{'='*120}")

    # Aggregate statistics across runs
    baseline_throughputs = [r.throughput_tokens_per_sec for r in baseline_results]
    baseline_latencies = [r.latency_per_request_ms for r in baseline_results]
    dualpass_throughputs = [r.throughput_tokens_per_sec for r in dualpass_results]
    dualpass_latencies = [r.latency_per_request_ms for r in dualpass_results]

    # Calculate statistics
    baseline_stats = {
        'throughput_mean': np.mean(baseline_throughputs),
        'throughput_std': np.std(baseline_throughputs),
        'latency_mean': np.mean(baseline_latencies),
        'latency_std': np.std(baseline_latencies),
        'latency_p50': np.mean([r.latency_p50 for r in baseline_results]),
        'latency_p90': np.mean([r.latency_p90 for r in baseline_results]),
        'latency_p95': np.mean([r.latency_p95 for r in baseline_results]),
        'latency_p99': np.mean([r.latency_p99 for r in baseline_results]),
    }
    
    dualpass_stats = {
        'throughput_mean': np.mean(dualpass_throughputs),
        'throughput_std': np.std(dualpass_throughputs),
        'latency_mean': np.mean(dualpass_latencies),
        'latency_std': np.std(dualpass_latencies),
        'latency_p50': np.mean([r.latency_p50 for r in dualpass_results]),
        'latency_p90': np.mean([r.latency_p90 for r in dualpass_results]),
        'latency_p95': np.mean([r.latency_p95 for r in dualpass_results]),
        'latency_p99': np.mean([r.latency_p99 for r in dualpass_results]),
    }

    # Print throughput comparison
    print(f"\n{'THROUGHPUT (tokens/sec)':<40} {'Baseline':<20} {'Dual-Pass':<20} {'Overhead':<15}")
    print("-" * 95)
    print(f"{'Mean':<40} {baseline_stats['throughput_mean']:>19.2f} {dualpass_stats['throughput_mean']:>19.2f} {((baseline_stats['throughput_mean'] - dualpass_stats['throughput_mean']) / baseline_stats['throughput_mean'] * 100):>13.2f}%")
    print(f"{'Std Dev':<40} {baseline_stats['throughput_std']:>19.2f} {dualpass_stats['throughput_std']:>19.2f}")

    # Print latency comparison
    print(f"\n{'LATENCY (ms/request)':<40} {'Baseline':<20} {'Dual-Pass':<20} {'Overhead':<15}")
    print("-" * 95)
    print(f"{'Mean':<40} {baseline_stats['latency_mean']:>19.2f} {dualpass_stats['latency_mean']:>19.2f} {((dualpass_stats['latency_mean'] - baseline_stats['latency_mean']) / baseline_stats['latency_mean'] * 100):>13.2f}%")
    print(f"{'Std Dev':<40} {baseline_stats['latency_std']:>19.2f} {dualpass_stats['latency_std']:>19.2f}")
    print(f"{'P50 (median)':<40} {baseline_stats['latency_p50']:>19.2f} {dualpass_stats['latency_p50']:>19.2f}")
    print(f"{'P90':<40} {baseline_stats['latency_p90']:>19.2f} {dualpass_stats['latency_p90']:>19.2f}")
    print(f"{'P95':<40} {baseline_stats['latency_p95']:>19.2f} {dualpass_stats['latency_p95']:>19.2f}")
    print(f"{'P99':<40} {baseline_stats['latency_p99']:>19.2f} {dualpass_stats['latency_p99']:>19.2f}")

    # Calculate overall overhead
    throughput_overhead = ((baseline_stats['throughput_mean'] - dualpass_stats['throughput_mean']) / baseline_stats['throughput_mean']) * 100
    latency_overhead = ((dualpass_stats['latency_mean'] - baseline_stats['latency_mean']) / baseline_stats['latency_mean']) * 100

    print(f"\n{'='*120}")
    print("SUMMARY")
    print(f"{'='*120}")
    print(f"\nDual-pass prefix caching overhead:")
    print(f"  Throughput: {throughput_overhead:+.2f}% ({baseline_stats['throughput_mean']:.1f} → {dualpass_stats['throughput_mean']:.1f} tok/s)")
    print(f"  Latency:    {latency_overhead:+.2f}% ({baseline_stats['latency_mean']:.1f} → {dualpass_stats['latency_mean']:.1f} ms/req)")
    
    print(f"\n{'='*80}")
    print("ABOUT DETERMINISM")
    print(f"{'='*80}")
    print(f"\nThis benchmark focuses on PERFORMANCE overhead measurement.")
    print(f"The dual-pass mode fixes prefix cache determinism WITHIN an engine instance.")
    print(f"\nNote: Output consistency across separate engine instances (different runs)")
    print(f"      may vary even with dual-pass due to ROCm's rocBLAS non-determinism.")
    print(f"      The fix ensures repeated requests within the SAME instance are deterministic.")
    print(f"\nFor comprehensive determinism testing and validation, run:")
    print(f"  pytest tests/v1/core/test_dual_pass_prefix_cache_determinism.py")
    print(f"\nRelated issue: https://github.com/vllm-project/vllm/issues/33123")
    
    print(f"\n{'='*120}\n")


def main(args):
    """Main benchmark entry point."""
    print(f"\n{'='*100}")
    print("vLLM Dual-Pass Prefix Caching Performance Benchmark")
    print(f"{'='*100}")
    print(f"Model: {args.model}")
    print(f"Num prompts: {args.num_prompts}")
    print(f"Repeat count: {args.repeat_count}")
    print(f"Warmup runs: {args.num_warmup_runs}")
    print(f"Benchmark runs per config: {args.runs_per_config}")
    print(f"Output length: {args.output_len}")
    print(f"{'='*100}\n")

    # Setup
    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    input_length_range = tuple(map(int, args.input_length_range.split(":")))
    if hasattr(args, 'seed'):
        random.seed(args.seed)
    else:
        random.seed(42)

    # Sample requests
    if args.dataset_path is not None:
        print(f"Sampling {args.num_prompts} prompts from {args.dataset_path}")
        requests = sample_requests_from_dataset(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            input_length_range=input_length_range,
            fixed_output_len=args.output_len,
        )
    else:
        print(f"Generating {args.num_prompts} random prompts with shared prefix")
        requests = sample_requests_from_random(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            input_length_range=input_length_range,
            fixed_output_len=args.output_len,
            prefix_len=args.prefix_len,
        )

    # Print request stats
    prompt_lens = [req.prompt_len for req in requests]
    print(f"\nRequest Statistics:")
    print(f"  Total requests: {len(requests)}")
    print(f"  Avg input length: {np.mean(prompt_lens):.1f} tokens")
    print(f"  P50 input length: {np.percentile(prompt_lens, 50):.1f} tokens")
    print(f"  P90 input length: {np.percentile(prompt_lens, 90):.1f} tokens")
    print(f"  Min input length: {min(prompt_lens)} tokens")
    print(f"  Max input length: {max(prompt_lens)} tokens")

    # Prepare prompts (repeat and shuffle)
    prompts = [req.prompt for req in requests] * args.repeat_count
    random.shuffle(prompts)
    print(f"\nTotal requests (after {args.repeat_count}x repeat): {len(prompts)}")

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.output_len,
        ignore_eos=True,
    )

    # Benchmark 1: Baseline (prefix caching only)
    print(f"\n{'='*100}")
    print("BENCHMARKING: Baseline (prefix caching only)")
    print(f"{'='*100}")
    
    baseline_results = []
    
    # Warmup runs
    if args.num_warmup_runs > 0:
        print(f"\nRunning {args.num_warmup_runs} warmup run(s)...")
        for warmup in range(args.num_warmup_runs):
            engine_args = EngineArgs.from_cli_args(args)
            engine_args.enable_prefix_caching = True
            engine_args.enable_dual_pass_prefix_cache = False
            llm = LLM(**dataclasses.asdict(engine_args))
            _ = run_benchmark(llm, prompts, sampling_params, f"Warmup {warmup + 1}", is_warmup=True)
            del llm
        print("✓ Warmup complete")
    
    # Actual benchmark runs
    for run in range(args.runs_per_config):
        print(f"\n--- Baseline Run {run + 1}/{args.runs_per_config} ---")
        engine_args = EngineArgs.from_cli_args(args)
        engine_args.enable_prefix_caching = True
        engine_args.enable_dual_pass_prefix_cache = False

        llm = LLM(**dataclasses.asdict(engine_args))
        result = run_benchmark(
            llm, prompts, sampling_params, f"Baseline Run {run + 1}"
        )
        baseline_results.append(result)
        del llm  # Free memory

    # Benchmark 2: Dual-pass prefix caching
    print(f"\n{'='*100}")
    print("BENCHMARKING: Dual-pass prefix caching (deterministic)")
    print(f"{'='*100}")
    
    dualpass_results = []
    
    # Warmup runs
    if args.num_warmup_runs > 0:
        print(f"\nRunning {args.num_warmup_runs} warmup run(s)...")
        for warmup in range(args.num_warmup_runs):
            engine_args = EngineArgs.from_cli_args(args)
            engine_args.enable_prefix_caching = True
            engine_args.enable_dual_pass_prefix_cache = True
            llm = LLM(**dataclasses.asdict(engine_args))
            _ = run_benchmark(llm, prompts, sampling_params, f"Warmup {warmup + 1}", is_warmup=True)
            del llm
        print("✓ Warmup complete")
    
    # Actual benchmark runs
    for run in range(args.runs_per_config):
        print(f"\n--- Dual-pass Run {run + 1}/{args.runs_per_config} ---")
        engine_args = EngineArgs.from_cli_args(args)
        engine_args.enable_prefix_caching = True
        engine_args.enable_dual_pass_prefix_cache = True

        llm = LLM(**dataclasses.asdict(engine_args))
        result = run_benchmark(
            llm, prompts, sampling_params, f"Dual-pass Run {run + 1}"
        )
        dualpass_results.append(result)
        del llm  # Free memory

    # Print summary
    print_summary_table(baseline_results, dualpass_results)


def create_argument_parser():
    """Create argument parser with all benchmark options."""
    parser = FlexibleArgumentParser(
        description="Benchmark dual-pass prefix caching performance and determinism."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to ShareGPT-style dataset (optional, uses random prompts if not provided)",
    )
    parser.add_argument(
        "--output-len", type=int, default=32, help="Number of tokens to generate"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        required=True,
        help="Number of unique prompts to sample",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=1,
        help="Number of times to repeat each prompt (tests prefix caching effectiveness)",
    )
    parser.add_argument(
        "--runs-per-config",
        type=int,
        default=3,
        help="Number of benchmark runs per configuration for statistical significance",
    )
    parser.add_argument(
        "--num-warmup-runs",
        type=int,
        default=1,
        help="Number of warmup runs before benchmarking (recommended: 1-2)",
    )
    parser.add_argument(
        "--input-length-range",
        type=str,
        required=True,
        help='Range of input lengths as "min:max" (e.g., "128:256")',
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=64,
        help="Length of common prefix for random prompts (ignored if dataset-path is provided)",
    )

    parser = EngineArgs.add_cli_args(parser)
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    main(args)
