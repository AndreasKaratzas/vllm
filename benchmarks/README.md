# Benchmarks

This directory used to contain vLLM's benchmark scripts and utilities for performance testing and evaluation.

## Contents

- **Serving benchmarks**: Scripts for testing online inference performance (latency, throughput)
- **Throughput benchmarks**: Scripts for testing offline batch inference performance
- **Specialized benchmarks**: Tools for testing specific features like structured output, prefix caching, long document QA, request prioritization, and multi-modal inference
  - `benchmark_dual_pass_prefix_caching.py`: Benchmark comparing standard prefix caching vs dual-pass mode for deterministic outputs
- **Dataset utilities**: Framework for loading and sampling from various benchmark datasets (ShareGPT, HuggingFace datasets, synthetic data, etc.)

## Usage

For detailed usage instructions, examples, and dataset information, see the [Benchmark CLI documentation](https://docs.vllm.ai/en/latest/contributing/benchmarks.html#benchmark-cli).

For full CLI reference see:

- <https://docs.vllm.ai/en/latest/cli/bench/latency.html>
- <https://docs.vllm.ai/en/latest/cli/bench/serve.html>
- <https://docs.vllm.ai/en/latest/cli/bench/throughput.html>

### Dual-Pass Prefix Caching (ROCm)

The `benchmark_dual_pass_prefix_caching.py` script benchmarks the performance and determinism of dual-pass prefix caching. This feature addressesis realted to the non-deterministic behavior in rocBLAS BF16 GEMM operations when different batch sizes are used with prefix caching (https://github.com/vllm-project/vllm/issues/33123).

**Quick start:**
```bash
# Synthetic benchmark with 10 prompts, each repeated 5 times
python benchmark_dual_pass_prefix_caching.py \
    --model Qwen/Qwen3-0.6B \
    --num-prompts 10 \
    --repeat-count 5 \
    --input-length-range 64:128 \
    --output-len 32 \
    --dtype bfloat16
```

The benchmark will run both configurations (baseline and dual-pass) multiple times and report:
- Throughput (tokens/sec) and latency (ms/request) for each configuration
- Performance overhead of dual-pass mode
- Determinism analysis showing whether outputs are consistent across runs
- Recommendation on whether to use dual-pass mode
