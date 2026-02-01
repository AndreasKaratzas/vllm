#!/bin/bash
# Benchmark script for deterministic prefix caching
#
# Usage:
#   ./scripts/benchmark_deterministic_cache.sh [quick|full]
#
# Modes:
#   quick - Fast benchmark with fewer iterations
#   full  - Comprehensive benchmark (default)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
MODE="${1:-full}"
OUTPUT_DIR="$VLLM_ROOT/benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Deterministic Prefix Cache Performance Benchmark          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Mode:${NC} $MODE"
echo -e "${YELLOW}Output:${NC} $OUTPUT_DIR"
echo -e "${YELLOW}Timestamp:${NC} $TIMESTAMP"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# GPU configuration
if [ -z "$HIP_VISIBLE_DEVICES" ]; then
    echo -e "${YELLOW}⚠ HIP_VISIBLE_DEVICES not set. Using default GPUs.${NC}"
    export HIP_VISIBLE_DEVICES="4,5,6,7"
fi

echo -e "${GREEN}✓${NC} Using GPUs: $HIP_VISIBLE_DEVICES"
echo ""

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}✗ pytest not found. Please install pytest.${NC}"
    exit 1
fi

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Phase 1: Baseline (Normal Mode)                           ${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

unset VLLM_DETERMINISTIC_PREFIX_CACHE
export VLLM_DEBUG_PREFIX_CACHE=0

if [ "$MODE" = "quick" ]; then
    echo -e "${YELLOW}Running quick baseline benchmark...${NC}"
    pytest -s -v \
        tests/entrypoints/openai/test_prefix_cache_microbenchmark.py::test_two_pass_overhead \
        2>&1 | tee "$OUTPUT_DIR/baseline_${TIMESTAMP}.log"
else
    echo -e "${YELLOW}Running full baseline benchmark...${NC}"
    pytest -s -v \
        tests/entrypoints/openai/test_prefix_cache_microbenchmark.py \
        2>&1 | tee "$OUTPUT_DIR/baseline_${TIMESTAMP}.log"
fi

echo ""
echo -e "${GREEN}✓ Baseline benchmark complete${NC}"
echo ""

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Phase 2: Deterministic Mode                                ${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

export VLLM_DETERMINISTIC_PREFIX_CACHE=1
export VLLM_DEBUG_PREFIX_CACHE=0

if [ "$MODE" = "quick" ]; then
    echo -e "${YELLOW}Running quick deterministic benchmark...${NC}"
    pytest -s -v \
        tests/entrypoints/openai/test_prefix_cache_microbenchmark.py::test_two_pass_overhead \
        2>&1 | tee "$OUTPUT_DIR/deterministic_${TIMESTAMP}.log"
else
    echo -e "${YELLOW}Running full deterministic benchmark...${NC}"
    pytest -s -v \
        tests/entrypoints/openai/test_prefix_cache_microbenchmark.py \
        2>&1 | tee "$OUTPUT_DIR/deterministic_${TIMESTAMP}.log"
fi

echo ""
echo -e "${GREEN}✓ Deterministic benchmark complete${NC}"
echo ""

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Phase 3: Side-by-Side Comparison                          ${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${YELLOW}Running side-by-side comparison benchmark...${NC}"

# Note: The full benchmark test needs to start/stop servers
# For now, just document the command
echo ""
echo -e "${YELLOW}To run comprehensive comparison:${NC}"
echo ""
echo "  HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES \\"
echo "    pytest -s tests/entrypoints/openai/test_prefix_cache_benchmark.py::test_benchmark_prefix_cache_performance[both]"
echo ""
echo "This will:"
echo "  • Start server in normal mode"
echo "  • Run benchmark suite"
echo "  • Stop server"
echo "  • Start server in deterministic mode"
echo "  • Run benchmark suite"
echo "  • Stop server"
echo "  • Generate comparison report"
echo ""

# Generate summary
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Summary                                                    ${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# Extract key metrics from logs if possible
if [ -f "$OUTPUT_DIR/baseline_${TIMESTAMP}.log" ] && [ -f "$OUTPUT_DIR/deterministic_${TIMESTAMP}.log" ]; then
    echo -e "${GREEN}✓ Benchmark logs saved:${NC}"
    echo "  • Baseline:      $OUTPUT_DIR/baseline_${TIMESTAMP}.log"
    echo "  • Deterministic: $OUTPUT_DIR/deterministic_${TIMESTAMP}.log"
    echo ""
    
    # Try to extract overhead from logs
    if grep -q "Average R1 overhead" "$OUTPUT_DIR/deterministic_${TIMESTAMP}.log"; then
        OVERHEAD=$(grep "Average R1 overhead" "$OUTPUT_DIR/deterministic_${TIMESTAMP}.log" | tail -1 | awk '{print $4}')
        echo -e "${YELLOW}Average R1 Overhead:${NC} $OVERHEAD"
    fi
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Benchmark Complete!                                        ${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Next steps:"
echo "  1. Review logs in: $OUTPUT_DIR/"
echo "  2. Run full comparison if needed (see command above)"
echo "  3. Analyze results for your workload"
echo ""
