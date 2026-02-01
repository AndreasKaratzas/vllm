#!/usr/bin/env python3
"""
Test with Triton kernel forced (to use our precision fix).
The native HIP kernel may bypass our fix.
"""

import os
import sys

# Force disable custom ROCm paged attention to use Triton
os.environ['VLLM_ROCM_CUSTOM_PAGED_ATTN'] = '0'

import pytest

# Import and run the test
sys.path.insert(0, '/app/vllm')
from tests.entrypoints.openai.test_prefix_cache_debug import test_side_by_side

if __name__ == "__main__":
    print("="*80)
    print("FORCING TRITON KERNEL (DISABLING CUSTOM ROCM PAGED ATTENTION)")
    print("This ensures our IN_PRECISION fix is used")
    print("="*80)
    print()

    pytest.main([
        '/app/vllm/tests/entrypoints/openai/test_prefix_cache_debug.py::test_side_by_side',
        '-v', '-s'
    ])
