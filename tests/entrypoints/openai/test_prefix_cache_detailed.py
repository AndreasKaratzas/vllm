"""
Detailed prefix caching test with comprehensive layer-by-layer comparison.
Run with: 
  VLLM_DEBUG_PREFIX_CACHE=1 VLLM_DEBUG_CACHE_WRITES=1 VLLM_DEBUG_CACHE_READS=1 \
  pytest -v -s test_prefix_cache_detailed.py
"""

import subprocess
import pytest
import httpx
import torch
import time
from transformers import AutoTokenizer

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"


def test_prefix_cache_detailed():
    """
    Run vLLM with and without prefix caching, capturing detailed layer outputs.
    Compare layer-by-layer to identify where numerical differences occur.
    """
    print(f"\n{'='*100}")
    print("DETAILED PREFIX CACHE COMPARISON TEST")
    print(f"{'='*100}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How many countries are in the EU?"},
    ]
    
    token_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False,
    )
    prompt_text = tokenizer.decode(token_ids)
    
    print(f"\nPrompt: {prompt_text[:100]}...")
    print(f"Prompt tokens: {len(token_ids)}")
    
    # ========================================================================
    # TEST 1: WITH PREFIX CACHING - Multiple runs
    # ========================================================================
    print(f"\n{'='*100}")
    print("TEST 1: VLLM WITH PREFIX CACHING")
    print(f"{'='*100}")
    
    with RemoteOpenAIServer(
        MODEL_NAME,
        [
            "--dtype", "bfloat16",
            "--max-model-len", "1024",
            "--enforce-eager",
            "--generation-config", "vllm",
        ],
        env_dict={
            "VLLM_DEBUG_PREFIX_CACHE": "1",
            "VLLM_DEBUG_CACHE_WRITES": "1",
            "VLLM_DEBUG_CACHE_READS": "1",
        }
    ) as server:
        client = server.get_client()
        
        results_with_cache = []
        
        # Run multiple times to test cache hit path
        for run_idx in range(3):
            print(f"\n{'-'*80}")
            print(f"RUN {run_idx + 1} (WITH PREFIX CACHING)")
            print(f"{'-'*80}")
            
            response = client.post(
                "/v1/completions",
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt_text,
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "logprobs": 5,
                    "seed": 0,
                },
                timeout=30,
            )
            
            assert response.status_code == 200
            result = response.json()
            
            choice = result["choices"][0]
            text = choice["text"]
            logprobs_data = choice["logprobs"]
            
            print(f"Generated: '{text}'")
            if logprobs_data and logprobs_data.get("tokens"):
                token = logprobs_data["tokens"][0]
                logprob = logprobs_data["token_logprobs"][0]
                print(f"Token: {token}, Logprob: {logprob:.10f}")
            
            results_with_cache.append({
                "text": text,
                "logprobs": logprobs_data,
            })
            
            # Small delay between runs
            time.sleep(0.5)
    
    # ========================================================================
    # TEST 2: WITHOUT PREFIX CACHING - Multiple runs
    # ========================================================================
    print(f"\n{'='*100}")
    print("TEST 2: VLLM WITHOUT PREFIX CACHING")
    print(f"{'='*100}")
    
    with RemoteOpenAIServer(
        MODEL_NAME,
        [
            "--dtype", "bfloat16",
            "--max-model-len", "1024",
            "--enforce-eager",
            "--generation-config", "vllm",
            "--no-enable-prefix-caching",
        ],
        env_dict={
            "VLLM_DEBUG_PREFIX_CACHE": "1",
            "VLLM_DEBUG_CACHE_WRITES": "1",
            "VLLM_DEBUG_CACHE_READS": "1",
        }
    ) as server:
        client = server.get_client()
        
        results_without_cache = []
        
        for run_idx in range(3):
            print(f"\n{'-'*80}")
            print(f"RUN {run_idx + 1} (WITHOUT PREFIX CACHING)")
            print(f"{'-'*80}")
            
            response = client.post(
                "/v1/completions",
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt_text,
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "logprobs": 5,
                    "seed": 0,
                },
                timeout=30,
            )
            
            assert response.status_code == 200
            result = response.json()
            
            choice = result["choices"][0]
            text = choice["text"]
            logprobs_data = choice["logprobs"]
            
            print(f"Generated: '{text}'")
            if logprobs_data and logprobs_data.get("tokens"):
                token = logprobs_data["tokens"][0]
                logprob = logprobs_data["token_logprobs"][0]
                print(f"Token: {token}, Logprob: {logprob:.10f}")
            
            results_without_cache.append({
                "text": text,
                "logprobs": logprobs_data,
            })
            
            time.sleep(0.5)
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print(f"\n{'='*100}")
    print("ANALYSIS")
    print(f"{'='*100}")
    
    # Check determinism within each group
    print("\n1. Determinism Check:")
    
    with_cache_tokens = [r["text"] for r in results_with_cache]
    without_cache_tokens = [r["text"] for r in results_without_cache]
    
    with_cache_deterministic = len(set(with_cache_tokens)) == 1
    without_cache_deterministic = len(set(without_cache_tokens)) == 1
    
    print(f"   WITH prefix caching:    {'✓ All 3 runs identical' if with_cache_deterministic else '✗ Runs differ'}")
    print(f"   WITHOUT prefix caching: {'✓ All 3 runs identical' if without_cache_deterministic else '✗ Runs differ'}")
    
    if not with_cache_deterministic:
        print("\n   ⚠️  WITH prefix caching outputs:")
        for i, text in enumerate(with_cache_tokens):
            print(f"      Run {i+1}: '{text}'")
    
    if not without_cache_deterministic:
        print("\n   ⚠️  WITHOUT prefix caching outputs:")
        for i, text in enumerate(without_cache_tokens):
            print(f"      Run {i+1}: '{text}'")
    
    # Compare WITH vs WITHOUT
    print("\n2. Cross-comparison:")
    match = with_cache_tokens[0] == without_cache_tokens[0]
    print(f"   WITH[0] == WITHOUT[0]: {'✓ Match' if match else '✗ Differ'}")
    
    if not match:
        print(f"      WITH prefix caching:    '{with_cache_tokens[0]}'")
        print(f"      WITHOUT prefix caching: '{without_cache_tokens[0]}'")
    
    # Detailed logprob comparison
    print("\n3. Logprob Analysis:")
    
    def get_first_logprob(result):
        if result["logprobs"] and result["logprobs"].get("token_logprobs"):
            return result["logprobs"]["token_logprobs"][0]
        return None
    
    with_cache_logprobs = [get_first_logprob(r) for r in results_with_cache]
    without_cache_logprobs = [get_first_logprob(r) for r in results_without_cache]
    
    print("\n   WITH prefix caching logprobs:")
    for i, lp in enumerate(with_cache_logprobs):
        if lp is not None:
            print(f"      Run {i+1}: {lp:.10f}")
    
    print("\n   WITHOUT prefix caching logprobs:")
    for i, lp in enumerate(without_cache_logprobs):
        if lp is not None:
            print(f"      Run {i+1}: {lp:.10f}")
    
    # Check if first run with cache differs from rest (cache miss vs hit)
    if len(with_cache_logprobs) >= 2 and all(lp is not None for lp in with_cache_logprobs):
        run1_vs_run2 = abs(with_cache_logprobs[0] - with_cache_logprobs[1])
        print(f"\n   Difference between Run 1 and Run 2 (WITH cache): {run1_vs_run2:.10f}")
        
        if run1_vs_run2 > 1e-6:
            print(f"      ⚠️  NUMERICAL INSTABILITY DETECTED!")
            print(f"      This suggests cache miss (Run 1) differs from cache hit (Run 2+)")
    
    # ========================================================================
    # CONCLUSION
    # ========================================================================
    print(f"\n{'='*100}")
    print("CONCLUSION")
    print(f"{'='*100}")
    
    issues = []
    
    if not with_cache_deterministic:
        issues.append("❌ PREFIX CACHING NON-DETERMINISTIC: First run differs from subsequent runs")
    
    if not without_cache_deterministic:
        issues.append("❌ WITHOUT CACHING NON-DETERMINISTIC: Unexpected - should be deterministic")
    
    if not match:
        issues.append("❌ WITH vs WITHOUT DIFFER: Prefix caching affects output")
    
    if not issues:
        print("✅ ALL CHECKS PASSED")
    else:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    
    print(f"\n{'='*100}")
    
    # Assert for CI
    assert with_cache_deterministic, "Prefix caching must be deterministic (all runs identical)"
    assert without_cache_deterministic, "Non-cached path must be deterministic"
    assert match, "WITH and WITHOUT prefix caching must produce the same output"


if __name__ == "__main__":
    test_prefix_cache_detailed()
