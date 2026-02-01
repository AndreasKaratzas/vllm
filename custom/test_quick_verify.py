#!/usr/bin/env python3
"""Quick test to verify deterministic prefix caching."""

import subprocess
import sys
import time

# Start vLLM server
print("Starting vLLM server...")
server_proc = subprocess.Popen([
    "vllm", "serve", "Qwen/Qwen3-0.6B",
    "--dtype", "bfloat16",
    "--max-model-len", "1024",
    "--enforce-eager",
    "--port", "8001",
], env={**subprocess.os.environ, "VLLM_DEBUG_PREFIX_CACHE": "1", "VLLM_DETERMINISTIC_CACHE": "1"})

time.sleep(15)  # Wait for server to start

try:
    # Send 2 identical requests
    import openai
    client = openai.OpenAI(base_url="http://localhost:8001/v1", api_key="dummy")
    
    prompt = "The European Union consists of"
    
    print("\n" + "="*80)
    print("REQUEST 1 (cache miss)")
    print("="*80)
    r1 = client.completions.create(
        model="Qwen/Qwen3-0.6B",
        prompt=prompt,
        max_tokens=10,
        temperature=0,
        logprobs=1,
    )
    
    print("\n" + "="*80)
    print("REQUEST 2 (cache hit)")
    print("="*80)
    r2 = client.completions.create(
        model="Qwen/Qwen3-0.6B",
        prompt=prompt,
        max_tokens=10,
        temperature=0,
        logprobs=1,
    )
    
    # Compare logprobs
    print("\n" + "="*80)
    print("LOGPROB COMPARISON")
    print("="*80)
    
    lp1 = r1.choices[0].logprobs.token_logprobs
    lp2 = r2.choices[0].logprobs.token_logprobs
    
    max_diff = 0
    for i, (l1, l2) in enumerate(zip(lp1, lp2)):
        diff = abs(l1 - l2) if (l1 is not None and l2 is not None) else 0
        max_diff = max(max_diff, diff)
        status = "✓" if diff < 0.001 else "✗"
        print(f"{status} Token {i}: diff={diff:.6f} (R1={l1:.6f}, R2={l2:.6f})")
    
    print("\n" + "="*80)
    if max_diff < 0.001:
        print("✅ SUCCESS: Max difference =", max_diff)
        exit_code = 0
    else:
        print(f"❌ FAILED: Max difference = {max_diff}")
        exit_code = 1
    print("="*80)
    
finally:
    server_proc.terminate()
    server_proc.wait()

sys.exit(exit_code)
