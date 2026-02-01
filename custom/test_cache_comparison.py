#!/usr/bin/env python3
"""
Direct test to compare cache miss vs cache hit with detailed instrumentation.
"""
import requests
import time
import json

# Simple synchronous test using vLLM server API
def test_cache_comparison():
    print("="*100)
    print("CACHE MISS vs CACHE HIT COMPARISON TEST")
    print("="*100)
    
    # Server should already be running with debug flags enabled
    base_url = "http://localhost:8000"
    
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHow many countries are in the EU?<|im_end|>\n<|im_start|>assistant\n"
    
    print(f"\nPrompt: {prompt[:100]}...")
    print(f"Prompt length: {len(prompt)} characters")
    
    # Request 1: Cache MISS (first time seeing this prompt)
    print(f"\n{'='*100}")
    print("REQUEST 1: CACHE MISS (Expected)")
    print(f"{'='*100}")
    
    response1 = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": "Qwen/Qwen3-0.6B",
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 5,
            "seed": 0,
        },
        timeout=30,
    )
    
    result1 = response1.json()
    print(f"Generated: '{result1['choices'][0]['text']}'")
    if result1['choices'][0].get('logprobs'):
        logprob1 = result1['choices'][0]['logprobs']['token_logprobs'][0]
        print(f"Logprob: {logprob1:.15f}")
    
    time.sleep(1)  # Small delay
    
    # Request 2: Cache HIT (same prompt again)
    print(f"\n{'='*100}")
    print("REQUEST 2: CACHE HIT (Expected)")
    print(f"{'='*100}")
    
    response2 = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": "Qwen/Qwen3-0.6B",
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 5,
            "seed": 0,
        },
        timeout=30,
    )
    
    result2 = response2.json()
    print(f"Generated: '{result2['choices'][0]['text']}'")
    if result2['choices'][0].get('logprobs'):
        logprob2 = result2['choices'][0]['logprobs']['token_logprobs'][0]
        print(f"Logprob: {logprob2:.15f}")
    
    time.sleep(1)  # Small delay
    
    # Request 3: Cache HIT (same prompt third time)
    print(f"\n{'='*100}")
    print("REQUEST 3: CACHE HIT (Expected)")
    print(f"{'='*100}")
    
    response3 = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": "Qwen/Qwen3-0.6B",
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 5,
            "seed": 0,
        },
        timeout=30,
    )
    
    result3 = response3.json()
    print(f"Generated: '{result3['choices'][0]['text']}'")
    if result3['choices'][0].get('logprobs'):
        logprob3 = result3['choices'][0]['logprobs']['token_logprobs'][0]
        print(f"Logprob: {logprob3:.15f}")
    
    # Analysis
    print(f"\n{'='*100}")
    print("COMPARISON ANALYSIS")
    print(f"{'='*100}")
    
    token1 = result1['choices'][0]['text']
    token2 = result2['choices'][0]['text']
    token3 = result3['choices'][0]['text']
    
    print(f"\nToken comparison:")
    print(f"  Request 1 (miss): '{token1}'")
    print(f"  Request 2 (hit):  '{token2}'")
    print(f"  Request 3 (hit):  '{token3}'")
    
    if result1['choices'][0].get('logprobs') and result2['choices'][0].get('logprobs') and result3['choices'][0].get('logprobs'):
        logprob1 = result1['choices'][0]['logprobs']['token_logprobs'][0]
        logprob2 = result2['choices'][0]['logprobs']['token_logprobs'][0]
        logprob3 = result3['choices'][0]['logprobs']['token_logprobs'][0]
        
        print(f"\nLogprob comparison:")
        print(f"  Request 1 (miss): {logprob1:.15f}")
        print(f"  Request 2 (hit):  {logprob2:.15f}")
        print(f"  Request 3 (hit):  {logprob3:.15f}")
        
        diff_1_2 = abs(logprob1 - logprob2)
        diff_2_3 = abs(logprob2 - logprob3)
        diff_1_3 = abs(logprob1 - logprob3)
        
        print(f"\nNumerical differences:")
        print(f"  Request 1 vs 2: {diff_1_2:.15e}")
        print(f"  Request 2 vs 3: {diff_2_3:.15e}")
        print(f"  Request 1 vs 3: {diff_1_3:.15e}")
        
        print(f"\nDeterminism check:")
        if diff_1_2 < 1e-10:
            print(f"  ✅ Cache miss == Cache hit (Request 1 vs 2)")
        else:
            print(f"  ❌ Cache miss != Cache hit (Request 1 vs 2)")
        
        if diff_2_3 < 1e-10:
            print(f"  ✅ Cache hit == Cache hit (Request 2 vs 3)")
        else:
            print(f"  ❌ Cache hit != Cache hit (Request 2 vs 3)")
        
        if token1 == token2 == token3:
            print(f"  ✅ All tokens identical")
        else:
            print(f"  ❌ Tokens differ")
    
    print(f"\n{'='*100}")
    print("CONCLUSION")
    print(f"{'='*100}")
    
    if token1 == token2 == token3:
        print("✅ PREFIX CACHING IS DETERMINISTIC")
        print("   All requests produced identical outputs")
    else:
        print("❌ PREFIX CACHING HAS ISSUES")
        print("   Requests produced different outputs")
    
    print(f"\n{'='*100}")

if __name__ == "__main__":
    test_cache_comparison()
