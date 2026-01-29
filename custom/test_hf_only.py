#!/usr/bin/env python3
"""
Test HuggingFace directly with bfloat16 to isolate the issue
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-0.6B"

def test_hf_bfloat16():
    print("\n" + "="*80)
    print("TESTING HUGGINGFACE WITH BFLOAT16")
    print("="*80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How many countries are in the EU?"},
    ]

    token_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False
    )
    input_ids = torch.tensor([token_ids], device=model.device)

    print(f"\nInput tokens: {len(token_ids)}")
    print(f"Model dtype: {model.dtype}")
    print()

    # Test 1: WITH cache (2 runs)
    print("-" * 80)
    print("TEST 1: WITH KV CACHE")
    print("-" * 80)

    results_with_cache = []
    for i in range(2):
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                use_cache=True,  # ← WITH cache
            )
        generated_ids = outputs[0, len(token_ids):].tolist()
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results_with_cache.append((generated_ids, decoded))
        print(f"  Run {i+1}: '{decoded}'")

    # Test 2: WITHOUT cache (2 runs)
    print("\n" + "-" * 80)
    print("TEST 2: WITHOUT KV CACHE")
    print("-" * 80)

    results_without_cache = []
    for i in range(2):
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                use_cache=False,  # ← WITHOUT cache
            )
        generated_ids = outputs[0, len(token_ids):].tolist()
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results_without_cache.append((generated_ids, decoded))
        print(f"  Run {i+1}: '{decoded}'")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    with_cache_consistent = results_with_cache[0][0] == results_with_cache[1][0]
    without_cache_consistent = results_without_cache[0][0] == results_without_cache[1][0]
    cache_matches_no_cache = results_with_cache[0][0] == results_without_cache[0][0]

    print(f"With cache (run 1 == run 2):    {'✓' if with_cache_consistent else '✗'}")
    print(f"Without cache (run 1 == run 2): {'✓' if without_cache_consistent else '✗'}")
    print(f"With cache == Without cache:    {'✓' if cache_matches_no_cache else '✗'}")

    if not cache_matches_no_cache:
        print(f"\n⚠️  KV cache affects output on HuggingFace too!")
        print(f"  With cache:    '{results_with_cache[0][1]}'")
        print(f"  Without cache: '{results_without_cache[0][1]}'")
        print("\n  → This is a HuggingFace/model issue, not vLLM-specific!")
    else:
        print("\n✓ HuggingFace produces consistent results")

    # Clean up
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    test_hf_bfloat16()
