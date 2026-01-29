#!/usr/bin/env python3
"""
Simple debug script to diagnose AITER prefix caching non-determinism.
Runs 3 requests with same prompt, compares logprobs and tracks divergence.
"""

import httpx
import asyncio
from transformers import AutoTokenizer
import sys
import os

# Add tests directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vllm', 'tests'))

from utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"
GEN_ENDPOINT = "/inference/v1/generate"


async def run_3_requests(client, token_ids, tokenizer):
    """Run 3 identical requests and collect logprobs."""
    results = []

    print(f"\n{'='*100}")
    print(f"RUNNING 3 IDENTICAL REQUESTS")
    print(f"{'='*100}")
    print(f"Token IDs ({len(token_ids)} tokens): {token_ids}")
    print()

    for i in range(3):
        payload = {
            "model": MODEL_NAME,
            "token_ids": token_ids,
            "sampling_params": {
                "max_tokens": 10,
                "temperature": 0.0,
                "detokenize": False,
                "logprobs": 5,  # Top 5
            },
            "stream": False,
        }

        print(f"--- Request {i+1} ---")
        resp = await client.post(GEN_ENDPOINT, json=payload)
        data = resp.json()

        tokens = data["choices"][0]["token_ids"]
        logprobs_data = data["choices"][0].get("logprobs", {}).get("content", [])
        logprobs = [lp.get("logprob") for lp in logprobs_data]
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)

        results.append({
            'run': i + 1,
            'tokens': tokens,
            'logprobs': logprobs,
            'decoded': decoded,
        })

        print(f"  Tokens: {tokens}")
        print(f"  Decoded: '{decoded}'")
        print()

    return results


def print_comparison(results, tokenizer):
    """Print detailed comparison of 3 runs."""
    print(f"\n{'='*120}")
    print("LOGPROBS COMPARISON TABLE")
    print(f"{'='*120}")
    print(f"{'Pos':<4} {'Token':<12} {'Run 1':<16} {'Run 2':<16} {'Run 3':<16} {'R1-R2':<14} {'R2-R3':<14}")
    print("-" * 120)

    num_positions = max(len(r['logprobs']) for r in results)

    first_divergence = None

    for pos in range(num_positions):
        lp1 = results[0]['logprobs'][pos] if pos < len(results[0]['logprobs']) else None
        lp2 = results[1]['logprobs'][pos] if pos < len(results[1]['logprobs']) else None
        lp3 = results[2]['logprobs'][pos] if pos < len(results[2]['logprobs']) else None

        if lp1 is None or lp2 is None or lp3 is None:
            continue

        token_id1 = results[0]['tokens'][pos] if pos < len(results[0]['tokens']) else 0
        token_id2 = results[1]['tokens'][pos] if pos < len(results[1]['tokens']) else 0
        token_id3 = results[2]['tokens'][pos] if pos < len(results[2]['tokens']) else 0

        # Check if tokens match
        tokens_match = (token_id1 == token_id2 == token_id3)
        token_str = repr(tokenizer.decode([token_id1]))[:10]

        diff_12 = lp1 - lp2
        diff_23 = lp2 - lp3

        # Mark divergence
        marker = ""
        if not tokens_match:
            marker = "❌ TOKEN MISMATCH!"
            if first_divergence is None:
                first_divergence = pos
        elif abs(diff_12) > 1e-6 or abs(diff_23) > 1e-6:
            marker = "⚠️  logprob diff"
        else:
            marker = "✓"

        print(f"{pos:<4} {token_str:<12} {lp1:<16.8f} {lp2:<16.8f} {lp3:<16.8f} "
              f"{diff_12:+.6e}     {diff_23:+.6e}  {marker}")

    print("-" * 120)

    # Print token comparison
    print(f"\n{'='*120}")
    print("TOKEN COMPARISON")
    print(f"{'='*120}")
    print(f"  Run 1: {results[0]['tokens']}")
    print(f"  Run 2: {results[1]['tokens']}")
    print(f"  Run 3: {results[2]['tokens']}")
    print()

    all_match = (results[0]['tokens'] == results[1]['tokens'] == results[2]['tokens'])

    if all_match:
        print("✅ ALL RUNS PRODUCED IDENTICAL TOKENS!")
    else:
        print("❌ TOKENS DIFFER BETWEEN RUNS!")
        if first_divergence is not None:
            print(f"\n📍 First token divergence at position {first_divergence}:")
            print(f"   Run 1: token_id={results[0]['tokens'][first_divergence]} ({repr(tokenizer.decode([results[0]['tokens'][first_divergence]]))})")
            print(f"   Run 2: token_id={results[1]['tokens'][first_divergence]} ({repr(tokenizer.decode([results[1]['tokens'][first_divergence]]))})")
            print(f"   Run 3: token_id={results[2]['tokens'][first_divergence]} ({repr(tokenizer.decode([results[2]['tokens'][first_divergence]]))})")

        print(f"\nDecoded outputs:")
        print(f"  Run 1: '{results[0]['decoded']}'")
        print(f"  Run 2: '{results[1]['decoded']}'")
        print(f"  Run 3: '{results[2]['decoded']}'")

    print(f"{'='*120}\n")


async def main():
    print(f"\n{'='*100}")
    print(f"AITER PREFIX CACHE DEBUG")
    print(f"{'='*100}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How many countries are in the EU?"},
    ]

    token_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False
    )

    args = [
        "--dtype", "bfloat16",
        "--max-model-len", "1024",
        "--enforce-eager",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as server:
        transport = httpx.AsyncHTTPTransport(uds=server.uds) if server.uds else None
        headers = {"Authorization": f"Bearer {server.DUMMY_API_KEY}"}

        async with httpx.AsyncClient(
            transport=transport,
            base_url=server.url_root,
            timeout=600,
            headers=headers,
        ) as client:
            print(f"\nWaiting for server to be ready...")
            for _ in range(30):
                try:
                    resp = await client.get("/health")
                    if resp.status_code == 200:
                        print("✓ Server ready!")
                        break
                except:
                    await asyncio.sleep(1)

            results = await run_3_requests(client, token_ids, tokenizer)

    print_comparison(results, tokenizer)


if __name__ == "__main__":
    asyncio.run(main())
