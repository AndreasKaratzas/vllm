#!/usr/bin/env python3
"""
Test AITER with prefix caching DISABLED to see if outputs are deterministic.
This will tell us if the issue is in AITER itself or only when caching is involved.
"""

import httpx
import asyncio
from transformers import AutoTokenizer
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vllm', 'tests'))

from utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"
GEN_ENDPOINT = "/inference/v1/generate"


async def run_3_requests(client, token_ids, tokenizer):
    """Run 3 identical requests and collect outputs."""
    results = []

    print(f"\nToken IDs ({len(token_ids)} tokens): {token_ids}\n")

    for i in range(3):
        payload = {
            "model": MODEL_NAME,
            "token_ids": token_ids,
            "sampling_params": {
                "max_tokens": 10,
                "temperature": 0.0,
                "detokenize": False,
            },
            "stream": False,
        }

        print(f"--- Request {i+1} ---")
        resp = await client.post(GEN_ENDPOINT, json=payload)
        data = resp.json()

        tokens = data["choices"][0]["token_ids"]
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)

        results.append({
            'run': i + 1,
            'tokens': tokens,
            'decoded': decoded,
        })

        print(f"  Tokens: {tokens}")
        print(f"  Decoded: '{decoded}'")

    return results


async def main():
    print(f"\n{'='*100}")
    print(f"AITER WITHOUT PREFIX CACHING - Determinism Test")
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
        "--no-enable-prefix-caching",  # DISABLE PREFIX CACHING
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
            print(f"\nWaiting for server...")
            for _ in range(30):
                try:
                    resp = await client.get("/health")
                    if resp.status_code == 200:
                        print("✓ Server ready!")
                        break
                except:
                    await asyncio.sleep(1)

            results = await run_3_requests(client, token_ids, tokenizer)

    # Check determinism
    print(f"\n{'='*100}")
    print("DETERMINISM CHECK")
    print(f"{'='*100}")

    all_match = (results[0]['tokens'] == results[1]['tokens'] == results[2]['tokens'])

    if all_match:
        print("✅ ALL 3 RUNS PRODUCED IDENTICAL TOKENS!")
        print(f"  Tokens: {results[0]['tokens']}")
        print(f"  Decoded: '{results[0]['decoded']}'")
    else:
        print("❌ TOKENS DIFFER BETWEEN RUNS!")
        for i, r in enumerate(results, 1):
            print(f"  Run {i}: {r['tokens']} -> '{r['decoded']}'")

        # Find first divergence
        for pos in range(min(len(r['tokens']) for r in results)):
            tokens_at_pos = [r['tokens'][pos] for r in results]
            if len(set(tokens_at_pos)) > 1:
                print(f"\n📍 First divergence at position {pos}:")
                for i, r in enumerate(results, 1):
                    print(f"  Run {i}: token_id={r['tokens'][pos]}")
                break

    print(f"{'='*100}\n")


if __name__ == "__main__":
    asyncio.run(main())
