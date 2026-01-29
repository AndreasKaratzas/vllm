#!/usr/bin/env python3
"""
Run test with debug logging to see if IN_PRECISION is being set
"""

import os
os.environ['VLLM_LOGGING_LEVEL'] = 'WARNING'  # Will show our warning messages

import asyncio
import httpx
from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"
GEN_ENDPOINT = "/inference/v1/generate"

async def test():
    print("\n" + "="*80)
    print("TESTING WITH DEBUG LOGGING")
    print("Look for '🔍 GFX950 BF16 DETECTED' messages")
    print("="*80)

    base_args = [
        "--dtype", "bfloat16",
        "--max-model-len", "512",
        "--enforce-eager",
        "--generation-config", "vllm",
        "--max_num_seqs", "1",
    ]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How many countries are in the EU?"},
    ]

    token_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False
    )

    with RemoteOpenAIServer(MODEL_NAME, base_args) as server:
        transport = httpx.AsyncHTTPTransport(uds=server.uds) if server.uds else None
        headers = {"Authorization": f"Bearer {server.DUMMY_API_KEY}"}

        async with httpx.AsyncClient(
            transport=transport,
            base_url=server.url_root,
            timeout=600,
            headers=headers,
        ) as client:

            payload = {
                "model": MODEL_NAME,
                "token_ids": token_ids,
                "sampling_params": {
                    "max_tokens": 3,
                    "temperature": 0.0,
                    "detokenize": False,
                },
                "stream": False,
            }

            print("\n" + "-"*80)
            print("REQUEST 1 (cache miss)")
            print("-"*80)
            resp1 = await client.post(GEN_ENDPOINT, json=payload)
            data1 = resp1.json()
            tokens1 = data1["choices"][0]["token_ids"]
            decoded1 = tokenizer.decode(tokens1, skip_special_tokens=True)
            print(f"Output: '{decoded1}'")

            print("\n" + "-"*80)
            print("REQUEST 2 (cache hit)")
            print("-"*80)
            resp2 = await client.post(GEN_ENDPOINT, json=payload)
            data2 = resp2.json()
            tokens2 = data2["choices"][0]["token_ids"]
            decoded2 = tokenizer.decode(tokens2, skip_special_tokens=True)
            print(f"Output: '{decoded2}'")

            print("\n" + "="*80)
            if tokens1 == tokens2:
                print("✓ OUTPUTS MATCH")
            else:
                print("✗ OUTPUTS DIFFER")
                print(f"  Run 1: {decoded1}")
                print(f"  Run 2: {decoded2}")
            print("="*80)

if __name__ == "__main__":
    asyncio.run(test())
