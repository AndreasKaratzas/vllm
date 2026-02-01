#!/usr/bin/env python3
"""
Debug test with logging enabled to see which code paths are taken
"""

import os
os.environ['VLLM_LOGGING_LEVEL'] = 'DEBUG'

import asyncio
import httpx
from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"
GEN_ENDPOINT = "/inference/v1/generate"

async def test_with_logging():
    print("\n" + "="*80)
    print("DEBUGGING WITH LOGGING ENABLED")
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

    print(f"\nInput: {len(token_ids)} tokens")
    print("Watching for:")
    print("  - 'max_query_len' values")
    print("  - 'context_attention_fwd' calls")
    print("  - 'kernel_paged_attention_2d' calls")
    print("  - 'IN_PRECISION' values")
    print()

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
                    "max_tokens": 2,  # Only 2 tokens for shorter logs
                    "temperature": 0.0,
                    "detokenize": False,
                },
                "stream": False,
            }

            print("="*80)
            print("REQUEST 1 (cache miss)")
            print("="*80)
            resp1 = await client.post(GEN_ENDPOINT, json=payload)
            data1 = resp1.json()
            tokens1 = data1["choices"][0]["token_ids"]
            decoded1 = tokenizer.decode(tokens1, skip_special_tokens=True)
            print(f"\nResult: '{decoded1}'")
            print(f"Tokens: {tokens1}")

            print("\n" + "="*80)
            print("REQUEST 2 (cache hit expected)")
            print("="*80)
            resp2 = await client.post(GEN_ENDPOINT, json=payload)
            data2 = resp2.json()
            tokens2 = data2["choices"][0]["token_ids"]
            decoded2 = tokenizer.decode(tokens2, skip_special_tokens=True)
            print(f"\nResult: '{decoded2}'")
            print(f"Tokens: {tokens2}")

            print("\n" + "="*80)
            print("COMPARISON")
            print("="*80)
            if tokens1 == tokens2:
                print("✓ MATCH")
            else:
                print("✗ DIFFER")
                print(f"  Run 1: {decoded1}")
                print(f"  Run 2: {decoded2}")

if __name__ == "__main__":
    asyncio.run(test_with_logging())
