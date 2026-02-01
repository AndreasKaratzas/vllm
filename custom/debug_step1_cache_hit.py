"""
Step 1: Check if prefix caching is actually working
We'll add debug prints to see if cache hits are occurring
"""

import asyncio
import httpx
from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"
GEN_ENDPOINT = "/inference/v1/generate"

async def test_cache_usage():
    print("\n" + "="*80)
    print("STEP 1: CHECKING IF PREFIX CACHE IS WORKING")
    print("="*80)

    # Same test configuration as your main test
    base_args = [
        "--dtype", "float16",
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

    print(f"\nInput prompt length: {len(token_ids)} tokens")
    print(f"First 10 tokens: {token_ids[:10]}")

    with RemoteOpenAIServer(MODEL_NAME, base_args) as server:
        transport = httpx.AsyncHTTPTransport(uds=server.uds) if server.uds else None
        headers = {"Authorization": f"Bearer {server.DUMMY_API_KEY}"}

        async with httpx.AsyncClient(
            transport=transport,
            base_url=server.url_root,
            timeout=600,
            headers=headers,
        ) as client:

            print("\n" + "-"*80)
            print("REQUEST 1 (should be cache MISS - cold start)")
            print("-"*80)

            payload = {
                "model": MODEL_NAME,
                "token_ids": token_ids,
                "sampling_params": {
                    "max_tokens": 10,
                    "temperature": 0.0,
                    "detokenize": False,
                    "logprobs": 1,
                },
                "stream": False,
            }

            resp1 = await client.post(GEN_ENDPOINT, json=payload)
            data1 = resp1.json()
            tokens1 = data1["choices"][0]["token_ids"]
            decoded1 = tokenizer.decode(tokens1, skip_special_tokens=True)

            print(f"Response: '{decoded1}'")
            print(f"Tokens: {tokens1}")

            # Check if there's any cache info in the response
            if "usage" in data1:
                print(f"Usage: {data1['usage']}")

            print("\n" + "-"*80)
            print("REQUEST 2 (should be cache HIT - same prompt)")
            print("-"*80)

            resp2 = await client.post(GEN_ENDPOINT, json=payload)
            data2 = resp2.json()
            tokens2 = data2["choices"][0]["token_ids"]
            decoded2 = tokenizer.decode(tokens2, skip_special_tokens=True)

            print(f"Response: '{decoded2}'")
            print(f"Tokens: {tokens2}")

            if "usage" in data2:
                print(f"Usage: {data2['usage']}")

            print("\n" + "="*80)
            print("COMPARISON")
            print("="*80)

            if tokens1 == tokens2:
                print("✓ Tokens MATCH (no bug)")
            else:
                print("✗ Tokens DIFFER (bug present)")
                print(f"  Run 1: {decoded1}")
                print(f"  Run 2: {decoded2}")

                # Find first different token
                for i, (t1, t2) in enumerate(zip(tokens1, tokens2)):
                    if t1 != t2:
                        print(f"\n  First difference at position {i}:")
                        print(f"    Run 1: token_id={t1}, token={repr(tokenizer.decode([t1]))}")
                        print(f"    Run 2: token_id={t2}, token={repr(tokenizer.decode([t2]))}")
                        break

if __name__ == "__main__":
    asyncio.run(test_cache_usage())
