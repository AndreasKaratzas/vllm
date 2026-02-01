"""
Step 2: Test if the issue is with prefix caching specifically or generation in general
We'll send the SAME input twice but with cache DISABLED
"""

import asyncio
import httpx
from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"
GEN_ENDPOINT = "/inference/v1/generate"

async def test_without_prefix_caching():
    print("\n" + "="*80)
    print("STEP 2: TESTING WITHOUT PREFIX CACHING")
    print("="*80)

    base_args = [
        "--dtype", "float16",
        "--max-model-len", "512",
        "--enforce-eager",
        "--generation-config", "vllm",
        "--max_num_seqs", "1",
        "--no-enable-prefix-caching",  # ← Disable prefix caching
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
    print("Prefix caching: DISABLED")

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
                    "max_tokens": 10,
                    "temperature": 0.0,
                    "detokenize": False,
                    "logprobs": 1,
                },
                "stream": False,
            }

            print("\nRunning 3 identical requests...")
            outputs = []

            for i in range(3):
                print(f"\nRequest {i+1}...", end=" ")
                resp = await client.post(GEN_ENDPOINT, json=payload)
                data = resp.json()
                tokens = data["choices"][0]["token_ids"]
                decoded = tokenizer.decode(tokens, skip_special_tokens=True)
                outputs.append((tokens, decoded))
                print(f"'{decoded}'")

            print("\n" + "="*80)
            print("RESULTS WITHOUT PREFIX CACHING")
            print("="*80)

            all_same = all(tokens == outputs[0][0] for tokens, _ in outputs)

            if all_same:
                print("✓ ALL REQUESTS IDENTICAL (expected)")
                print(f"  Output: '{outputs[0][1]}'")
            else:
                print("✗ REQUESTS DIFFER (unexpected!)")
                for i, (tokens, decoded) in enumerate(outputs):
                    print(f"  Run {i+1}: '{decoded}'")

                print("\n  → This suggests a DIFFERENT issue (not prefix caching)")

if __name__ == "__main__":
    asyncio.run(test_without_prefix_caching())
