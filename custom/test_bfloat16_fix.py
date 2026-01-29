#!/usr/bin/env python3
"""
Test with bfloat16 - this is where the bug occurs on MI355X
"""

import asyncio
import httpx
from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"
GEN_ENDPOINT = "/inference/v1/generate"

async def test_bfloat16():
    print("\n" + "="*80)
    print("TESTING WITH BFLOAT16 (where bug occurs on MI355X)")
    print("="*80)

    base_args = [
        "--dtype", "bfloat16",  # ← Using bfloat16, not float16
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
    print(f"dtype: bfloat16")
    print(f"Prefix caching: ENABLED (default)")

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
                    "logprobs": 3,
                },
                "stream": False,
            }

            print("\n" + "-"*80)
            print("Running 5 requests with PREFIX CACHING + BFLOAT16")
            print("-"*80)

            outputs = []
            for i in range(5):
                resp = await client.post(GEN_ENDPOINT, json=payload)
                data = resp.json()
                tokens = data["choices"][0]["token_ids"]
                logprobs_data = data["choices"][0].get("logprobs", {}).get("content", [])

                # Get logprobs for position 3 (where divergence occurs)
                pos3_logprob = None
                if len(logprobs_data) >= 4:
                    pos3_logprob = logprobs_data[3].get("logprob")

                decoded = tokenizer.decode(tokens, skip_special_tokens=True)
                outputs.append((tokens, decoded, pos3_logprob))
                print(f"  Run {i+1}: '{decoded}'")
                if pos3_logprob is not None:
                    print(f"         Position 3 logprob: {pos3_logprob:.6f}")

            print("\n" + "="*80)
            print("RESULTS")
            print("="*80)

            all_same = all(tokens == outputs[0][0] for tokens, _, _ in outputs)
            first_differs = (
                outputs[0][0] != outputs[1][0] and
                all(tokens == outputs[1][0] for tokens, _, _ in outputs[1:])
            )

            if all_same:
                print("✓ ALL 5 RUNS IDENTICAL (FIX WORKED!)")
                print(f"  Output: '{outputs[0][1]}'")
            elif first_differs:
                print("✗ FIRST RUN DIFFERS (BUG STILL PRESENT)")
                print(f"  Run 1: '{outputs[0][1]}'")
                print(f"  Run 2-5: '{outputs[1][1]}'")

                # Show logprob differences at position 3
                if outputs[0][2] is not None and outputs[1][2] is not None:
                    diff = outputs[0][2] - outputs[1][2]
                    print(f"\n  Position 3 logprob difference: {diff:.6f}")
                    print(f"    Run 1: {outputs[0][2]:.6f}")
                    print(f"    Run 2: {outputs[1][2]:.6f}")
            else:
                print("✗ MULTIPLE DIFFERENT OUTPUTS (UNEXPECTED)")
                for i, (tokens, decoded, _) in enumerate(outputs):
                    print(f"  Run {i+1}: '{decoded}'")

            print("="*80)

if __name__ == "__main__":
    asyncio.run(test_bfloat16())
