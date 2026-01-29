#!/usr/bin/env python3
"""
Simplest possible test - just compare outputs between runs
"""

import asyncio
import httpx
from transformers import AutoTokenizer
from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"

async def test_bf16():
    print("\n" + "="*80)
    print("SIMPLE BF16 CONSISTENCY TEST")
    print("="*80)

    base_args = [
        "--dtype", "bfloat16",
        "--max-model-len", "512",
        "--enforce-eager",
        "--generation-config", "vllm",
        "--max_num_seqs", "1",
    ]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How many countries are in the EU?"},
    ]
    token_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False
    )

    print(f"Input: {len(token_ids)} tokens")

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
                    "max_tokens": 5,
                    "temperature": 0.0,
                    "detokenize": True,
                },
                "stream": False,
            }

            runs = []
            for i in range(3):
                print(f"\n{'-'*80}")
                print(f"Run {i+1}")
                print('-'*80)

                resp = await client.post("/inference/v1/generate", json=payload)
                data = resp.json()

                # Handle error responses (no 'choices' key when engine crashes)
                if "choices" not in data:
                    logger.error(f"Run {i+1} failed. Response: {data}")
                    return False

                tokens = data["choices"][0].get("token_ids", [])
                text = data["choices"][0].get("text", "")

                print(f"  Output tokens: {tokens}")
                print(f"  Output text: '{text}'")

                runs.append({'tokens': tokens, 'text': text})

            # Compare
            print(f"\n{'='*80}")
            print("COMPARISON")
            print('='*80)

            all_match = all(r['tokens'] == runs[0]['tokens'] for r in runs[1:])
            print(f"All runs produced same tokens: {all_match}")

            if not all_match:
                print("\n⚠️  DIVERGENCE DETECTED!")
                for i, run in enumerate(runs):
                    print(f"  Run {i+1}: {run['tokens']} -> '{run['text']}'")

                # Find first diff
                max_len = max(len(r['tokens']) for r in runs)
                for pos in range(max_len):
                    tokens_at_pos = [r['tokens'][pos] if pos < len(r['tokens']) else None
                                   for r in runs]
                    if len(set(t for t in tokens_at_pos if t is not None)) > 1:
                        print(f"\n📍 First divergence at position {pos}")
                        for i, run in enumerate(runs):
                            if pos < len(run['tokens']):
                                print(f"  Run {i+1}: token={run['tokens'][pos]}")
                        break
            else:
                print("\n✓ All runs are CONSISTENT!")

            return all_match

if __name__ == "__main__":
    result = asyncio.run(test_bf16())
    print("\n" + "="*80)
    if not result:
        print("❌ BUG CONFIRMED - Outputs differ between runs")
        print("\nCheck the logs above for [KV_WRITE], [ATTN_FWD], [ATTN_OUT] messages")
        print("Look for where values first start to differ")
    else:
        print("✅ BUG FIXED - All runs are consistent!")
    print("="*80)
