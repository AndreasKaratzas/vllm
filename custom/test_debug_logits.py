#!/usr/bin/env python3
"""
Debug script to capture logits at different stages
This will help us identify WHERE the divergence happens
"""

import asyncio
import httpx
from transformers import AutoTokenizer
from tests.utils import RemoteOpenAIServer
import numpy as np

MODEL_NAME = "Qwen/Qwen3-0.6B"
GEN_ENDPOINT = "/inference/v1/generate"

async def test_with_logprobs(dtype_str):
    print(f"\n{'='*80}")
    print(f"Testing {dtype_str} with detailed logprobs")
    print('='*80)

    base_args = [
        "--dtype", dtype_str,
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

    print(f"Input length: {len(token_ids)} tokens")

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
                    "max_tokens": 10,  # Generate 10 tokens
                    "temperature": 0.0,
                    "logprobs": 5,  # Get top-5 logprobs
                    "detokenize": False,
                },
                "stream": False,
            }

            runs = []
            for run_idx in range(3):
                print(f"\n{'-'*80}")
                print(f"Run {run_idx + 1}")
                print('-'*80)

                resp = await client.post(GEN_ENDPOINT, json=payload)
                data = resp.json()

                tokens = data["choices"][0]["token_ids"]
                decoded = tokenizer.decode(tokens, skip_special_tokens=True)
                logprobs = data["choices"][0].get("logprobs", [])

                print(f"Output: '{decoded}'")
                print(f"Tokens: {tokens}")

                if logprobs:
                    print("\nDetailed logprobs for each position:")
                    for pos, lp in enumerate(logprobs):
                        if lp:
                            print(f"  Position {pos}: logprob={lp.get('logprob', 'N/A'):.6f}, "
                                  f"token={lp.get('decoded_token', 'N/A')}")

                runs.append({
                    'tokens': tokens,
                    'decoded': decoded,
                    'logprobs': logprobs
                })

            # Compare runs
            print(f"\n{'='*80}")
            print("COMPARISON")
            print('='*80)

            all_match = all(r['tokens'] == runs[0]['tokens'] for r in runs[1:])
            print(f"All runs match: {all_match}")

            if not all_match:
                print("\nDivergence details:")
                for pos in range(min(len(r['tokens']) for r in runs)):
                    tokens_at_pos = [r['tokens'][pos] for r in runs]
                    if len(set(tokens_at_pos)) > 1:
                        print(f"\nPosition {pos} DIFFERS:")
                        for run_idx, run in enumerate(runs):
                            if pos < len(run['logprobs']) and run['logprobs'][pos]:
                                lp = run['logprobs'][pos].get('logprob', 0.0)
                                tok = run['logprobs'][pos].get('decoded_token', '?')
                                print(f"  Run {run_idx + 1}: token={run['tokens'][pos]}, "
                                      f"'{tok}', logprob={lp:.6f}")

                # Show where they START to differ
                first_diff_pos = None
                for pos in range(min(len(r['tokens']) for r in runs)):
                    tokens_at_pos = [r['tokens'][pos] for r in runs]
                    if len(set(tokens_at_pos)) > 1:
                        first_diff_pos = pos
                        break

                if first_diff_pos is not None:
                    print(f"\n⚠️  First divergence at position {first_diff_pos}")
                    print(f"   This is the {first_diff_pos + 1}th generated token")
                    if first_diff_pos == 0:
                        print("   → Divergence happens on FIRST generated token")
                        print("   → This means prefix cache is affecting the computation")
                    else:
                        print(f"   → Generated {first_diff_pos} consistent tokens before diverging")

            return runs

async def main():
    # Test float16
    print("\n" + "#"*80)
    print("# FLOAT16 (should be consistent)")
    print("#"*80)
    fp16_runs = await test_with_logprobs("float16")

    # Test bfloat16
    print("\n" + "#"*80)
    print("# BFLOAT16 (has the bug)")
    print("#"*80)
    bf16_runs = await test_with_logprobs("bfloat16")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    fp16_consistent = all(r['tokens'] == fp16_runs[0]['tokens'] for r in fp16_runs[1:])
    bf16_consistent = all(r['tokens'] == bf16_runs[0]['tokens'] for r in bf16_runs[1:])

    print(f"FP16 consistent:  {fp16_consistent}")
    print(f"BF16 consistent:  {bf16_consistent}")

if __name__ == "__main__":
    asyncio.run(main())
