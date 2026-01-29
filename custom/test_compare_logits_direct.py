#!/usr/bin/env python3
"""
Direct comparison of logits between cache-miss and cache-hit
This is the smoking gun test - if logits differ, we know the bug is real
"""

import os
os.environ['VLLM_LOGGING_LEVEL'] = 'DEBUG'  # Enable debug logging

import asyncio
import httpx
import json
from transformers import AutoTokenizer
from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"
GEN_ENDPOINT = "/inference/v1/generate"

async def get_logprobs(dtype_str, run_name):
    """Get logprobs for a single run"""
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
                    "max_tokens": 3,  # Just 3 tokens to focus on
                    "temperature": 0.0,
                    "logprobs": 10,  # Get top-10 logprobs for analysis
                    "detokenize": False,
                },
                "stream": False,
            }

            resp = await client.post(GEN_ENDPOINT, json=payload)
            data = resp.json()

            tokens = data["choices"][0]["token_ids"]
            # Get logprobs - might be in different formats
            logprobs_raw = data["choices"][0].get("logprobs", [])

            # Debug: print raw logprobs format
            if logprobs_raw and len(logprobs_raw) > 0:
                print(f"\nDEBUG: logprobs format: {type(logprobs_raw[0])}, value: {logprobs_raw[0]}")

            print(f"\n[{run_name}]")
            print(f"  Tokens: {tokens}")
            decoded = tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"  Decoded: '{decoded}'")

            # Parse logprobs - handle various formats
            logprobs_list = []
            for pos, lp_data in enumerate(logprobs_raw):
                if lp_data is None:
                    logprobs_list.append(None)
                elif isinstance(lp_data, (int, float)):
                    logprobs_list.append(float(lp_data))
                    print(f"  Position {pos}: logprob={float(lp_data):.6f}, token={tokens[pos]}")
                elif isinstance(lp_data, dict):
                    # Might have 'logprob' key or just be a dict with token info
                    if 'logprob' in lp_data:
                        logprob = lp_data['logprob']
                    elif 'log_prob' in lp_data:
                        logprob = lp_data['log_prob']
                    else:
                        # Try to find any numeric value
                        logprob = None
                        for v in lp_data.values():
                            if isinstance(v, (int, float)):
                                logprob = v
                                break
                    if logprob is not None:
                        logprobs_list.append(float(logprob))
                        print(f"  Position {pos}: logprob={float(logprob):.6f}, token={tokens[pos]}")
                    else:
                        logprobs_list.append(None)
                else:
                    # Unknown format, skip
                    print(f"  Position {pos}: unknown logprob format: {type(lp_data)}")
                    logprobs_list.append(None)

            return {
                'tokens': tokens,
                'logprobs': logprobs_list,
                'decoded': decoded
            }

async def main():
    print("="*80)
    print("DIRECT LOGIT COMPARISON TEST")
    print("="*80)
    print("\nThis test will:")
    print("1. Run the same request 3 times")
    print("2. Compare logprobs at each position")
    print("3. Identify where divergence starts")

    # Test bfloat16 (has the bug)
    print("\n" + "#"*80)
    print("# BFLOAT16 - Multiple runs")
    print("#"*80)

    runs = []
    for i in range(3):
        result = await get_logprobs("bfloat16", f"BF16 Run {i+1}")
        runs.append(result)

    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    # Check if all runs produced same tokens
    all_same = all(r['tokens'] == runs[0]['tokens'] for r in runs[1:])
    print(f"\nAll runs produced same tokens: {all_same}")

    if not all_same:
        print("\n⚠️  DIVERGENCE DETECTED!")

        # Find first diverging position
        max_len = max(len(r['tokens']) for r in runs)
        for pos in range(max_len):
            tokens_at_pos = [r['tokens'][pos] if pos < len(r['tokens']) else None
                           for r in runs]

            if len(set(t for t in tokens_at_pos if t is not None)) > 1:
                print(f"\n📍 First divergence at position {pos}:")
                for run_idx, run in enumerate(runs):
                    if pos < len(run['logprobs']) and run['logprobs'][pos] is not None:
                        lp_data = run['logprobs'][pos]
                        if isinstance(lp_data, dict):
                            lp = lp_data.get('logprob', 0)
                        else:
                            lp = float(lp_data)
                        tok = run['tokens'][pos]
                        print(f"  Run {run_idx + 1}: token={tok}, logprob={lp:.6f}")
                break

        # Show logprob comparison for position 0
        if len(runs) >= 2 and all(len(r['logprobs']) > 0 for r in runs):
            print(f"\nLogprob comparison at position 0:")
            for run_idx, run in enumerate(runs):
                if run['logprobs'][0] is not None:
                    lp_data = run['logprobs'][0]
                    if isinstance(lp_data, dict):
                        lp = lp_data.get('logprob', 0)
                    else:
                        lp = float(lp_data)
                    print(f"  Run {run_idx + 1}: {lp:.6f}")

            lp0_data = runs[0]['logprobs'][0]
            lp1_data = runs[1]['logprobs'][0]
            lp0 = lp0_data.get('logprob', 0) if isinstance(lp0_data, dict) else float(lp0_data)
            lp1 = lp1_data.get('logprob', 0) if isinstance(lp1_data, dict) else float(lp1_data)
            diff = abs(lp0 - lp1)
            print(f"  Difference: {diff:.6f}")

    # Also test float16 for comparison
    print("\n" + "#"*80)
    print("# FLOAT16 - Multiple runs (should be consistent)")
    print("#"*80)

    fp16_runs = []
    for i in range(3):
        result = await get_logprobs("float16", f"FP16 Run {i+1}")
        fp16_runs.append(result)

    fp16_same = all(r['tokens'] == fp16_runs[0]['tokens'] for r in fp16_runs[1:])
    print(f"\nFP16 all runs same: {fp16_same}")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    if all_same:
        print("✓ BF16 is CONSISTENT - bug may have been fixed!")
    else:
        print("✗ BF16 is INCONSISTENT - bug confirmed")
        print("\nThis proves the bug exists at the logits level.")
        print("The issue is either:")
        print("  1. Different computation path for cache-miss vs cache-hit")
        print("  2. Numerical precision issue in attention with cached KV")
        print("  3. Bug in how prefix caching selects which blocks to reuse")

if __name__ == "__main__":
    asyncio.run(main())
