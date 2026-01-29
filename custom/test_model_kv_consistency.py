#!/usr/bin/env python3
"""
Test if the MODEL produces consistent K/V values with bfloat16
This is the real test - does the model layer computation produce identical K/V?
"""

import torch
import asyncio
from transformers import AutoTokenizer
from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"

async def capture_kv_values(dtype_str, num_runs=2):
    """Capture actual K/V values from model computation"""
    print(f"\n{'='*80}")
    print(f"Capturing K/V values with {dtype_str}")
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

    # We'll monkey-patch the model to capture K/V values
    import sys
    import importlib

    kv_captures = []

    # Patch approach: we need to intercept where K/V are computed
    # This happens in the attention layer's QKV projection
    # For Qwen3, this is in the Qwen2Attention layer

    with RemoteOpenAIServer(MODEL_NAME, base_args) as server:
        # Import vllm after server starts
        if 'vllm.model_executor.models.qwen2' in sys.modules:
            importlib.reload(sys.modules['vllm.model_executor.models.qwen2'])

        import httpx
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
                    "max_tokens": 1,  # Just one token to keep it simple
                    "temperature": 0.0,
                    "detokenize": False,
                },
                "stream": False,
            }

            for run_idx in range(num_runs):
                print(f"\nRun {run_idx + 1}:")
                resp = await client.post("/inference/v1/generate", json=payload)
                data = resp.json()

                tokens = data["choices"][0]["token_ids"]
                decoded = tokenizer.decode(tokens, skip_special_tokens=True)

                print(f"  Output: '{decoded}'")
                print(f"  Token IDs: {tokens}")

                # Get logprobs
                if "logprobs" in data["choices"][0]:
                    logprobs = data["choices"][0]["logprobs"]
                    if logprobs and len(logprobs) > 0:
                        print(f"  First token logprob: {logprobs[0]}")

    return kv_captures

async def main():
    """Compare K/V values between dtypes"""
    print("\nTesting if model produces consistent K/V values...")

    # Test float16
    print("\n" + "#"*80)
    print("# FLOAT16")
    print("#"*80)
    fp16_kv = await capture_kv_values("float16", num_runs=3)

    # Test bfloat16
    print("\n" + "#"*80)
    print("# BFLOAT16")
    print("#"*80)
    bf16_kv = await capture_kv_values("bfloat16", num_runs=3)

    print("\n" + "="*80)
    print("NEXT STEP: Need to add K/V capture hooks to model")
    print("="*80)
    print("""
To actually capture K/V values, we need to:
1. Add hooks to the attention layer to intercept K/V tensors
2. Save them during forward pass
3. Compare across runs

This requires modifying the model code or using PyTorch hooks.
Let's try a different approach...
""")

if __name__ == "__main__":
    asyncio.run(main())
