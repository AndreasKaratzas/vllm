#!/usr/bin/env python3
"""Debug what the API is actually returning."""
import asyncio
import httpx
import json
import os

# Set env vars
os.environ['VLLM_DEBUG_PREFIX_CACHE'] = '1'
os.environ['VLLM_DETERMINISTIC_PREFIX_CACHE'] = '1'

GEN_ENDPOINT = "/inference/v1/generate"

async def test_response():
    """Check actual API response format."""
    import subprocess
    import time
    
    # Start server
    env = os.environ.copy()
    env['HIP_VISIBLE_DEVICES'] = '4,5,6,7'
    
    server = subprocess.Popen(
        [
            'vllm', 'serve', 'Qwen/Qwen3-0.6B',
            '--dtype', 'bfloat16',
            '--max-model-len', '1024',
            '--enforce-eager',
            '--port', '9000',
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    print("Waiting for server...")
    time.sleep(25)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Try the endpoint
            payload = {
                "model": "Qwen/Qwen3-0.6B",
                "token_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "sampling_params": {
                    "max_tokens": 5,
                    "temperature": 0.0,
                    "detokenize": False,
                    "logprobs": 5,
                },
                "stream": False,
            }
            
            print(f"\nSending request to http://localhost:9000{GEN_ENDPOINT}")
            print(f"Payload: {json.dumps(payload, indent=2)}")
            
            resp = await client.post(f"http://localhost:9000{GEN_ENDPOINT}", json=payload)
            
            print(f"\nStatus: {resp.status_code}")
            print(f"Headers: {dict(resp.headers)}")
            
            try:
                data = resp.json()
                print(f"\nResponse JSON:")
                print(json.dumps(data, indent=2))
                
                if 'choices' in data:
                    print("\n✓ Has 'choices' key")
                else:
                    print(f"\n✗ Missing 'choices' key!")
                    print(f"Available keys: {list(data.keys())}")
                    
            except Exception as e:
                print(f"\n✗ Failed to parse JSON: {e}")
                print(f"Raw response: {resp.text}")
                
    finally:
        server.terminate()
        server.wait()
        print("\nServer stopped")

if __name__ == "__main__":
    asyncio.run(test_response())
