"""Simple test to verify deterministic prefix cache implementation."""
import os
import asyncio
import httpx

async def test_single_request():
    """Test a single request with deterministic mode."""
    base_url = "http://localhost:8000"
    
    # Start server in background
    import subprocess
    import time
    
    env = os.environ.copy()
    env['VLLM_DEBUG_PREFIX_CACHE'] = '1'
    env['VLLM_DETERMINISTIC_PREFIX_CACHE'] = '1'
    env['HIP_VISIBLE_DEVICES'] = '4,5,6,7'
    
    server = subprocess.Popen(
        [
            'python', '-m', 'vllm.entrypoints.openai.api_server',
            '--model', 'Qwen/Qwen3-0.6B',
            '--dtype', 'bfloat16',
            '--max-model-len', '1024',
            '--enforce-eager',
            '--enable-prefix-caching',
            '--port', '8000',
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(20)
    
    try:
        # Test tokens
        test_prompt = "The capital of France is Paris. The capital of Spain is Madrid. The capital of Germany is"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            print(f"\nRequest 1:")
            response = await client.post(
                f"{base_url}/v1/completions",
                json={
                    "model": "Qwen/Qwen3-0.6B",
                    "prompt": test_prompt,
                    "max_tokens": 5,
                    "temperature": 0,
                    "logprobs": 5,
                }
            )
            
            print(f"Status: {response.status_code}")
            data = response.json()
            print(f"Response keys: {data.keys()}")
            if 'choices' in data:
                print(f"Choices: {data['choices']}")
                print(f"Text: {data['choices'][0]['text']}")
            else:
                print(f"Full response: {data}")
            
            print(f"\nRequest 2 (should use cache):")
            response2 = await client.post(
                f"{base_url}/v1/completions",
                json={
                    "model": "Qwen/Qwen3-0.6B",
                    "prompt": test_prompt,
                    "max_tokens": 5,
                    "temperature": 0,
                    "logprobs": 5,
                }
            )
            
            print(f"Status: {response2.status_code}")
            data2 = response2.json()
            if 'choices' in data2:
                print(f"Text: {data2['choices'][0]['text']}")
            else:
                print(f"Full response: {data2}")
    
    finally:
        server.terminate()
        server.wait()
        print("Server stopped")

if __name__ == "__main__":
    asyncio.run(test_single_request())
