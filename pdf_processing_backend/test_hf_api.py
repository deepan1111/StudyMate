# test_hf_simple.py
import asyncio
import aiohttp
import json

async def test_hf_api():
    api_key = "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"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": "What is the capital of France?",
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.3
        }
    }
    
    url = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                print(f"Status: {response.status}")
                result = await response.json()
                print(f"Response: {json.dumps(result, indent=2)}")
                
                if response.status == 503:
                    print("Model is loading, try again in a few minutes")
                elif response.status == 429:
                    print("Rate limit reached")
                elif response.status == 200:
                    print("API is working correctly!")
                    
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_hf_api())