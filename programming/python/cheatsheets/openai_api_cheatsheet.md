# OpenAI API Cheatsheet - Universal LLM Interface

## Overview

The OpenAI Python client provides a standard interface that works with:
- OpenAI models (GPT-4, GPT-3.5, etc.)
- Other providers (Anthropic, Google, etc. via compatible APIs)
- Local models (Ollama, LM Studio, vLLM, etc.)

## Installation

```bash
pip install openai
```

## Basic Setup

### OpenAI
```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-..."  # Or set OPENAI_API_KEY env variable
)
```

### Other Providers (OpenAI-compatible)
```python
from openai import OpenAI

# Ollama (local models)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Ollama doesn't need real key
)

# Azure OpenAI
client = OpenAI(
    api_key="your-azure-key",
    base_url="https://your-resource.openai.azure.com/",
    default_headers={"api-version": "2024-02-01"}
)

# LM Studio (local)
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

# vLLM (self-hosted)
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"
)

# Together AI
client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key="your-together-key"
)

# Anyscale
client = OpenAI(
    base_url="https://api.endpoints.anyscale.com/v1",
    api_key="your-anyscale-key"
)
```

## Chat Completions (Main API)

### Simple Request
```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### With System Prompt
```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"}
    ]
)
```

### Multi-turn Conversation
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."},
    {"role": "user", "content": "What are its main features?"}
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=messages
)
```

### Common Parameters
```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    
    # Temperature (0-2): Lower = more focused, Higher = more random
    temperature=0.7,
    
    # Max tokens to generate
    max_tokens=1000,
    
    # Top-p sampling (0-1): Alternative to temperature
    top_p=1.0,
    
    # Frequency penalty (-2 to 2): Penalize repeated tokens
    frequency_penalty=0.0,
    
    # Presence penalty (-2 to 2): Penalize tokens that appeared
    presence_penalty=0.0,
    
    # Stop sequences: Stop generation at these strings
    stop=["END", "\n\n"],
    
    # Number of completions to generate
    n=1,
    
    # Random seed for reproducibility
    seed=42
)
```

## Streaming Responses

### Basic Streaming
```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### Streaming with Error Handling
```python
try:
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True
    )
    
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
            
except Exception as e:
    print(f"Error: {e}")
```

## Response Structure

### Accessing Response Data
```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)

# Get message content
content = response.choices[0].message.content

# Get full message object
message = response.choices[0].message

# Get finish reason ("stop", "length", "content_filter")
finish_reason = response.choices[0].finish_reason

# Token usage
prompt_tokens = response.usage.prompt_tokens
completion_tokens = response.usage.completion_tokens
total_tokens = response.usage.total_tokens

# Model used
model = response.model

# Response ID
response_id = response.id
```

## Function Calling / Tool Use

### Define Tools
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]
```

### Request with Tools
```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"  # or "none", "required", or specific tool
)

# Check if model wants to call a function
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)
    
    # Call your function
    result = get_weather(**function_args)
    
    # Send result back
    messages.append(response.choices[0].message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })
    
    # Get final response
    final_response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools
    )
```

## Vision (Image Input)

### Image from URL
```python
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg"
                    }
                }
            ]
        }
    ],
    max_tokens=300
)
```

### Image from Base64
```python
import base64

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

base64_image = encode_image("photo.jpg")

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
)
```

## Embeddings

### Create Embeddings
```python
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input="Your text here"
)

embedding = response.data[0].embedding  # List of floats
```

### Batch Embeddings
```python
texts = ["Text 1", "Text 2", "Text 3"]

response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=texts
)

embeddings = [item.embedding for item in response.data]
```

## Error Handling

### Common Errors
```python
from openai import OpenAI, APIError, RateLimitError, APIConnectionError

client = OpenAI()

try:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except APIConnectionError as e:
    print(f"Connection error: {e}")
except APIError as e:
    print(f"API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Retry Logic
```python
import time
from openai import RateLimitError

def chat_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
```

## Async Support

### Async Client
```python
from openai import AsyncOpenAI
import asyncio

async def main():
    client = AsyncOpenAI(api_key="sk-...")
    
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
    
    print(response.choices[0].message.content)

asyncio.run(main())
```

### Async Streaming
```python
async def stream_response():
    client = AsyncOpenAI()
    
    stream = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Write a story"}],
        stream=True
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")

asyncio.run(stream_response())
```

### Parallel Requests
```python
async def parallel_requests():
    client = AsyncOpenAI()
    
    tasks = [
        client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Question {i}"}]
        )
        for i in range(5)
    ]
    
    responses = await asyncio.gather(*tasks)
    
    for i, response in enumerate(responses):
        print(f"Response {i}: {response.choices[0].message.content}")

asyncio.run(parallel_requests())
```

## Ollama-Specific Examples

### List Available Models
```python
# Ollama doesn't have official list endpoint via OpenAI interface
# Use Ollama CLI: ollama list
# Or use requests library:
import requests
models = requests.get("http://localhost:11434/api/tags").json()
```

### Using Ollama Models
```python
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Use any pulled model (llama2, mistral, codellama, etc.)
response = client.chat.completions.create(
    model="llama2",  # or "mistral", "codellama", etc.
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Ollama with Streaming
```python
stream = client.chat.completions.create(
    model="llama2",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Best Practices

### 1. Environment Variables
```python
import os
from openai import OpenAI

# Set in .env file or environment
# OPENAI_API_KEY=sk-...

client = OpenAI()  # Automatically reads OPENAI_API_KEY
```

### 2. Token Counting (Approximate)
```python
def estimate_tokens(text):
    """Rough estimate: ~4 chars per token for English"""
    return len(text) // 4

def count_message_tokens(messages):
    """Estimate tokens in message list"""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg["content"])
        total += 4  # Message formatting overhead
    total += 2  # Conversation formatting
    return total
```

### 3. Conversation Management
```python
class Conversation:
    def __init__(self, client, model="gpt-4", system_prompt=None):
        self.client = client
        self.model = model
        self.messages = []
        
        if system_prompt:
            self.messages.append({
                "role": "system",
                "content": system_prompt
            })
    
    def add_user_message(self, content):
        self.messages.append({"role": "user", "content": content})
    
    def get_response(self, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            **kwargs
        )
        
        assistant_message = response.choices[0].message.content
        self.messages.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def clear(self):
        system_messages = [m for m in self.messages if m["role"] == "system"]
        self.messages = system_messages

# Usage
conv = Conversation(client, system_prompt="You are a helpful assistant")
conv.add_user_message("What is Python?")
response = conv.get_response()
print(response)

conv.add_user_message("What are its benefits?")
response = conv.get_response()
print(response)
```

### 4. Cost Tracking
```python
# Approximate costs (as of 2024, check current pricing)
COSTS = {
    "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
    "gpt-3.5-turbo": {"input": 0.0005 / 1000, "output": 0.0015 / 1000}
}

def calculate_cost(response, model):
    if model not in COSTS:
        return 0
    
    input_cost = response.usage.prompt_tokens * COSTS[model]["input"]
    output_cost = response.usage.completion_tokens * COSTS[model]["output"]
    
    return input_cost + output_cost

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)

cost = calculate_cost(response, "gpt-4")
print(f"Cost: ${cost:.4f}")
```

## Common Model Names

### OpenAI Models
```python
# Chat models
"gpt-4"
"gpt-4-turbo-preview"
"gpt-4-vision-preview"
"gpt-3.5-turbo"
"gpt-3.5-turbo-16k"

# Embedding models
"text-embedding-ada-002"
"text-embedding-3-small"
"text-embedding-3-large"
```

### Ollama Models (examples)
```python
# After pulling with: ollama pull <model>
"llama2"
"llama2:13b"
"mistral"
"codellama"
"vicuna"
"phi"
"neural-chat"
```

## Quick Reference: Common Patterns

### Simple Chat
```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Your prompt"}]
)
print(response.choices[0].message.content)
```

### Streaming Chat
```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Your prompt"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### With Temperature
```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Be creative"}],
    temperature=0.9
)
```

### JSON Mode (OpenAI only)
```python
response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "system", "content": "You respond in JSON"},
        {"role": "user", "content": "Give me user data"}
    ],
    response_format={"type": "json_object"}
)
```

## Testing Connection

```python
def test_connection(client, model="gpt-3.5-turbo"):
    """Test if client is properly configured"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        print(f"✓ Connected! Model: {response.model}")
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

# Test OpenAI
client = OpenAI()
test_connection(client)

# Test Ollama
ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
test_connection(ollama_client, model="llama2")
```