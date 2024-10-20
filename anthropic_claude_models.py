import os
from anthropic import Anthropic

# Ensure you have set your API key as an environment variable
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

client = Anthropic(api_key=api_key)

# Updated list of known Claude models, including Claude 3.5
known_models = [
    "claude-3.5-sonnet-20240229",  # Claude 3.5 Sonnet
    "claude-3.5-opus-20240229",  # Claude 3.5 Opus
    "claude-3.5-haiku-20240307",  # Claude 3.5 Haiku
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-2.1",
    "claude-2.0",
    "claude-instant-1.2",
]

print("Known Claude models (including Claude 3.5):")
for model in known_models:
    print(f"- {model}")

print("\nTesting model availability:")

for model in known_models:
    try:
        # Attempt to create a message with each model
        response = client.messages.create(
            model=model, max_tokens=10, messages=[{"role": "user", "content": "Hello"}]
        )
        print(f"- {model}: Available")
    except Exception as e:
        print(f"- {model}: Not available or error ({str(e)})")

print(
    "\nNote: This script tests model availability based on known model names. "
    "It may not include all available models or reflect real-time changes in the API."
)
