import os
from dotenv import load_dotenv
from anthropic import Anthropic

# Load environment variables from .env file
load_dotenv()

# Get the API key from the .env file
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

client = Anthropic(api_key=api_key)

# Updated list of known Claude models, including Claude 3.5
# https://docs.anthropic.com/en/docs/about-claude/models
known_models = [
    "claude-3-5-sonnet-20240620",  # Claude 3.5 Sonnet
    # Claude 3.5 Opus -- Later this year
    # Claude 3.5 Haiku -- Later this year
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
