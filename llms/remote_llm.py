import os
import openai
from dotenv import load_dotenv
from .base_llm import BaseLLM

# Load variables from .env at project root
load_dotenv()

class OpenAILLM(BaseLLM):
    """Wrapper for OpenAI models like GPT-4, GPT-4o-mini, etc."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        self.model = model
        # Prefer passed key → else read from .env → else error
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "❌ Missing OpenAI API key. "
                "Set OPENAI_API_KEY in .env or pass api_key to OpenAILLM()."
            )
        openai.api_key = self.api_key

    def generate(self, prompt: str, **kwargs) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response["choices"][0]["message"]["content"]
