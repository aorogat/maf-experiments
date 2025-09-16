# Run with: python -m llms.remote_llm
import os
from dotenv import load_dotenv
from openai import OpenAI
from .base_llm import BaseLLM

# Load variables from .env at project root
load_dotenv()

class OpenAILLM(BaseLLM):
    """Wrapper for OpenAI models like GPT-4, GPT-4o-mini, etc."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ Missing OpenAI API key. Please set OPENAI_API_KEY in your .env file."
            )
        # Initialize the OpenAI client once
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    llm = OpenAILLM(model="gpt-4o-mini")
    prompt = "What is 2+2?"
    print(f"Prompt: {prompt}")
    output = llm.generate(prompt, temperature=0, max_tokens=16)
    print(f"Output: {output}")
