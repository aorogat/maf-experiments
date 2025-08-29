import subprocess
from .base_llm import BaseLLM

class LocalOllamaLLM(BaseLLM):
    """Wrapper for Ollama local models inside WSL."""

    def __init__(self, model: str = "deepseek-llm:7b"):
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        result = subprocess.run(
            ["ollama", "run", self.model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return result.stdout.decode("utf-8").strip()
