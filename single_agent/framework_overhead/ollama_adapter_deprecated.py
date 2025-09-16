# single_agent/framework_overhead/ollama_adapter.py
import json
from llms.local_llm import LocalOllamaLLM

class OllamaAdapter:
    def __init__(self, model="gpt-oss:20b"):
        self.llm = LocalOllamaLLM(model)

    def generate(self, text: str) -> str:
        return self.llm.generate(text)
    
    def sample_text(self, prompt: str, **kwargs):
        """Generate free-form text (Concordia expects this)."""
        return self.llm.generate(prompt)

    def sample_choice(self, question=None, options=None, **kwargs):
        """
        Concordia may call with:
          - sample_choice(question, options)
          - sample_choice(prompt=..., options=...)
          - sample_choice(options=...) (no question at all)
        This implementation is robust to all cases.
        """
        # Map alternative keyword
        if question is None and "prompt" in kwargs:
            question = kwargs["prompt"]
        if options is None and "options" in kwargs:
            options = kwargs["options"]

        # If still missing, assign defaults
        if question is None:
            question = "Choose the best option."
        if options is None or not isinstance(options, (list, tuple)) or len(options) == 0:
            options = ["Yes", "No"]

        # Ask the LLM
        prompt = f"""
You are given a multiple-choice question.

Question: {question}
Options: {options}

Respond ONLY in strict JSON format:
{{"choice": <index of best option>}}
        """
        raw_response = self.llm.generate(prompt).strip()

        # Default choice
        idx = 0
        try:
            parsed = json.loads(raw_response)
            if isinstance(parsed, dict) and "choice" in parsed:
                choice_val = parsed["choice"]
                if isinstance(choice_val, int) and 0 <= choice_val < len(options):
                    idx = choice_val
        except Exception:
            # fallback: keep idx=0
            pass

        return idx, raw_response, {"raw": raw_response}
