# single_agent/framework_overhead/ollama_adapter.py
from llms.local_llm import LocalOllamaLLM

class OllamaAdapter:
    """Adapter so Concordia can use LocalOllamaLLM instead of OpenAI."""

    def __init__(self, model_name="deepseek-llm:7b"):
        self.llm = LocalOllamaLLM(model_name)

    def sample_text(self, prompt: str, **kwargs):
        """Generate free-form text (Concordia expects this)."""
        return self.llm.generate(prompt)

    def sample_choice(self, question: str, options: list[str], **kwargs):
        """
        Multiple choice wrapper for Concordia.
        Must return (index, response, debug_info).
        """
        formatted = (
            f"Question: {question}\n"
            f"Options: {', '.join(options)}\n"
            f"Respond with exactly one option."
        )
        resp = self.llm.generate(formatted).strip()

        # Handle Yes/No
        if set(opt.lower() for opt in options) == {"yes", "no"}:
            if "yes" in resp.lower():
                return options.index("Yes"), "Yes", {"raw": resp}
            if "no" in resp.lower():
                return options.index("No"), "No", {"raw": resp}

        # Match any option
        for i, opt in enumerate(options):
            if opt.lower() in resp.lower():
                return i, opt, {"raw": resp}

        # Fallback to first option
        return 0, options[0], {"raw": resp}

    def sample_log_probs(self, prompt: str, **kwargs):
        """Stub to satisfy Concordia API."""
        return {"tokens": list(prompt), "logprobs": [0.0] * len(prompt)}
