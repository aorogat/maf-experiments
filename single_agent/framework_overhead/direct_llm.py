"""
Direct LLM baseline runner (Ollama or OpenAI).
Usage:
    python -m single_agent.framework_overhead.direct_llm

Backends supported:
  - OpenAI (default example uses: openai/gpt-4o-mini)
  - Ollama (example uses: ollama/deepseek-llm:7b)
"""

import os
import re
import time
import json
from typing import Optional, Tuple, Any, Iterable

from dotenv import load_dotenv

# Local Ollama wrapper (your project module)
from llms.local_llm import LocalOllamaLLM

# OpenAI SDK
from openai import OpenAI, RateLimitError, APIError


QUESTION = "What is 2+2?"


# --------------------------- helpers ---------------------------

def _infer_backend_and_model(backend: Optional[str], model: str) -> Tuple[str, str]:
    """
    Decide backend/model.
    If backend is None or "auto", infer from prefix "openai/" or "ollama/".
    If explicit backend is given, strip any known prefix from model.
    """
    if backend in (None, "auto"):
        m = re.match(r"^(openai|ollama)/(.*)$", model.strip())
        if m:
            return m.group(1), m.group(2)
        # default to ollama if no prefix is given
        return "ollama", model

    # explicit backend: strip prefix if present
    if model.startswith("openai/"):
        model = model.split("openai/", 1)[1]
    elif model.startswith("ollama/"):
        model = model.split("ollama/", 1)[1]
    return backend, model


def _normalize_ollama_output(obj: Any) -> str:
    """
    Handle common Ollama return shapes: str, dict, iterable of chunks.
    Expected text fields: 'response', 'text', or 'message'/'content'.
    """
    if isinstance(obj, str):
        return obj

    if isinstance(obj, Iterable) and not isinstance(obj, (bytes, dict)):
        return "".join(_normalize_ollama_output(x) for x in obj)

    if isinstance(obj, dict):
        # e.g., {"message": {"content": "..."}}
        if isinstance(obj.get("message"), dict):
            content = obj["message"].get("content")
            if isinstance(content, str):
                return content
        for k in ("response", "text", "content"):
            v = obj.get(k)
            if isinstance(v, str):
                return v

    return str(obj)


# --------------------------- runner ---------------------------

class DirectLLMRunner:
    """Reusable Direct LLM runner for overhead experiments."""

    def __init__(self, model: str = "openai/gpt-4o-mini", backend: Optional[str] = "auto"):
        """
        model:   "openai/gpt-4o-mini", "ollama/deepseek-llm:7b", or bare model id
        backend: "openai", "ollama", or "auto"/None to infer from model
        """
        load_dotenv()
        self.backend, self.model = _infer_backend_and_model(backend, model)

        if self.backend == "ollama":
            self.llm = LocalOllamaLLM(self.model)
            self.client = None
        elif self.backend == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing OPENAI_API_KEY in environment or .env")
            self.client = OpenAI(api_key=api_key)
            self.llm = None
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    # ---------- OpenAI: Chat Completions (GPT-4 family) ----------
    def _openai_call_chat(self, question: str, model_id: Optional[str] = None) -> str:
        """
        Chat Completions with GPT-4 family (uses max_tokens).
        Forces text output and prints a compact debug if content is empty.
        """
        sys_msg = "Answer with a single number. No words, no punctuation."
        model_id = model_id or self.model

        # tiny retry for transient RPM only
        for attempt in range(2):
            try:
                r = self.client.chat.completions.create(
                    model=model_id,  # e.g., "gpt-4o-mini"
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": question},
                    ],
                    temperature=0,
                    max_tokens=16,                      # safe minimum
                    response_format={"type": "text"},   # force text content
                )
                choice = r.choices[0] if r.choices else None
                msg = choice.message if choice else None
                txt = (msg.content or "").strip() if msg else ""
                if not txt:
                    try:
                        payload = r.model_dump() if hasattr(r, "model_dump") else json.loads(r.json())
                    except Exception:
                        payload = repr(r)
                    print("   [DEBUG] Chat finish_reason:", getattr(choice, "finish_reason", None))
                    print("   [DEBUG] Chat msg type:", type(msg))
                    print("   [DEBUG] Chat raw (truncated):",
                          (json.dumps(payload)[:800] + "...") if isinstance(payload, dict)
                          else str(payload)[:800] + "...")
                return txt
            except RateLimitError as e:
                m = re.search(r"try again in ([\d\.]+)s", str(e))
                if m and attempt == 0:
                    time.sleep(min(float(m.group(1)), 2.0))
                    continue
                raise
            except APIError as e:
                print("   [DEBUG] OpenAI APIError:", e)
                break

        return ""

    # ---------- public API ----------
    def run(self, question: str = QUESTION):
        """
        Execute one LLM call and return (response_text, latency_ms).
        """
        t0 = time.perf_counter()

        if self.backend == "ollama":
            # Coerce terse numeric output from local model
            prompt = f"Answer with a single number. No words, no punctuation.\n{question}"
            raw = self.llm.generate(prompt)
            response = _normalize_ollama_output(raw).strip()

        elif self.backend == "openai":
            response = self._openai_call_chat(question).strip()

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        t1 = time.perf_counter()

        if not response:
            print(f"   [DEBUG] Empty response. Backend: {self.backend} Model: {self.model}")

        return response, (t1 - t0) * 1000.0  # ms


# --------------------------- quick self-test ---------------------------

if __name__ == "__main__":
    print("=== Direct LLM Test ===")
    print("   [DEBUG] Running file:", __file__)

    # OpenAI: GPT-4o-mini
    runner_openai = DirectLLMRunner(model="openai/gpt-4o-mini", backend="auto")
    resp, latency = runner_openai.run(QUESTION)
    print(f"[OPENAI] Q={QUESTION} | A={resp} | ⏱️ {latency:.2f} ms")

    # Ollama: deepseek-llm:7b (optional local baseline)
    runner_ollama = DirectLLMRunner(model="ollama/deepseek-llm:7b", backend="auto")
    resp, latency = runner_ollama.run(QUESTION)
    print(f"[OLLAMA] Q={QUESTION} | A={resp} | ⏱️ {latency:.2f} ms")
