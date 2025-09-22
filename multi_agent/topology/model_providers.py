"""
model_providers.py
------------------
Defines the mapping between model names and their provider
(OpenAI, Anthropic, Google, Ollama, Together, etc).
"""

MODEL_PROVIDER = {
    # ---- OpenAI ----
    "gpt-4o-mini": "openai",
    "gpt-4.1-mini": "openai",
    "gpt-4o": "openai",
    "o1": "openai",
    "o3-mini": "openai",
    "o4-mini": "openai",

    # ---- Ollama ----
    "llama3.1": "ollama",
    "deepseek-llm:7b": "ollama",
    "qwen:7b": "ollama",
    "gpt-oss:20b":  "ollama",

    # ---- Google ----
    "gemini-2.0-flash": "google-genai",
    "gemini-2.0-flash-lite": "google-genai",
    "gemini-2.5-flash-preview-04-17": "google-genai",
    "gemini-2.5-flash-preview-04-17-thinking": "google-genai",
    "gemini-2.5-pro-exp-03-25": "google-genai",
    "gemini-2.5-pro-preview-03-25": "google-genai",
    "gemini-2.5-pro-preview-05-06": "google-genai",
    "gemini-1.5-pro": "google-genai",

    # ---- Anthropic ----
    "claude-3-5-haiku-20241022": "anthropic",
    "claude-3-opus-20240229": "anthropic",
    "claude-3-7-sonnet-20250219": "anthropic",
    "claude-3-7-sonnet-20250219-thinking": "anthropic",

    # ---- Together ----
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "together",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "together",
}
