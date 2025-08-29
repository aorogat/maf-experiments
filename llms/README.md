# LLMs Module

This folder centralizes all LLM integrations for the experiments.

- `base_llm.py` → abstract interface shared by all frameworks
- `local_llm.py` → local models (e.g., Ollama inside WSL)
- `remote_llm.py` → remote models (OpenAI, Anthropic, etc.)

### Usage Example

```python
from llms.local_llm import LocalOllamaLLM
from llms.remote_llm import OpenAILLM

# Local Ollama
local_model = LocalOllamaLLM("deepseek-llm:7b")
print(local_model.generate("Hello from Ollama!"))

# Remote OpenAI
remote_model = OpenAILLM("gpt-4o-mini", api_key="sk-...")
print(remote_model.generate("Hello from OpenAI!"))
```