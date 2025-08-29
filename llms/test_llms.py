"""
Quick test script for local and remote LLM integrations.
Run from project root:

    source mafenv/bin/activate
    python -m llms.test_llms
"""

from llms.local_llm import LocalOllamaLLM
from llms.remote_llm import OpenAILLM

def test_local():
    try:
        print("🔹 Testing Local Ollama LLM...")
        ollama_model = LocalOllamaLLM("deepseek-llm:7b")
        response = ollama_model.generate("Hello from Ollama!")
        print("✅ Local response:", response[:200], "...\n")
    except Exception as e:
        print("❌ Local Ollama test failed:", e)

def test_remote():
    try:
        print("🔹 Testing Remote OpenAI LLM...")
        openai_model = OpenAILLM(model="gpt-4o-mini")
        response = openai_model.generate("Hello from OpenAI!")
        print("✅ Remote response:", response[:200], "...\n")
    except Exception as e:
        print("❌ Remote OpenAI test failed:", e)

if __name__ == "__main__":
    test_local()
    test_remote()
