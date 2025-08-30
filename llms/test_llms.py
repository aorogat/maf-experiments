"""
Quick test script for local and remote LLM integrations.
Run from project root:

    source mafenv/bin/activate
    python -m llms.test_llms
"""

from llms.local_llm import LocalOllamaLLM
from llms.remote_llm import OpenAILLM

def test_local():
    # model_name = "gpt-oss:20b"
    model_name = "deepseek-llm:7b"
    try:
        print("🔹 Testing Local Ollama LLM...")
        ollama_model = LocalOllamaLLM(model_name)
        response = ollama_model.generate("In one paragraph, write about multi-agent frameworks")
        print("✅ Local response:", response)
    except Exception as e:
        print("❌ Local Ollama test failed:", e)

def test_remote():
    try:
        print("🔹 Testing Remote OpenAI LLM...")
        openai_model = OpenAILLM(model="gpt-4o-mini")
        response = openai_model.generate("In one paragraph, write about multi-agent frameworks")
        print("✅ Remote response:", response)
    except Exception as e:
        print("❌ Remote OpenAI test failed:", e)

if __name__ == "__main__":
    test_local()
    test_remote()
