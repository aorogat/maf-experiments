"""
LangGraph baseline runner with OpenAI or Ollama.
Usage:
    python -m single_agent.framework_overhead.langgraph_runner --model ollama/deepseek-llm:7b
    python -m single_agent.framework_overhead.langgraph_runner --model openai/gpt-4o-mini
"""

import time
import os
import argparse
from dotenv import load_dotenv

from langgraph.graph import StateGraph
from llms.local_llm import LocalOllamaLLM
from openai import OpenAI


QUESTION = "What is 2+2?"


class LangGraphRunner:
    """Reusable LangGraph runner for overhead experiments."""

    def __init__(self, model="ollama/deepseek-llm:7b"):
        load_dotenv()

        if model.startswith("openai/"):
            # Expect models like "openai/gpt-4o-mini"
            self.is_openai = True
            self.model = model.split("openai/")[1]
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in .env")
            self.client = OpenAI(api_key=api_key)
        else:
            # Default to Ollama
            self.is_openai = False
            self.llm = LocalOllamaLLM(model)

        # Define minimal state graph with one node
        workflow = StateGraph(dict)

        def node_llm(state: dict):
            """Pass-through node that queries the chosen LLM."""
            if self.is_openai:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": state["question"]}],
                )
                answer = response.choices[0].message.content
            else:
                answer = self.llm.generate(state["question"])
            return {"answer": answer}

        workflow.add_node("llm_node", node_llm)
        workflow.set_entry_point("llm_node")
        workflow.set_finish_point("llm_node")

        # Compile into runnable graph
        self.graph = workflow.compile()

    def run(self, question: str = QUESTION):
        """Run the LangGraph workflow with a new question."""
        start = time.perf_counter()
        result = self.graph.invoke({"question": question})
        end = time.perf_counter()
        return str(result.get("answer")).strip(), (end - start) * 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ollama/deepseek-llm:7b",
                        help="Model to use (e.g., 'ollama/deepseek-llm:7b' or 'openai/gpt-4o-mini')")
    args = parser.parse_args()

    runner = LangGraphRunner(model=args.model)
    print("=== LangGraph Test ===")
    for i in range(3):
        resp, latency = runner.run(QUESTION)
        print(f"Run {i+1}: Q={QUESTION} | A={resp} | ⏱️ {latency:.2f} ms")
