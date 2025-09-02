"""
LangGraph baseline runner with Ollama.
Usage:
    python -m single_agent.framework_overhead.langgraph_runner
"""

import time
from langgraph.graph import StateGraph
from llms.local_llm import LocalOllamaLLM

QUESTION = "What is 2+2?"


class LangGraphRunner:
    """Reusable LangGraph runner for overhead experiments."""

    def __init__(self, model="deepseek-llm:7b"):
        # Initialize local Ollama model once
        self.llm = LocalOllamaLLM(model)

        # Define minimal state graph with one node
        workflow = StateGraph(dict)

        def node_llm(state: dict):
            """Pass-through node that queries the LLM."""
            response = self.llm.generate(state["question"])
            return {"answer": response}

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
    runner = LangGraphRunner()
    print("=== LangGraph Test ===")
    for i in range(3):
        resp, latency = runner.run(QUESTION)
        print(f"Run {i+1}: Q={QUESTION} | A={resp} | ⏱️ {latency:.2f} ms")
