"""
CrewAI baseline runner with Ollama.
Usage:
    python -m single_agent.framework_overhead.crewai_runner
"""

import time
from crewai import Agent, Task, Crew
from llms.local_llm import LocalOllamaLLM

QUESTION = "What is 2+2?"


class CrewAIRunner:
    """Reusable CrewAI runner for overhead experiments."""

    def __init__(self, model="ollama/deepseek-llm:7b"):
        # explicitly use Ollama instead of OpenAI
        self.llm = LocalOllamaLLM(model)

        # Define agent once
        self.agent = Agent(
            role="", # Must be a string (keep it empty string)
            goal="",
            backstory="",
            memory=False,
            planning=False,
            verbose=False,
            llm=self.llm   # 🚀 pass local Ollama model
        )

        # Define task template once (will update description later)
        self.task = Task(
            description="",
            expected_output="",
            agent=self.agent
        )

        # Build crew once
        self.crew = Crew(
            agents=[self.agent],
            tasks=[self.task],
            verbose=False
        )

    def run(self, question: str = QUESTION):
        """Run the same crew with a new question."""
        self.task.description = question
        start = time.perf_counter()
        result = self.crew.kickoff()
        end = time.perf_counter()
        return str(result).strip(), (end - start) * 1000


if __name__ == "__main__":
    runner = CrewAIRunner()
    print("=== CrewAI Test ===")
    for i in range(3):
        resp, latency = runner.run(QUESTION)
        print(f"Run {i+1}: Q={QUESTION} | A={resp} | ⏱️ {latency:.2f} ms")
