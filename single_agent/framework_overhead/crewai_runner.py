"""
CrewAI baseline runner with OpenAI or Ollama.
Usage:
    python -m single_agent.framework_overhead.crewai_runner --model openai/gpt-5-nano
    python -m single_agent.framework_overhead.crewai_runner --model ollama/deepseek-llm:7b
"""

import time
import os
import argparse
from dotenv import load_dotenv

from crewai import Agent, Task, Crew
from llms.local_llm import LocalOllamaLLM
from openai import OpenAI


QUESTION = "What is 2+2?"


class CrewAIRunner:
    """Reusable CrewAI runner for overhead experiments."""

    def __init__(self, model="ollama/deepseek-llm:7b"):
        load_dotenv()

        self.is_openai = model.startswith("openai/")
        if self.is_openai:
            # Extract raw model name (e.g., "gpt-4o-mini")
            openai_model = model.split("openai/")[1]
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in .env")
            client = OpenAI(api_key=api_key)

            # Wrap OpenAI client so it can be used by CrewAI’s agent
            class OpenAICrewLLM:
                def __init__(self, client, model):
                    self.client = client
                    self.model = model
                def __call__(self, prompt: str) -> str:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return resp.choices[0].message.content.strip()

            self.llm = OpenAICrewLLM(client, openai_model)

        else:
            # Default: Ollama through our LocalOllamaLLM wrapper
            self.llm = LocalOllamaLLM(model)

        # Define agent once (uses CrewAI’s abstraction regardless of backend)
        self.agent = Agent(
            role="Math Solver",
            goal="Answer simple questions quickly.",
            backstory="This agent is a lightweight test worker for benchmarking.",
            memory=False,
            planning=False,
            verbose=False,
            llm=self.llm,  # 🔹 Now works for both OpenAI and Ollama
        )

        # Define task template once
        self.task = Task(
            description="",
            expected_output="Answer in plain text",
            agent=self.agent
        )

        # Build crew once
        self.crew = Crew(
            agents=[self.agent],
            tasks=[self.task],
            verbose=False
        )

    def run(self, question: str = QUESTION):
        """Run the same Crew with a new question, measuring overhead."""
        self.task.description = question
        start = time.perf_counter()
        result = self.crew.kickoff()
        end = time.perf_counter()
        return str(result).strip(), (end - start) * 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ollama/deepseek-llm:7b",
                        help="Model to use (e.g., 'ollama/deepseek-llm:7b' or 'openai/gpt-4o-mini')")
    args = parser.parse_args()

    runner = CrewAIRunner(model=args.model)
    print("=== CrewAI Test ===")
    for i in range(3):
        resp, latency = runner.run(QUESTION)
        print(f"Run {i+1}: Q={QUESTION} | A={resp} | ⏱️ {latency:.2f} ms")
