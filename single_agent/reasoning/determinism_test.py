"""
Determinism Test Runner
-----------------------
Run the CrewAI planning experiment repeatedly on MATH-100 to measure
LLM determinism (mean/std of accuracy & runtime, plus prediction consistency).

Models: gpt-5.6-luna, gpt-5.6-terra, groq/llama-3.1-8b-instant
  (Groq Llama 3.1 8B replaces local ollama/deepseek-llm:7b for speed.)
Mode:   planning=True only
Runs:   5 per model

Usage:
    source mafenv/bin/activate
    python -m single_agent.reasoning.determinism_test
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/shared_mnt/MASBench/.env", override=True)

from single_agent.reasoning.config import CONFIG
from single_agent.reasoning.crewai_test import run_crewai_on_benchmark, sanitize_filename_component
from single_agent.reasoning.crew_math import SingleAgentCrewMATH
from benchmarks.math import MATHBenchmark


# Luna/terra already completed; only Groq Llama remains.
MODELS = ["groq/llama-3.1-8b-instant"]
N_RUNS = 5
OUT_DIR = "results/planning/determinism"


def main():
    for model in MODELS:
        llm_tag = sanitize_filename_component(model)
        print("\n==============================================")
        print(f"   Determinism: LLM = {model}")
        print("==============================================")

        for run in range(1, N_RUNS + 1):
            out_name = f"crewai_math_planning_{llm_tag}_run{run}.json"
            out_path = Path(OUT_DIR) / out_name
            if out_path.exists():
                print(f"\n=== Skip existing {out_name} ===")
                continue

            CONFIG.update({
                "llm": model,
                "planning_llm": model,
                "math_judge_llm": "gpt-4o-mini",
                "planning": True,
                "benchmarks": ["math"],
                "n_math": 100,
                "results_dir": OUT_DIR,
            })

            print(f"\n=== Run {run}/{N_RUNS} | planning=True | math-100 | {model} ===")
            bench = MATHBenchmark(root="data/MATH/test", n=100)
            run_crewai_on_benchmark(
                bench,
                SingleAgentCrewMATH,
                "logs/SingleAgentCrewMATH.json",
                out_name,
            )

    print("\n=== Determinism Test Completed ===")
    print(f"Results written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
