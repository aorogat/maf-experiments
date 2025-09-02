"""
Run Framework Overhead Experiment
Usage:
    python -m single_agent.framework_overhead.run_overhead
"""

import os
import time
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import your runners
from single_agent.framework_overhead.direct_llm import DirectLLMRunner
from single_agent.framework_overhead.crewai_runner import CrewAIRunner
from single_agent.framework_overhead.langgraph_runner import LangGraphRunner
from single_agent.framework_overhead.concordia_runner import ConcordiaRunner

QUESTION = "What is 2+2?"
TRIALS = 20   # 🔹 Adjust to 1000 for final experiment
CONCURRENCY = 4
RESULTS_DIR = "results/framework_overhead"


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def run_trials(runner, name, trials=TRIALS, concurrency=CONCURRENCY):
    """Run N trials of a runner and collect metrics."""
    latencies = []
    responses = []

    start_total = time.perf_counter()

    def _one_run(i):
        resp, latency = runner.run(QUESTION)
        return latency, resp

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_one_run, i) for i in range(trials)]
        for f in as_completed(futures):
            latency, resp = f.result()
            latencies.append(latency)
            responses.append(resp)

    end_total = time.perf_counter()
    throughput = trials / (end_total - start_total)

    return {
        "name": name,
        "p50_latency": statistics.median(latencies),
        "p95_latency": statistics.quantiles(latencies, n=100)[94],  # 95th percentile
        "throughput": throughput,
        "responses": responses[:3],  # preview first few
    }


def save_results(results, filename="framework_overhead.json"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"📄 Saved results to {out_path}")


# -------------------------------------------------------------------
# Main Experiment
# -------------------------------------------------------------------

def main():
    runners = {
        "Direct LLM": DirectLLMRunner(model="deepseek-llm:7b"),
        "CrewAI": CrewAIRunner(model="ollama/deepseek-llm:7b"),
        "LangGraph": LangGraphRunner(model="deepseek-llm:7b"),
        "Concordia": ConcordiaRunner(model_name="deepseek-llm:7b"),
    }

    all_results = []

    for name, runner in runners.items():
        print(f"\n=== Running {name} for {TRIALS} trials ===")
        results = run_trials(runner, name, TRIALS, CONCURRENCY)
        all_results.append(results)
        print(f"{name}: p50={results['p50_latency']:.2f} ms | "
              f"p95={results['p95_latency']:.2f} ms | "
              f"Throughput={results['throughput']:.2f} req/s")

    save_results(all_results)


if __name__ == "__main__":
    main()
