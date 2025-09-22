"""
python -m multi_agent.topology.run_all_experiments
----------------------

Master runner for scalability experiments across multiple frameworks.

- LangGraph (ws, ba, dt)
- Concordia (all-to-all connectivity, derived from HF graphs)
- CrewAI (TODO: to be implemented later)

Usage:
    python -m multi_agent.topology.run_all_experiments
"""

import asyncio
import subprocess
import sys
import itertools
import json
import glob
import pandas as pd
from pathlib import Path

# --- Configurations ---
# TASKS = ["coloring", "matching", "vertex_cover", "leader_election", "consensus"]
TASKS = ["coloring"]
SIZES = [50]

# TASKS = ["coloring"]
# SIZES = [4]

FRAMEWORKS = {
    "langgraph_ws": ("langgraph", "ws"),
    # "langgraph_ba": ("langgraph", "ba"),
    # "langgraph_dt": ("langgraph", "dt"),
    # "concordia": ("concordia", "ws"),   # HF graph, patched to all-connected inside concordia_runner
    # "crewai_seq": ("crewai", None),     # TODO
    # "crewai_hier": ("crewai", None),    # TODO
}

MODEL = "gpt-4o-mini"
# MODEL = "gpt-oss:20b"
ROUNDS = 8   # fixed max rounds for all experiments
SAMPLES = 1
RESULTS_DIR = Path("results/ontology")
AGGREGATED_CSV = Path("results/ontology/scalability_summary.csv")
AGGREGATED_JSON = Path("results/ontology/scalability_summary.json")


async def run_experiment(task, fw_key, framework, graph_model, size):
    """
    Run one experiment via subprocess, sequentially.
    """
    cmd = [
        sys.executable, "-m", "multi_agent.topology.runner",
        "--task", task,
        "--model", MODEL,
        "--graph_size", str(size),
        "--samples_per_graph_model", str(SAMPLES),
        "--rounds", str(ROUNDS),
        "--framework", framework,
    ]
    if graph_model:
        cmd += ["--graph_models", graph_model]

    print(f"\n[RUN] Task={task}, Framework={framework}, Graph={graph_model}, Size={size}")
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    # Stream logs line by line
    async for line in process.stdout:
        print(line.decode().rstrip())

    await process.wait()


def aggregate_results():
    """
    Aggregate all JSON result files into a single CSV + JSON.
    """
    files = glob.glob(str(RESULTS_DIR / "*.json"))
    rows = []
    for f in files:
        try:
            with open(f, "r") as fp:
                data = json.load(fp)
                rows.append({
                    "task": data.get("task"),
                    "framework": data.get("framework", "unknown"),       # NEW: true framework
                    "graph_model": data.get("graph_generator", "unknown"),  # NEW: topology
                    "n": data.get("num_nodes"),
                    "rounds": data.get("rounds"),
                    "score": data.get("score"),
                    "runtime": data.get("runtime_seconds"),
                    "success": data.get("successful"),
                    "error": data.get("error_message"),
                })
        except Exception as e:
            print(f"[WARN] Failed to load {f}: {e}")

    if not rows:
        print("[!] No result files found to aggregate.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(AGGREGATED_CSV, index=False)
    with open(AGGREGATED_JSON, "w") as out:
        json.dump(rows, out, indent=2)

    print(f"[OK] Aggregated {len(rows)} runs → {AGGREGATED_CSV}, {AGGREGATED_JSON}")


async def main():
    for task, (fw_key, (framework, graph_model)), size in itertools.product(
        TASKS, FRAMEWORKS.items(), SIZES
    ):
        # Skip CrewAI until implemented
        if framework == "crewai":
            print(f"[SKIP] CrewAI not implemented yet: {fw_key}, Task={task}, Size={size}")
            continue

        # Skip tasks not supported in Concordia
        if framework == "concordia" and task in ["coloring", "vertexcover"]:
            print(f"[SKIP] Concordia not suitable for {task}, skipping size={size}")
            continue

        # Run sequentially
        await run_experiment(task, fw_key, framework, graph_model, size)

    # Aggregate after all runs
    aggregate_results()


if __name__ == "__main__":
    asyncio.run(main())
