"""
Scalability Dashboard for Multi-Agent Results

This script aggregates JSON result files and generates
an interactive HTML dashboard showing scalability metrics.

Run:
    python -m multi_agent.topology.visualization.scalability_dashboard results/ontology/*.json --output scalability.html
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_result(filepath: Path):
    """Load a JSON result file safely."""
    with open(filepath, "r") as f:
        return json.load(f)


def aggregate_results(files):
    """
    Organize results by task.
    Returns dict: task -> list of records
    Each record = {nodes, score, runtime, tokens}
    """
    results = defaultdict(list)

    for f in files:
        try:
            data = load_result(f)
        except Exception:
            continue

        task = data.get("task", "unknown")
        num_nodes = data.get("num_nodes", None)
        score = data.get("score", None)
        runtime = data.get("runtime_seconds", None)
        tokens = data.get("total_tokens", None)

        if num_nodes is None:
            # try parse from filename if missing
            name = f.stem
            if "nodes" in name:
                try:
                    num_nodes = int(name.split("nodes")[-1].split("_")[0])
                except Exception:
                    pass

        results[task].append({
            "nodes": num_nodes,
            "score": score,
            "runtime": runtime,
            "tokens": tokens
        })

    # sort each task by node count
    for task in results:
        results[task] = sorted(results[task], key=lambda x: x["nodes"] or 0)

    return results


def create_dashboard(results, output_path: Path):
    """
    Create an HTML dashboard with scalability plots for each task.
    """
    figs = []

    for task, records in results.items():
        nodes = [r["nodes"] for r in records]
        scores = [r["score"] for r in records]
        runtimes = [r["runtime"] for r in records]
        tokens = [r["tokens"] for r in records]

        fig = make_subplots(rows=1, cols=3, subplot_titles=("Accuracy", "Runtime (s)", "Total Tokens"))

        # Accuracy
        fig.add_trace(
            go.Scatter(x=nodes, y=scores, mode="lines+markers", name="Accuracy"),
            row=1, col=1
        )
        # Runtime
        fig.add_trace(
            go.Scatter(x=nodes, y=runtimes, mode="lines+markers", name="Runtime", line=dict(color="orange")),
            row=1, col=2
        )
        # Tokens
        fig.add_trace(
            go.Scatter(x=nodes, y=tokens, mode="lines+markers", name="Tokens", line=dict(color="green")),
            row=1, col=3
        )

        fig.update_layout(
            title_text=f"Scalability - {task.title()}",
            showlegend=False,
            height=400,
            width=1200
        )
        figs.append(fig)

    # Combine all tasks into one HTML file
    with open(output_path, "w") as f:
        for fig in figs:
            f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    print(f"✅ Dashboard saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate scalability dashboard")
    parser.add_argument("files", nargs="+", help="Result JSON files")
    parser.add_argument("--output", "-o", default="scalability.html", help="Output HTML file")
    args = parser.parse_args()

    files = [Path(f) for f in args.files]
    results = aggregate_results(files)
    create_dashboard(results, Path(args.output))


if __name__ == "__main__":
    main()
