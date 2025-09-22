"""
Consensus Scalability Dashboard

Generates interactive plots (Accuracy, Runtime, Tokens vs Node Count)
for consensus tasks only.

Run:
    python -m multi_agent.topology.visualization.consensus_scalability results/ontology/*.json --output consensus_scalability.html
"""

import argparse
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_result(filepath: Path):
    """Load a JSON result file safely."""
    with open(filepath, "r") as f:
        return json.load(f)


def extract_consensus_results(files):
    """
    Extract results only for consensus tasks.
    """
    nodes, scores, runtimes, tokens = [], [], [], []

    for f in files:
        try:
            data = load_result(f)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        if data.get("task") != "consensus":
            continue

        num_nodes = data.get("num_nodes")
        score = data.get("score")
        runtime = data.get("runtime_seconds")

        # token_summary is nested
        token_summary = data.get("token_summary", {})
        token_count = token_summary.get("total_tokens")

        if num_nodes is not None:
            nodes.append(num_nodes)
            scores.append(score)
            runtimes.append(runtime)
            tokens.append(token_count)

    # sort by node count
    order = sorted(range(len(nodes)), key=lambda i: nodes[i])
    nodes = [nodes[i] for i in order]
    scores = [scores[i] for i in order]
    runtimes = [runtimes[i] for i in order]
    tokens = [tokens[i] for i in order]

    return dict(nodes=nodes, scores=scores, runtimes=runtimes, tokens=tokens)


def create_dashboard(results, output_path: Path):
    """Create interactive HTML scalability plots for consensus."""
    nodes = results["nodes"]
    scores = results["scores"]
    runtimes = results["runtimes"]
    tokens = results["tokens"]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Accuracy", "Runtime (s)", "Total Tokens")
    )

    # Accuracy
    fig.add_trace(go.Scatter(x=nodes, y=scores, mode="lines+markers", name="Accuracy"), row=1, col=1)

    # Runtime
    fig.add_trace(go.Scatter(x=nodes, y=runtimes, mode="lines+markers", name="Runtime", line=dict(color="orange")), row=1, col=2)

    # Tokens
    fig.add_trace(go.Scatter(x=nodes, y=tokens, mode="lines+markers", name="Tokens", line=dict(color="green")), row=1, col=3)

    fig.update_layout(
        title_text="Consensus Scalability",
        showlegend=False,
        height=500,
        width=1400
    )

    fig.write_html(output_path)
    print(f"✅ Consensus scalability dashboard saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate consensus scalability dashboard")
    parser.add_argument("files", nargs="+", help="Result JSON files")
    parser.add_argument("--output", "-o", default="consensus_scalability.html", help="Output HTML file")
    args = parser.parse_args()

    files = [Path(f) for f in args.files]
    results = extract_consensus_results(files)

    if not results["nodes"]:
        print("⚠️ No consensus results found in provided files.")
        return

    create_dashboard(results, Path(args.output))


if __name__ == "__main__":
    main()
