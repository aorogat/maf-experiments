"""
results.py
----------
Helper functions for saving experiment results and transcripts.
"""

import os
import json
import datetime
import networkx as nx
from networkx.readwrite import json_graph


def summarize_tokens(transcripts):
    """
    Summarize token usage across all agents from transcripts.
    Returns a dict with total and per-agent token usage.
    """
    summary = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "per_agent": {}
    }

    for agent, messages in transcripts.items():
        agent_summary = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for msg in messages:
            usage = (
                msg.get("data", {})
                .get("response_metadata", {})
                .get("token_usage", None)
            )
            if usage:
                in_tokens = usage.get("prompt_tokens", 0)
                out_tokens = usage.get("completion_tokens", 0)
                total = usage.get("total_tokens", in_tokens + out_tokens)

                agent_summary["input_tokens"] += in_tokens
                agent_summary["output_tokens"] += out_tokens
                agent_summary["total_tokens"] += total

                summary["total_input_tokens"] += in_tokens
                summary["total_output_tokens"] += out_tokens
                summary["total_tokens"] += total

        summary["per_agent"][agent] = agent_summary

    return summary


def save_results(
    answers,
    framework,
    rounds,
    model_name,
    task,
    score,
    commit_hash,
    graph_generator,
    graph_index,
    successful,
    error_message,
    chain_of_thought,
    num_fallbacks,
    num_failed_json_parsings_after_retry,
    num_failed_answer_parsings_after_retry,
    runtime,
    transcripts,
    graph,
):
    """
    Saves experiment results and transcripts into a timestamped JSON file.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"{task}_results_{timestamp}_rounds{rounds}_"
        f"{model_name.split('/')[-1]}_nodes{len(graph.nodes())}.json"
    )

    output_dir = "results/ontology"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    # 🔹 Add token summary here
    token_summary = summarize_tokens(transcripts)

    with open(filepath, "w") as f:
        json.dump(
            {
                "answers": answers,
                "num_nodes": len(graph.nodes()),
                "framework": framework,
                "diameter": nx.diameter(graph),
                "max_degree": max(dict(graph.degree()).values()),
                "rounds": rounds,
                "model_name": model_name,
                "task": task,
                "score": score,
                "runtime_seconds": runtime,
                "commit_hash": commit_hash,
                "graph_generator": graph_generator,
                "graph_index": graph_index,
                "successful": successful,
                "error_message": error_message,
                "chain_of_thought": chain_of_thought,
                "num_fallbacks": num_fallbacks,
                "num_failed_json_parsings_after_retry": num_failed_json_parsings_after_retry,
                "num_failed_answer_parsings_after_retry": num_failed_answer_parsings_after_retry,
                "token_summary": token_summary,   # ✅ new summary
                "transcripts": transcripts,
                "graph": json_graph.node_link_data(graph),
            },
            f,
            indent=4,
        )

    print(f"✅ Results saved to {filepath}")
