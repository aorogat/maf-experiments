"""
Run a single-topology (small-world) experiment using the original
AgentsNet LiteralMessagePassing tasks, but with the model fixed to gpt-4o-mini.

Usage:
  python -m multi_agent.topology.langgraph_smallworld_test --task coloring --n 8 --rounds 4
  # ensure OPENAI_API_KEY is set in your environment or .env
"""
import argparse, asyncio, datetime, json, os, time, subprocess, re   # <<< added re
import networkx as nx
from networkx.readwrite import json_graph

# import the original AgentsNet code you placed here
from multi_agent.agentsNetOriginalCode import LiteralMessagePassing as lmp
import os
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import networkx as nx

load_dotenv()  # this loads variables from .env into os.environ


# <<< added monkey-patch for json.loads >>>
import json as _json
import re

def safe_json_loads(s):
    if not isinstance(s, str):
        return s
    cleaned = re.sub(r"```(?:json)?|```", "", s).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return _json.loads(cleaned)

# patch only LiteralMessagePassing
from multi_agent import agentsNetOriginalCode as anc
anc.LiteralMessagePassing.json_loads = staticmethod(safe_json_loads)
# --- End patch ---

def draw_final_coloring(G: nx.Graph, groups, title: str, save_path: str):
    """Draw final assignment. Node shapes = groups, dashed edges = conflicts."""
    pos = nx.spring_layout(G, seed=42)

    uniq = [g for g in sorted(set(groups)) if g is not None]
    by_group = {g: [n for n in G.nodes() if groups[n] == g] for g in uniq}
    unknown = [n for n in G.nodes() if groups[n] is None]

    conflicts = [(u, v) for u, v in G.edges() if groups[u] is not None and groups[u] == groups[v]]
    ok = [e for e in G.edges() if e not in conflicts]

    shapes = ["o", "s", "^", "D", "P", "X"]
    for i, g in enumerate(uniq):
        nx.draw_networkx_nodes(G, pos, nodelist=by_group[g], node_shape=shapes[i % len(shapes)], node_size=650)
    if unknown:
        nx.draw_networkx_nodes(G, pos, nodelist=unknown, node_shape="o", node_size=450)

    if ok:
        nx.draw_networkx_edges(G, pos, edgelist=ok, width=1.0, style="solid")
    if conflicts:
        nx.draw_networkx_edges(G, pos, edgelist=conflicts, width=3.0, style="dashed")

    labels = {n: f"{G.nodes[n].get('name', n)}\n({groups[n] if groups[n] else '?'})" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.title(title)
    plt.axis("off")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


TASKS = {
    "matching": lmp.Matching,
    "consensus": lmp.Consensus,
    "coloring": lmp.Coloring,
    "leader_election": lmp.LeaderElection,
    "vertex_cover": lmp.VertexCover,
}

def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
    except Exception:
        return "None"

def determine_rounds(task: str, graph: nx.Graph, default_rounds: int) -> int:
    if task in ["consensus", "leader_election"] or graph.number_of_nodes() > 16:
        return 2 * nx.diameter(graph) + 1
    return default_rounds

def save_results(
    *, answers, transcripts, graph, rounds, model_name, task, score,
    commit_hash, graph_generator, successful, error_message,
    chain_of_thought, num_fallbacks, num_failed_json_parsings_after_retry,
    num_failed_answer_parsings_after_retry
):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{task}_results_{ts}_rounds{rounds}_{model_name.split('/')[-1]}_nodes{graph.number_of_nodes()}.json"

    # >>> save under results/<topology>/<filename>
    outdir = os.path.join("results", str(graph_generator))  # e.g., "results/ws"
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "answers": answers,
            "transcripts": transcripts,
            "graph": json_graph.node_link_data(graph),
            "num_nodes": graph.number_of_nodes(),
            "diameter": nx.diameter(graph),
            "max_degree": max(dict(graph.degree()).values()),
            "rounds": rounds,
            "model_name": model_name,
            "task": task,
            "score": score,
            "commit_hash": commit_hash,
            "graph_generator": graph_generator,
            "graph_index": 0,
            "successful": successful,
            "error_message": error_message,
            "chain_of_thought": chain_of_thought,
            "num_fallbacks": num_fallbacks,
            "num_failed_json_parsings_after_retry": num_failed_json_parsings_after_retry,
            "num_failed_answer_parsings_after_retry": num_failed_answer_parsings_after_retry,
        }, f, indent=2)

    print(f"[✓] Saved: {path}")

async def run(args):
    # ---- 1) Build one topology: small-world (Watts–Strogatz)
    graph = nx.watts_strogatz_graph(n=args.n, k=args.k, p=args.p, seed=args.seed)
    print(f"[info] small-world graph built: n={args.n}, k={args.k}, p={args.p}")

    # Ensure each node has a "name" attribute (AgentsNet expects it)
    try:
        from multi_agent.agentsNetOriginalCode.utils import names as _agentsnet_names
        name_list = _agentsnet_names(len(graph))
    except Exception:
        name_list = [f"Agent_{i}" for i in range(graph.number_of_nodes())]

    for i in graph.nodes:
        if "name" not in graph.nodes[i]:
            graph.nodes[i]["name"] = name_list[i] if i < len(name_list) else f"Agent_{i}"

    # ---- 2) Fix the model + provider
    model_name = "gpt-4o-mini"
    model_provider = "openai"
    chain_of_thought = not args.disable_chain_of_thought

    # ---- 3) Determine rounds
    rounds = determine_rounds(args.task, graph, args.rounds)
    print(f"[info] rounds selected: {rounds}")

    # ---- 4) Create and run the original task
    task_cls = TASKS[args.task]
    lmp_model: lmp.LiteralMessagePassing = task_cls(
        graph=graph,
        rounds=rounds,
        model_name=model_name,
        model_provider=model_provider,
        chain_of_thought=chain_of_thought,
    )

    # <<< added stricter JSON-only instruction >>>
    extra_instruction = (
        "IMPORTANT: Only output a JSON object with neighbor names as keys and messages as values. "
        "Do not include markdown, code fences, or explanations outside the JSON."
    )
    if hasattr(lmp_model, "system_message"):
        lmp_model.system_message += "\n" + extra_instruction
    # <<< end >>>

    await lmp_model.bootstrap()
    t0 = time.time()
    try:
        answers = await lmp_model.pass_messages()
        score = lmp_model.get_score(answers)
        successful = True
        error_message = None
    except (ValueError, KeyError) as e:
        answers = [None for _ in range(graph.order())]
        score = None
        successful = False
        error_message = repr(e)
    runtime = time.time() - t0

    print(f"[result] success={successful} score={score} runtime={runtime:.2f}s")

    # Draw final coloring snapshot
    final_png = os.path.join("results", "ws", f"{args.task}_final.png")
    draw_final_coloring(graph, answers, title=f"{args.task.capitalize()} — Final", save_path=final_png)
    print(f"[✓] Saved final plot: {final_png}")

    # ---- 5) Persist results
    save_results(
        answers=answers,
        transcripts=lmp_model.get_transcripts(),
        graph=graph,
        rounds=rounds,
        model_name=model_name,
        task=args.task,
        score=score,
        commit_hash=get_git_commit_hash(),
        graph_generator="ws",
        successful=successful,
        error_message=error_message,
        chain_of_thought=chain_of_thought,
        num_fallbacks=lmp_model.num_fallbacks,
        num_failed_json_parsings_after_retry=lmp_model.num_failed_json_parsings_after_retry,
        num_failed_answer_parsings_after_retry=lmp_model.num_failed_answer_parsings_after_retry,
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="coloring",
                    choices=["matching","consensus","coloring","leader_election","vertex_cover"])
    ap.add_argument("--n", type=int, default=8, help="number of agents")
    ap.add_argument("--k", type=int, default=3, help="ring neighbors in WS graph")
    ap.add_argument("--p", type=float, default=0.2, help="rewiring prob in WS graph")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--disable_chain_of_thought", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args))
