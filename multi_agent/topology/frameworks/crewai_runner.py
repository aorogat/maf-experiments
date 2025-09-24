"""
CrewAI Framework Runner
-----------------------
This runner executes multi-agent experiments in the CrewAI-style setup.

Why this is CrewAI:
- CrewAI supports sequential and hierarchical orchestration.
- Sequential = agents in a chain (path graph).
- Hierarchical = agents in a balanced 4-ary tree.
- We simulate this by loading the HuggingFace AgentsNet graphs for nodes,
  then rewiring edges into sequential or hierarchical structures.

Example usage:
    python -m multi_agent.topology.frameworks.crewai_runner \
        --task consensus \
        --model gpt-4o-mini \
        --graph_size 4 \
        --rounds 8 \
        --framework sequential

    python -m multi_agent.topology.frameworks.crewai_runner \
        --task leader_election \
        --model gpt-4o-mini \
        --graph_size 16 \
        --rounds 6 \
        --framework hierarchical
"""

import time
import random

from multi_agent.agentsNetOriginalCode import LiteralMessagePassing as lmp
from multi_agent.topology.tasks import TASKS
from multi_agent.topology.model_providers import MODEL_PROVIDER
from multi_agent.topology.results import save_results
from multi_agent.topology.utils import determine_rounds
from multi_agent.topology.graph_builder import get_graph

from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

async def run_framework(args, commit_hash):
    results = []
    random.seed(args.seed)

    for i in range(args.start_from_sample, args.samples_per_graph_model):

        # --- Build CrewAI graph (sequential or hierarchical) from HF ---
        if args.framework == "sequential":
            graph = get_graph(
                source="hf_sequential",
                graph_model=args.graph_models[0],
                graph_size=args.graph_size,
                num_sample=i,
            )
        elif args.framework == "hierarchical":
            graph = get_graph(
                source="hf_hierarchical",
                graph_model=args.graph_models[0],
                graph_size=args.graph_size,
                num_sample=i,
            )
        else:
            raise ValueError(f"Unknown CrewAI framework: {args.framework}")

        rounds = determine_rounds(
            args.task, graph, i, args.samples_per_graph_model, args.rounds
        )
        print(f"[CrewAIRunner:{args.framework}] Selected {rounds} rounds.")

        # --- Task and Model setup ---
        task_class = TASKS[args.task]
        model_provider = MODEL_PROVIDER[args.model]
        chain_of_thought = not args.disable_chain_of_thought

        lmp_model: lmp.LiteralMessagePassing = task_class(
            graph=graph,
            rounds=rounds,
            model_name=args.model,
            model_provider=model_provider,
            chain_of_thought=chain_of_thought,
        )

        await lmp_model.bootstrap()

        try:
            t0 = time.time()
            answers = await lmp_model.pass_messages()
            runtime = time.time() - t0
            score = lmp_model.get_score(answers)
            successful = True
            error_message = None
        except (ValueError, KeyError) as e:
            answers = [None for _ in range(graph.order())]
            runtime = None
            score = None
            successful = False
            error_message = repr(e)

        results.append(
            dict(
                model=args.model,
                task=args.task,
                rounds=rounds,
                seed=args.seed,
                score=score,
                runtime=runtime,
            )
        )

        save_results(
            answers=answers,
            rounds=rounds,
            framework=f"crewai-{args.framework}",
            model_name=lmp_model.model_name,
            task=args.task,
            score=score,
            commit_hash=commit_hash,
            graph_generator=args.framework,
            graph_index=i,
            successful=successful,
            error_message=error_message,
            chain_of_thought=chain_of_thought,
            num_fallbacks=lmp_model.num_fallbacks,
            num_failed_json_parsings_after_retry=lmp_model.num_failed_json_parsings_after_retry,
            num_failed_answer_parsings_after_retry=lmp_model.num_failed_answer_parsings_after_retry,
            runtime=runtime,
            transcripts=lmp_model.get_transcripts(),
            graph=lmp_model.graph,
        )

    return results


if __name__ == "__main__":
    import argparse
    import asyncio
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--task", type=str, default="consensus")
    parser.add_argument("--graph_models", type=str, nargs="+", default=["ws"])
    parser.add_argument("--start_from_sample", type=int, default=0)
    parser.add_argument("--samples_per_graph_model", type=int, default=1)
    parser.add_argument("--graph_size", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--framework", type=str, choices=["sequential", "hierarchical"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_chain_of_thought", action="store_true")
    args = parser.parse_args()

    # get commit hash (like in newMain.py)
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('utf-8').strip()
    except Exception:
        commit_hash = "None"

    asyncio.run(run_framework(args, commit_hash))
