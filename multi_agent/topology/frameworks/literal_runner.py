"""
Literal Framework Runner
------------------------
Runs the original LiteralMessagePassing framework for multi-agent experiments.
"""

import time
import random
import networkx as nx

from multi_agent.agentsNetOriginalCode import LiteralMessagePassing as lmp
from multi_agent.topology.tasks import TASKS
from multi_agent.topology.model_providers import MODEL_PROVIDER
from multi_agent.topology.results import save_results
from multi_agent.topology.utils import determine_rounds
from multi_agent.topology.graph_builder import get_graph


async def run_framework(args, commit_hash):
    """
    Run experiments using the LiteralMessagePassing framework.
    """
    results = []
    random.seed(args.seed)

    for graph_model in args.graph_models:
        for i in range(args.start_from_sample, args.samples_per_graph_model):

            # --- Build graph ---
            graph = get_graph(
                source="hf",  # from HuggingFace dataset
                graph_model=graph_model,
                graph_size=args.graph_size,
                num_sample=i,
            )

            rounds = determine_rounds(
                args.task, graph, i, args.samples_per_graph_model, args.rounds
            )
            print(f"[LiteralRunner] Selected {rounds} rounds.")

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
                model_name=lmp_model.model_name,
                task=args.task,
                score=score,
                commit_hash=commit_hash,
                graph_generator=graph_model,
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
