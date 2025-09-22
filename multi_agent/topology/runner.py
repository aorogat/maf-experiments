# """
# python -m multi_agent.topology.runner --task coloring   --model gpt-4o-mini   --graph_models ws   --graph_size 4   --samples_per_graph_model 1   --rounds 4
# ---------
# Main entry point for running multi-agent experiments.
# Now uses graph_builder.py to abstract graph construction.
# """

# import argparse
# import asyncio
# import random
# import time

# import networkx as nx

# from multi_agent.agentsNetOriginalCode import LiteralMessagePassing as lmp
# from multi_agent.topology.tasks import TASKS
# from multi_agent.topology.model_providers import MODEL_PROVIDER
# from multi_agent.topology.results import save_results
# from multi_agent.topology.utils import get_git_commit_hash, determine_rounds
# from multi_agent.topology.graph_builder import get_graph
# from dotenv import load_dotenv
# load_dotenv()

# async def run(args):
#     results = []
#     commit_hash = get_git_commit_hash()
#     random.seed(args.seed)

#     for graph_model in args.graph_models:
#         for i in range(args.start_from_sample, args.samples_per_graph_model):

#             # --- Build graph using graph_builder abstraction ---
#             graph = get_graph(
#                 source="hf",  # could also be "framework" later
#                 graph_model=graph_model,
#                 graph_size=args.graph_size,
#                 num_sample=i,
#             )

#             rounds = determine_rounds(
#                 args.task, graph, i, args.samples_per_graph_model, args.rounds
#             )
#             print(f"[Runner] Selected {rounds} rounds.")

#             task_class = TASKS[args.task]
#             model_provider = MODEL_PROVIDER[args.model]
#             chain_of_thought = not args.disable_chain_of_thought

#             lmp_model: lmp.LiteralMessagePassing = task_class(
#                 graph=graph,
#                 rounds=rounds,
#                 model_name=args.model,
#                 model_provider=model_provider,
#                 chain_of_thought=chain_of_thought,
#             )

#             await lmp_model.bootstrap()
#             try:
#                 t0 = time.time()
#                 answers = await lmp_model.pass_messages()
#                 runtime = time.time() - t0
#                 score = lmp_model.get_score(answers)
#                 successful = True
#                 error_message = None
#             except (ValueError, KeyError) as e:
#                 answers = [None for _ in range(graph.order())]
#                 runtime = None
#                 score = None
#                 successful = False
#                 error_message = repr(e)

#             results.append(
#                 dict(
#                     model=args.model,
#                     task=args.task,
#                     rounds=rounds,
#                     seed=args.seed,
#                     score=score,
#                     runtime=runtime,
#                 )
#             )
#             save_results(
#                 answers=answers,
#                 rounds=rounds,
#                 model_name=lmp_model.model_name,
#                 task=args.task,
#                 score=score,
#                 commit_hash=commit_hash,
#                 graph_generator=graph_model,
#                 graph_index=i,
#                 successful=successful,
#                 error_message=error_message,
#                 chain_of_thought=chain_of_thought,
#                 num_fallbacks=lmp_model.num_fallbacks,
#                 num_failed_json_parsings_after_retry=lmp_model.num_failed_json_parsings_after_retry,
#                 num_failed_answer_parsings_after_retry=lmp_model.num_failed_answer_parsings_after_retry,
#                 runtime=runtime,
#                 transcripts=lmp_model.get_transcripts(),
#                 graph=lmp_model.graph,
#             )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, default="gpt-4o-mini")
#     parser.add_argument("--task", type=str, default="coloring")
#     parser.add_argument("--graph_models", type=str, nargs="+", default=["ws", "ba", "dt"])
#     parser.add_argument("--start_from_sample", type=int, default=0)
#     parser.add_argument("--samples_per_graph_model", type=int, default=3)
#     parser.add_argument("--graph_size", type=int, default=10)
#     parser.add_argument("--rounds", type=int, default=4)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--disable_chain_of_thought", action="store_true")
#     parser.add_argument("--missing_run_file", type=str, default=None)
#     parser.add_argument("--framework", type=str, default="literal", 
#                     choices=["literal", "langgraph", "crewai"])

#     args = parser.parse_args()
#     asyncio.run(run(args))





"""
Main entry point for running multi-agent experiments.
To run:
```
python -m multi_agent.topology.runner \
  --task coloring \
  --model gpt-4o-mini \
  --graph_models ws \
  --graph_size 4 \
  --samples_per_graph_model 1 \
  --rounds 4 \
  --framework literal
```

Supports multiple frameworks (literal, langgraph, crewai, ...).
New frameworks can be added easily by dropping a runner file and
registering it in FRAMEWORK_REGISTRY.
"""

import argparse
import asyncio
import importlib
import random
from dotenv import load_dotenv

from multi_agent.topology.utils import get_git_commit_hash

# Load API keys from .env
load_dotenv()


# -------------------------
# Framework Registry
# -------------------------
FRAMEWORK_REGISTRY = {
    "literal": "multi_agent.topology.frameworks.literal_runner",
    "langgraph": "multi_agent.topology.frameworks.langgraph_runner",
    "crewai": "multi_agent.topology.frameworks.crewai_runner",
    "concordia": "multi_agent.topology.frameworks.concordia_runner",
    # Add new frameworks here later...
}


async def run(args):
    commit_hash = get_git_commit_hash()
    random.seed(args.seed)

    if args.framework not in FRAMEWORK_REGISTRY:
        raise ValueError(f"Unknown framework: {args.framework}")

    module_path = FRAMEWORK_REGISTRY[args.framework]
    runner_module = importlib.import_module(module_path)

    if not hasattr(runner_module, "run_framework"):
        raise AttributeError(f"{module_path} is missing `run_framework(args, commit_hash)`")

    # Dispatch to the chosen framework
    await runner_module.run_framework(args, commit_hash)


# -------------------------
# CLI Entry Point
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--task", type=str, default="coloring")
    parser.add_argument("--graph_models", type=str, nargs="+", default=["ws", "ba", "dt"])
    parser.add_argument("--start_from_sample", type=int, default=0)
    parser.add_argument("--samples_per_graph_model", type=int, default=3)
    parser.add_argument("--graph_size", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_chain_of_thought", action="store_true")
    parser.add_argument("--missing_run_file", type=str, default=None)
    parser.add_argument(
        "--framework",
        type=str,
        default="literal",
        choices=list(FRAMEWORK_REGISTRY.keys()),
        help="Framework to run the experiment with",
    )

    args = parser.parse_args()
    asyncio.run(run(args))
