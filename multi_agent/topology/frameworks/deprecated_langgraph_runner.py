"""
langgraph_runner.py
-------------------
Framework runner for experiments using LangGraph.

Reuses HuggingFace graphs but executes message passing
with LangGraph agents instead of LiteralMessagePassing.
Each agent has its own inbox/outbox like the original code.
"""

import time
import random
import networkx as nx
from typing import Dict, List
from pydantic import BaseModel

from langgraph.graph import StateGraph

from multi_agent.topology.tasks import TASKS
from multi_agent.topology.results import save_results
from multi_agent.topology.utils import determine_rounds
from multi_agent.topology.graph_builder import get_graph
from multi_agent.topology.model_providers import MODEL_PROVIDER


# -------------------------
# Define custom state schema
# -------------------------
class InboxesState(BaseModel):
    inboxes: Dict[str, List[str]]


async def run_framework(args, commit_hash):
    print("[LangGraphRunner] Starting run...")
    random.seed(args.seed)

    for graph_model in args.graph_models:
        for i in range(args.start_from_sample, args.samples_per_graph_model):

            # --- Build graph from HF ---
            graph = get_graph(
                source="hf",
                graph_model=graph_model,
                graph_size=args.graph_size,
                num_sample=i,
            )
            print(f"[LangGraphRunner] Loaded HF graph {graph_model}_{args.graph_size}_{i}")

            rounds = determine_rounds(
                args.task, graph, i, args.samples_per_graph_model, args.rounds
            )
            print(f"[LangGraphRunner] Selected {rounds} rounds")

            # --- Get task class ---
            task_class = TASKS[args.task]
            task = task_class(
                graph=graph,
                rounds=rounds,
                model_name=args.model,
                model_provider=MODEL_PROVIDER[args.model],
                chain_of_thought=not args.disable_chain_of_thought,
            )

            # --- Inject agent_step if missing ---
            if not hasattr(task, "agent_step"):
                import types

                def agent_step(self, node, inbox):
                    return f"[Stub] Agent {node} processed {len(inbox)} messages"

                task.agent_step = types.MethodType(agent_step, task)

            # --- Build LangGraph program ---
            builder = StateGraph(InboxesState)
            transcripts = {}

            for node in graph.nodes():

                async def agent_fn(state: InboxesState, node=node):
                    inbox = state.inboxes.get(str(node), [])
                    print(f"[LangGraphRunner] Agent {node} received {len(inbox)} messages")

                    reply = task.agent_step(node, inbox)

                    transcripts.setdefault(node, []).append(
                        {"in": inbox, "out": reply}
                    )

                    # Deliver reply only to neighbors
                    new_inboxes = {n: msgs.copy() for n, msgs in state.inboxes.items()}
                    for neighbor in graph.neighbors(node):
                        new_inboxes[str(neighbor)].append(f"{node}: {reply}")

                    return InboxesState(inboxes=new_inboxes)

                builder.add_node(str(node), agent_fn)

            # Arbitrary entrypoint
            entry = str(list(graph.nodes())[0])
            builder.set_entry_point(entry)

            compiled = builder.compile()
            print(f"[LangGraphRunner] Running graph with {len(graph.nodes())} agents")

            # --- Run message passing ---
            try:
                t0 = time.time()

                # Initial inbox state
                state = InboxesState(
                    inboxes={str(n): (["start"] if str(n) == entry else []) for n in graph.nodes()}
                )

                for r in range(rounds):
                    print(f"[LangGraphRunner] --- Round {r+1}/{rounds} ---")
                    for node in graph.nodes():
                        state = await compiled.ainvoke(
                            state,
                            config={"configurable": {"thread_id": f"round{r}_node{node}"}}
                        )

                runtime = time.time() - t0

                answers = task.get_answers()
                score = task.get_score(answers)
                successful = True
                error_message = None
            except Exception as e:
                print(f"[LangGraphRunner] Error during execution: {e}")
                answers = [None for _ in range(graph.order())]
                runtime = None
                score = None
                successful = False
                error_message = repr(e)

            # --- Save results ---
            save_results(
                answers=answers,
                rounds=rounds,
                model_name=args.model,
                task=args.task,
                score=score,
                commit_hash=commit_hash,
                graph_generator=graph_model,
                graph_index=i,
                successful=successful,
                error_message=error_message,
                chain_of_thought=not args.disable_chain_of_thought,
                num_fallbacks=getattr(task, "num_fallbacks", 0),
                num_failed_json_parsings_after_retry=getattr(task, "num_failed_json_parsings_after_retry", 0),
                num_failed_answer_parsings_after_retry=getattr(task, "num_failed_answer_parsings_after_retry", 0),
                runtime=runtime,
                transcripts=transcripts,
                graph=graph,
            )
            print(f"[LangGraphRunner] Finished sample {i} for {graph_model}")

    print("[LangGraphRunner] Run complete ✅")
