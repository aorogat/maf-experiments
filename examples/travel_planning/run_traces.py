"""Entry point: runs all three paradigms, writes traces + comparison tables."""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from examples.travel_planning.crewai_impl import run_crewai
from examples.travel_planning.gabm_impl import run_gabm
from examples.travel_planning.langgraph_impl import run_langgraph
from examples.travel_planning.task import TripRequest
from examples.travel_planning.trace import side_by_side, tool_invocation_comparison

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Travel-planning running example (Fig 1)")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="LLM model identifier")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for trace outputs")
    args = parser.parse_args(argv)

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY required for live runs.", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    request = TripRequest()

    def _print_result(label: str, trace) -> None:
        s = trace.summary()
        line = f"  {label}: {s['final_itinerary']}"
        if not s["plan_complete"]:
            line += " [INCOMPLETE — probabilistic tool skip; not a harness bug]"
        print(line)

    print(f"Running travel-planning traces with model={args.model} ...")

    trace_lg_node = run_langgraph(request=request, tool_mode="node", model=args.model)
    trace_lg_node.save_json(os.path.join(args.output_dir, "trace_langgraph_node.json"))
    _print_result("LangGraph (node)", trace_lg_node)

    trace_lg_agent = run_langgraph(request=request, tool_mode="agent", model=args.model)
    trace_lg_agent.save_json(os.path.join(args.output_dir, "trace_langgraph_agent.json"))
    _print_result("LangGraph (agent)", trace_lg_agent)

    trace_crew = run_crewai(request=request, model=args.model)
    trace_crew.save_json(os.path.join(args.output_dir, "trace_crewai.json"))
    _print_result("CrewAI", trace_crew)

    trace_gabm = run_gabm(request=request, model=args.model)
    trace_gabm.save_json(os.path.join(args.output_dir, "trace_gabm.json"))
    _print_result("GABM", trace_gabm)

    traces = {
        "langgraph_node": trace_lg_node,
        "langgraph_agent": trace_lg_agent,
        "crewai": trace_crew,
        "gabm": trace_gabm,
    }

    md, latex = side_by_side(traces)
    with open(os.path.join(args.output_dir, "side_by_side.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    with open(os.path.join(args.output_dir, "side_by_side.tex"), "w", encoding="utf-8") as f:
        f.write(latex + "\n")

    comparison_tex = tool_invocation_comparison(traces)
    with open(os.path.join(args.output_dir, "tool_invocation_comparison.tex"), "w", encoding="utf-8") as f:
        f.write(comparison_tex + "\n")

    print(f"Wrote traces and tables to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
