# Travel-Planning Running Example (Figure 1)

Illustrative artifact for the MASBench paper taxonomy — **not a benchmarked experiment**.
Produces structured execution traces and a deterministic-vs-probabilistic tool-invocation
comparison table for the supplement.

## What this is

The same fixed trip-planning task (Planner / Flight / Hotel / Budget agents; Web + Cal tools)
instantiated under three paradigms:

| Paradigm | Control flow | Tool invocation |
|----------|--------------|-----------------|
| **LangGraph** | Directed graph (Planner → Flight ∥ Hotel → Budget) | Graph tool nodes (deterministic) + agent-bound variant (probabilistic) |
| **CrewAI** | Role-conditioned delegation | Agent-bound (LLM decides) |
| **GABM skeleton** | Environment-mediated (Game Master) | Environment-executed |

**Concordia is intentionally absent.** It is simulation-only and cannot run this
task-oriented workload; that absence is a finding documented in the paper (W2-W1), not a gap
in this code.

## Run

From the repo root (requires `OPENAI_API_KEY` in `.env`):

```bash
python -m examples.travel_planning.run_traces --model openai/gpt-4o-mini
```

Outputs land in `examples/travel_planning/outputs/`:

- `trace_langgraph_node.json` — deterministic graph tool nodes
- `trace_langgraph_agent.json` — probabilistic agent-bound tools
- `trace_crewai.json`
- `trace_gabm.json`
- `side_by_side.md` / `side_by_side.tex` — cross-paradigm trace comparison (W2-W1)
- `tool_invocation_comparison.tex` — deterministic vs probabilistic (W2-W2)

## Tests

```bash
python -m unittest discover -s examples/travel_planning/tests -v
```

Tests use stub LLMs (no network).

## Implementation notes

- **Shared instrumented LLM client:** all paradigms record `llm_calls`, `input_tokens`,
  `output_tokens` through `frameworks/gabm_skeleton/llm_client.py` (or subclasses in this
  example).
- **LangGraph agent-bound variant** deliberately bypasses `ChatOpenAI.bind_tools` to hold the
  client constant for token parity; it isolates invocation mode, not LangGraph's native
  agent-tool path. The **graph-node variant is faithful** and is what R3.Q3 leans on.
- **LangGraph topology** matches Figure 1(a): Planner fans out to Flight and Hotel in
  parallel (separate graph branches), which join before Budget.
- **Incomplete agent-mode plans are intentional (R3.Q3).** If the LLM probabilistically
  skips `Web` but still calls `Cal`, traces show `plan_complete=False`, `status=incomplete`,
  and `budget_satisfied=False` — not a harness bug.
- **CrewAI** internal planner prompts are not fully retrievable (R1.O3); task descriptions
  are built from current shared state each step. Live runs capture tokens via
  `install_patches()` / `measure()` (same boundary stack as the overhead harness).
