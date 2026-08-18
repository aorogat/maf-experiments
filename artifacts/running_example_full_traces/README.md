# Running-example full traces (Footnote 2)

This directory is the supplementary artifact promised by Footnote 2.

| File | Role |
|------|------|
| `full_traces.tex` | Appendix-ready LaTeX (prompts, tokens, state, tool sites) |
| `full_traces.md` | Human-readable mirror |
| `trace_langgraph.json` | Frozen LangGraph graph-node trace |
| `trace_crewai.json` | Frozen CrewAI agent-bound trace |
| `trace_gabm.json` | Frozen GABM skeleton env-mediated trace |

Complements body tables `tab:paradigm-traces` and `tab:tool-invocation`.
Totals match Table 4: LangGraph 4 / 138/45; CrewAI 7 / 2677/463; GABM 4 / 296/204.

Canonical generators live under `examples/travel_planning/`.
