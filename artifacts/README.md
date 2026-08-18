# Artifacts

## Pinned experiment versions

Canonical pin file for the paper footnote (“Versions are pinned in the artifact”):

**[`experiment_versions.json`](experiment_versions.json)**

### AutoGen

| Distribution | Version | Role |
|---|---|---|
| `autogen-agentchat` | 0.7.5 | Evaluated AutoGen API (all experiments) |
| `autogen-core` | 0.7.5 | Dependency of agentchat |
| `autogen-ext` | 0.7.5 | Model clients / Chroma memory |

Install: `pip install -r requirements-autogen.lock`

**Do not install AG2** (`ag2` / `pyautogen`, import name `autogen`). MASBench does not import it. If both stacks are present, reviewers cannot tell which line was evaluated.

## Running-example full traces (Footnote 2)

**[`running_example_full_traces/`](running_example_full_traces/)** — full LangGraph / CrewAI / GABM skeleton traces (prompts, per-call tokens, inter-step state, tool-call sites). Complements body tables `tab:paradigm-traces` and `tab:tool-invocation`. Include `full_traces.tex` in the supplementary materials.

## Planning interface appendix (Footnote 6 + R3W3)

**[`planning_interface_appendix/`](planning_interface_appendix/)** — annotated NoPlan / Crew-Plan / Direct-LLM-Plan on one MATH question (Crew schema violation noted; no CrewAI plan callback), plus the determinism table.
