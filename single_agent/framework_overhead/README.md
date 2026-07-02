# Framework Overhead Experiment

This experiment measures the **baseline orchestration overhead** introduced by different multi-agent frameworks (CrewAI, LangGraph, Concordia, GABM Skeleton, etc.) compared to a direct LLM call.

The setup uses a trivial prompt (`"What is 2+2?"`) and disables all optional features (no memory, no planning, no tools). This isolates **framework cost** from model behavior.

---

## Structure

```
single_agent/framework_overhead/
├── direct_llm.py           # Direct LLM baseline runner
├── crewai_runner.py        # CrewAI runner (1 agent, 1 task)
├── langgraph_runner.py     # LangGraph runner (single-node graph)
├── concordia_runner.py     # Concordia runner (minimal simulation)
├── gabm_skeleton_runner.py # Minimal GABM skeleton (env-mediated)
├── autogen_runner.py       # AutoGen agentchat 0.7.5 runner
├── instrumentation.py      # Shared API-boundary token/time capture
└── run_overhead.py         # Master script: runs all frameworks & saves results
```

---

## How to Run

1. Install dependencies (OpenAI API key required for `openai/gpt-4o-mini` runs):
   ```bash
   pip install -r requirements.txt
   ```

2. Run the full overhead experiment:
   ```bash
   python -m single_agent.framework_overhead.run_overhead
   ```

3. Results are saved as JSON in:
   ```
   results/framework_overhead/framework_overhead_50_TRIALS.json
   ```

---

## Metrics

### Throughput metrics (existing)

- **Latency (p50 / p95)** — end-to-end wall time per trial (ms)
- **Throughput (req/s)** — trials per second at configured concurrency
- **Output size (chars)** — response character counts

### API-boundary decomposition (R1.O2)

For each framework, the harness also reports per-trial capture at the OpenAI SDK / litellm boundary, aggregated as means:

| Field | Meaning |
|-------|---------|
| `llm_calls_mean` | Number of LLM API calls observed |
| `input_tokens_mean` | Sum of input/prompt tokens |
| `output_tokens_mean` | Sum of output/completion tokens |
| `llm_time_s_mean` | Wall-clock time inside LLM calls (summed per trial) |
| `framework_residual_s_mean` | `total_time - llm_time` (framework/orchestration residual) |
| `llm_time_frac` | Fraction of total time attributable to LLM calls |
| `max_concurrency_observed` | Peak concurrent in-flight LLM calls in any trial |
| `overlap_trials` | Trials where concurrent calls broke the serial subset assumption |
| `overlap_warning` | `true` if any trial had overlapping LLM calls |
| `residual_valid_frac` | Fraction of trials with a valid serial residual |

**Instrumentation approach:** `instrumentation.py` patches the OpenAI SDK (`chat.completions.create`, `responses.create`, sync and async) and litellm's OpenAI request helpers. Patches compose with Concordia's existing arg-stripping patch. No per-runner edits required.

**Overlap guard:** When `max_concurrency > 1` or summed LLM time exceeds total wall time, the harness WARNs and marks the trial's residual as invalid (async frameworks may issue concurrent calls). Residual falls back to the union LLM span (`wall_llm_span_s`) rather than being silently floored to zero.

---

## Example output

```
Direct LLM: p50=773 ms | LLM calls=1.0 | tokens=30/1 | llm_time=773ms | residual=0ms
AutoGen:    p50=673 ms | LLM calls=1.0 | tokens=26/8 | llm_time=669ms | residual=4ms
CrewAI:     p50=978 ms | LLM calls=2.0 | tokens=168/23 | overlap_warning=true
Concordia:  p50=34334 ms | LLM calls=3.0 | tokens=3145/57 | residual=...
```

---

## Notes

- This experiment isolates framework **overhead only**; all advanced features are disabled.
- Default: 50 trials, concurrency=4.
- **OpenHands** runs via `openhands_runner.py` with tools disabled (`tools=[]`, `include_default_tools=[]`, `cli_mode=True`). Expect higher prompt-token overhead from the OpenHands system prompt; token capture uses the shared API-boundary instrumentation (litellm → OpenAI path).
- `openai-agents` is pinned to `0.0.19` for compatibility with `openai==1.102.0`.
