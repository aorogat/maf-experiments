# GABM Skeleton

A minimal Generative-Agent-Based-Modeling (GABM) framework that implements only the essential environment-mediated coordination loop. It is a deliberately stripped-down counterpart to the Concordia integration: same paradigm (agents coordinate through a shared environment mediated by a Game Master), but with none of Concordia's extra machinery.

## What it implements

- **Shared environment state** (`E`) with grounded variables
- **Game Master (GM)** that filters observations and applies agent actions via a transition function
- **Observe → act cycle**: each round, every agent receives a filtered observation, makes exactly one LLM call, and the GM applies the action to `E`
- **Round loop** with stopping on `max_rounds` or task-complete predicate
- **Instrumentation**: `llm_calls`, token counts, `llm_time_s`, `total_time_s`, `framework_residual_s`, `output_chars`

## What it intentionally omits

To keep measured overhead reflective of the environment-mediation pattern itself:

- No narrative/transcript text generation
- No natural-language scene/world descriptions
- No HTML or rich logging beyond structured metrics
- No personas, narrative memories, relationships, or social-simulation constructs
- No persistent multi-episode state
- No extra LLM calls beyond one action call per agent per round

**Mechanical rule:** with N agents over R rounds, total LLM calls = `N * R`.

## Layout

```
frameworks/gabm_skeleton/
├── environment.py      # E: shared state + transition
├── mediator.py         # GM: observation filtering + action application
├── agent.py            # SkeletonAgent: observe -> one LLM call -> action
├── runner.py           # GABMSkeletonRunner (public entry point)
├── metrics.py          # Metrics, RoundTrace, RunResult
├── llm_client.py       # Instrumented OpenAI client + test stub
└── tasks/
    ├── trivial.py      # "What is 2+2?" single-agent workload
    └── travel.py       # flight + hotel + budget agents
```

## Run commands

From the MASBench repo root:

```bash
# Smoke test (requires OPENAI_API_KEY)
python -m frameworks.gabm_skeleton.runner --task trivial --model openai/gpt-4o-mini
python -m frameworks.gabm_skeleton.runner --task travel --model openai/gpt-4o-mini

# Unit + integration tests (stub LLM, no network)
python -m unittest discover -s frameworks/gabm_skeleton/tests -v

# Overhead harness (registered as "GABM Skeleton")
python -m single_agent.framework_overhead.run_overhead
```

## Public API

```python
from frameworks.gabm_skeleton.runner import GABMSkeletonRunner
from frameworks.gabm_skeleton.tasks.trivial import TrivialTask
from frameworks.gabm_skeleton.tasks.travel import TravelTask

# Overhead harness compatibility: run(str) -> (answer, latency_ms)
runner = GABMSkeletonRunner(model="openai/gpt-4o-mini")
answer, latency_ms = runner.run("What is 2+2?")
metrics = runner.last_result.metrics

# Full task API: run(Task) -> RunResult
result = runner.run(TravelTask(budget=500, max_rounds=5))
print(result.trace)
print(result.trace_table())
```

## Frozen interfaces (for downstream tasks)

These shapes are stable contracts for later overhead, trace, and tool-comparison tasks:

- `RunResult`: `answer`, `final_state`, `metrics`, `trace`, `trace_table()`
- `RoundTrace`: `prompts_issued`, `llm_calls`, `state_snapshot`, `communication`, `tool_invocations`
- `tool_invocations[*].deterministic`: `bool` flag for deterministic vs probabilistic tools
- `Metrics`: `llm_calls`, `input_tokens`, `output_tokens`, `llm_time_s`, `total_time_s`, `framework_residual_s`, `output_chars`

## Measurement notes

- **Returned latency** covers the full `run()` span (environment setup + round loop), which is what the overhead harness records when it prefers runner-reported latency over its own wrapper timer.
- **Fresh `Metrics` per `run()`** — the 50-trial harness reuses one runner instance; metrics never accumulate across trials.
- **Cached OpenAI client** — `InstrumentedOpenAIClient` is created once per runner (like `DirectLLMRunner`), with per-trial metrics swapped in.
- **Trivial task only for overhead** — `llm_max_tokens=16` matches `direct_llm.py`; travel task timings are for trace/tool tasks, not overhead rows.
- **Token capture** — verified by `test_llm_client.TestInstrumentedOpenAIClientLive` and the smoke CLI against a live model.

```bash
# Verify real token capture (requires OPENAI_API_KEY):
python -m frameworks.gabm_skeleton.runner --task trivial --model openai/gpt-4o-mini
# EXPECT: latency ~hundreds of ms, input_tokens>0, output_tokens>0
```

## Registration

Registered in `single_agent/framework_overhead/run_overhead.py` as `"GABM Skeleton"`.
