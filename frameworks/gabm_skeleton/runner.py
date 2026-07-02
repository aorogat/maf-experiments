"""GABMSkeletonRunner: public entry point for the minimal GABM framework."""

from __future__ import annotations

import argparse
import time
from typing import Any, Union

from frameworks.gabm_skeleton.agent import SkeletonAgent
from frameworks.gabm_skeleton.llm_client import InstrumentedOpenAIClient, StubLLMClient
from frameworks.gabm_skeleton.mediator import GameMaster
from frameworks.gabm_skeleton.metrics import Metrics, RoundTrace, RunResult
from frameworks.gabm_skeleton.tasks.trivial import TrivialTask
from frameworks.gabm_skeleton.tasks.travel import TravelTask

QUESTION = "What is 2+2?"


class GABMSkeletonRunner:
    """Minimal environment-mediated GABM runner."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_rounds: int = 1,
        llm_client: InstrumentedOpenAIClient | StubLLMClient | None = None,
        **cfg: Any,
    ):
        self.model = model
        self.default_max_rounds = max_rounds
        self._cfg = cfg
        self._llm_client_override = llm_client
        self._openai_client: InstrumentedOpenAIClient | None = None
        self.last_result: RunResult | None = None

    def _resolve_llm_client(self, metrics: Metrics) -> InstrumentedOpenAIClient | StubLLMClient:
        """Bind a fresh Metrics instance; reuse cached OpenAI client across trials."""
        if self._llm_client_override is not None:
            client = self._llm_client_override
            client.metrics = metrics
            if isinstance(client, StubLLMClient):
                client.reset_for_run()
            return client

        if self._openai_client is None:
            self._openai_client = InstrumentedOpenAIClient(model=self.model, metrics=None)
        self._openai_client.metrics = metrics
        return self._openai_client

    def run(
        self,
        task_or_question: Union[TrivialTask, TravelTask, str, None] = None,
        *,
        question: str | None = None,
    ) -> Union[RunResult, tuple[str, float]]:
        """Execute a task through the environment-mediated loop.

        If given a str (or None with question kwarg), runs TrivialTask and returns
        (answer, latency_ms) for overhead harness compatibility.
        If given a Task object, returns RunResult.

        Returned latency covers the full run() span (setup + round loop), matching
        what the overhead harness records when it prefers runner-reported latency.
        """
        t0 = time.perf_counter()

        str_input = False
        if isinstance(task_or_question, str):
            task = TrivialTask(question=task_or_question)
            str_input = True
        elif task_or_question is None:
            task = TrivialTask(question=question or QUESTION)
            str_input = True
        else:
            task = task_or_question

        result = self._execute(task)
        result.metrics.total_time_s = time.perf_counter() - t0
        self.last_result = result

        if str_input:
            return result.answer, result.metrics.total_time_s * 1000.0
        return result

    def _execute(self, task: TrivialTask | TravelTask) -> RunResult:
        metrics = Metrics()
        llm_client = self._resolve_llm_client(metrics)

        environment = task.make_environment()
        gm = GameMaster(task, environment)
        agents: list[SkeletonAgent] = task.make_agents(llm_client)
        agent_map = {a.agent_id: a for a in agents}

        max_rounds = getattr(task, "max_rounds", self.default_max_rounds)
        trace: list[RoundTrace] = []
        rounds_executed = 0

        for _ in range(max_rounds):
            round_prompts: list[str] = []
            round_llm_calls_before = metrics.llm_calls
            actions: dict[str, dict] = {}

            for agent_id in task.agent_ids:
                obs = gm.observe(agent_id)
                agent = agent_map[agent_id]
                action = agent.act(obs)
                round_prompts.append(agent.last_prompt)
                actions[agent_id] = action

            tool_invocations = gm.apply_round(actions)
            rounds_executed += 1

            round_llm_calls = metrics.llm_calls - round_llm_calls_before
            trace.append(RoundTrace(
                prompts_issued=round_prompts,
                llm_calls=round_llm_calls,
                state_snapshot=environment.snapshot(),
                communication="gm_mediated",
                tool_invocations=tool_invocations,
            ))

            assert round_llm_calls == len(task.agent_ids), (
                f"Expected {len(task.agent_ids)} LLM calls, got {round_llm_calls}"
            )

            if task.is_complete(environment.state):
                break

        final_state = environment.snapshot()
        answer = task.extract_answer(final_state)
        metrics.output_chars = len(answer)

        expected_calls = len(task.agent_ids) * rounds_executed
        assert metrics.llm_calls == expected_calls, (
            f"llm_calls={metrics.llm_calls}, expected {expected_calls} "
            f"(N={len(task.agent_ids)} * R={rounds_executed})"
        )

        return RunResult(
            answer=answer,
            final_state=final_state,
            metrics=metrics,
            trace=trace,
        )


def _print_metrics(result: RunResult) -> None:
    m = result.metrics
    print(f"  llm_calls:           {m.llm_calls}")
    print(f"  input_tokens:        {m.input_tokens}")
    print(f"  output_tokens:       {m.output_tokens}")
    print(f"  llm_time_s:          {m.llm_time_s:.4f}")
    print(f"  total_time_s:        {m.total_time_s:.4f}")
    print(f"  framework_residual_s:{m.framework_residual_s:.4f}")
    print(f"  output_chars:        {m.output_chars}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GABM Skeleton smoke test")
    parser.add_argument("--task", choices=["trivial", "travel"], default="trivial")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--max-rounds", type=int, default=None)
    args = parser.parse_args()

    runner = GABMSkeletonRunner(model=args.model)

    if args.task == "trivial":
        result = runner.run(QUESTION)
        if isinstance(result, tuple):
            answer, latency_ms = result
            print(f"Answer: {answer}")
            print(f"Latency: {latency_ms:.2f} ms")
            if runner.last_result:
                print("Metrics:")
                _print_metrics(runner.last_result)
                if runner.last_result.metrics.input_tokens == 0:
                    print("  WARNING: input_tokens=0 — token capture may be broken")
    else:
        task = TravelTask(max_rounds=args.max_rounds or 5)
        result = runner.run(task)
        if isinstance(result, RunResult):
            print(f"Answer: {result.answer}")
            print("Metrics:")
            _print_metrics(result)
            print(f"Rounds: {len(result.trace)}")
            for row in result.trace_table():
                print(f"  {row}")


if __name__ == "__main__":
    main()
