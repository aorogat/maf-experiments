"""
Run Framework Overhead Experiment
Usage:
    python -m single_agent.framework_overhead.run_overhead
"""

import os
import time
import json
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import your runners
from single_agent.framework_overhead.direct_llm import DirectLLMRunner
from single_agent.framework_overhead.crewai_runner import CrewAIRunner
from single_agent.framework_overhead.langgraph_runner import LangGraphRunner
from single_agent.framework_overhead.concordia_runner import ConcordiaRunner
from single_agent.framework_overhead.openAgents_runner import OpenAgentsRunner
from single_agent.framework_overhead.openai_sdk_runner import OpenAISDKRunner
from single_agent.framework_overhead.agno_runner import AgnoRunner
from single_agent.framework_overhead.autogen_runner import AutoGenRunner
from single_agent.framework_overhead.gabm_skeleton_runner import GABMSkeletonRunner
from single_agent.framework_overhead.openhands_runner import OpenHandsRunner
from single_agent.framework_overhead.instrumentation import (
    install_patches,
    measure,
    overlap_flag,
    residual_ms,
)


# -------------------------------------------------------------------
# Global Experiment Settings
# -------------------------------------------------------------------
MODEL = "openai/gpt-4o-mini"   # Unified model (Concordia requires OpenAI -> all must match)
MODEL_NO_Prefix = "gpt-4o-mini"

QUESTION = "What is 2+2?"
TRIALS = 50                   # 🔹 Adjust as needed
CONCURRENCY = 4
RESULTS_DIR = "results/framework_overhead"


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def run_trials(runner, name, trials=TRIALS, concurrency=CONCURRENCY):
    latencies, responses, lengths = [], [], []
    llm_calls_list, input_tokens_list, output_tokens_list = [], [], []
    llm_time_ms_list, residual_ms_list = [], []
    max_concurrency_list = []
    overlap_count = 0
    residual_valid_count = 0

    start_total = time.perf_counter()
    lock = threading.Lock()
    completed = set()  # track seen trial indices

    def _one_run(i, worker_id):
        t0 = time.perf_counter()
        with measure() as m:
            resp, latency = runner.run(QUESTION)   # must raise on hard errors
        total_ms = (time.perf_counter() - t0) * 1000.0
        latency = latency if latency is not None else total_ms
        res_ms, res_valid = residual_ms(m, total_ms)

        if overlap_flag(m, total_ms):
            with lock:
                print(
                    f"   [WARN overlap] {name} trial {i+1}: "
                    f"max_concurrency={m.max_concurrency} "
                    f"llm_ms={m.llm_time_s*1000:.2f} total_ms={total_ms:.2f}",
                    flush=True,
                )

        with lock:
            print(
                f"   [Worker {worker_id}] Trial {i+1}/{trials} done "
                f"({latency:.2f} ms, resp_len={len(resp)}, "
                f"llm_calls={m.llm_calls}, tokens={m.input_tokens}/{m.output_tokens})",
                flush=True,
            )
        return {
            "i": i,
            "latency": latency,
            "resp": resp,
            "llm_calls": m.llm_calls,
            "input_tokens": m.input_tokens,
            "output_tokens": m.output_tokens,
            "llm_time_ms": m.llm_time_s * 1000.0,
            "residual_ms": res_ms,
            "residual_valid": res_valid,
            "max_concurrency": m.max_concurrency,
            "overlap": overlap_flag(m, total_ms),
        }

    futures = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for i in range(trials):
            futures.append(ex.submit(_one_run, i, (i % concurrency) + 1))

        for f in as_completed(futures):
            try:
                row = f.result()
            except Exception as e:
                with lock:
                    print(f"   [ERROR] Trial failed: {e!r}", flush=True)
                continue

            i = row["i"]
            if i in completed:
                continue
            completed.add(i)

            latencies.append(row["latency"])
            responses.append(row["resp"])
            lengths.append(len(row["resp"]))
            llm_calls_list.append(row["llm_calls"])
            input_tokens_list.append(row["input_tokens"])
            output_tokens_list.append(row["output_tokens"])
            llm_time_ms_list.append(row["llm_time_ms"])
            residual_ms_list.append(row["residual_ms"])
            max_concurrency_list.append(row["max_concurrency"])
            if row["overlap"]:
                overlap_count += 1
            if row["residual_valid"]:
                residual_valid_count += 1

    end_total = time.perf_counter()
    total_runtime = end_total - start_total
    throughput = len(latencies) / total_runtime if total_runtime > 0 else 0.0

    if len(latencies) < trials:
        print(f"   [WARN] Completed {len(latencies)}/{trials} successful trials.", flush=True)

    # Decomposition validation
    llm_calls_mean = statistics.mean(llm_calls_list) if llm_calls_list else 0.0
    input_tokens_mean = statistics.mean(input_tokens_list) if input_tokens_list else 0.0
    output_tokens_mean = statistics.mean(output_tokens_list) if output_tokens_list else 0.0
    llm_time_s_mean = (statistics.mean(llm_time_ms_list) / 1000.0) if llm_time_ms_list else 0.0
    framework_residual_s_mean = (statistics.mean(residual_ms_list) / 1000.0) if residual_ms_list else 0.0
    total_latency_s_mean = (statistics.mean(latencies) / 1000.0) if latencies else 0.0
    llm_time_frac = (llm_time_s_mean / total_latency_s_mean) if total_latency_s_mean > 0 else 0.0

    if llm_calls_mean < 1 or input_tokens_mean == 0:
        print(
            f"   [WARN zero-capture] {name}: boundary missed — "
            f"llm_calls_mean={llm_calls_mean:.2f} input_tokens_mean={input_tokens_mean:.1f}",
            flush=True,
        )
    if overlap_count > 0:
        print(
            f"   [WARN overlap] {name}: {overlap_count}/{len(latencies)} trials "
            f"had concurrent LLM calls (residual may be invalid)",
            flush=True,
        )

    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = (statistics.quantiles(latencies, n=100)[94]
           if len(latencies) >= 100 else
           (sorted(latencies)[max(0, int(len(latencies)*0.95)-1)] if latencies else 0.0))

    result = {
        "name": name,
        "trials": trials,
        "successful_trials": len(latencies),
        "concurrency": concurrency,
        "total_runtime_sec": total_runtime,
        "p50_latency": p50,
        "p95_latency": p95,
        "throughput_req_per_sec": throughput,
        "responses_preview": responses[:3],
        "output_chars_total": sum(lengths),
        "output_chars_mean": statistics.mean(lengths) if lengths else 0,
        "output_chars_max": max(lengths) if lengths else 0,
        "output_chars_min": min(lengths) if lengths else 0,
        # API-boundary decomposition (R1.O2)
        "llm_calls_mean": llm_calls_mean,
        "input_tokens_mean": input_tokens_mean,
        "output_tokens_mean": output_tokens_mean,
        "llm_time_s_mean": llm_time_s_mean,
        "framework_residual_s_mean": framework_residual_s_mean,
        "llm_time_frac": llm_time_frac,
        "max_concurrency_observed": max(max_concurrency_list) if max_concurrency_list else 0,
        "overlap_trials": overlap_count,
        "residual_valid_frac": (residual_valid_count / len(latencies)) if latencies else 0.0,
        "overlap_warning": overlap_count > 0,
    }
    return result


def save_results(results, filename=f"framework_overhead_{TRIALS}_TRIALS.json"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"📄 Saved results to {out_path}")


# -------------------------------------------------------------------
# Main Experiment
# -------------------------------------------------------------------

def main():
    install_patches()

    runners = {
        "Direct LLM": DirectLLMRunner(model=MODEL),
        "AutoGen": AutoGenRunner(model=MODEL_NO_Prefix),
        "CrewAI": CrewAIRunner(model=MODEL),
        "LangGraph": LangGraphRunner(model=MODEL),
        "OpenAgents": OpenAgentsRunner(model=MODEL),
        "OpenAISDK": OpenAISDKRunner(model=MODEL),
        "Agno": AgnoRunner(model=MODEL_NO_Prefix),
        "GABM Skeleton": GABMSkeletonRunner(model=MODEL),
        "Concordia": ConcordiaRunner(model=MODEL),  # each .run() will rebuild simulation
        "OpenHands": OpenHandsRunner(model=MODEL),
    }

    all_results = []

    for name, runner in runners.items():
        print(f"\n=== Running {name} for {TRIALS} trials with concurrency={CONCURRENCY} ===")
        results = run_trials(runner, name, TRIALS, CONCURRENCY)
        all_results.append(results)
        print(f"{name}: "
              f"p50={results['p50_latency']:.2f} ms | "
              f"p95={results['p95_latency']:.2f} ms | "
              f"Throughput={results['throughput_req_per_sec']:.2f} req/s | "
              f"Chars(mean={results['output_chars_mean']:.1f}, "
              f"max={results['output_chars_max']}, min={results['output_chars_min']}) | "
              f"LLM calls={results['llm_calls_mean']:.1f} | "
              f"tokens(in/out)={results['input_tokens_mean']:.0f}/{results['output_tokens_mean']:.0f} | "
              f"llm_time={results['llm_time_s_mean']*1000:.1f}ms | "
              f"residual={results['framework_residual_s_mean']*1000:.1f}ms | "
              f"Runtime={results['total_runtime_sec']:.2f}s")

    save_results(all_results)


if __name__ == "__main__":
    main()
