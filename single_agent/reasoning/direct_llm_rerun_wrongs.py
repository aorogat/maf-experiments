#!/usr/bin/env python
"""
Rerun Direct-LLM planning ONLY for questions currently marked incorrect,
using the correct answer_mode format constraints.

Updates the existing results JSON in place (keeps correct answers untouched).

Usage:
    source mafenv/bin/activate
    python -m single_agent.reasoning.direct_llm_rerun_wrongs \
        --bench gsm8k --model anthropic/claude-opus-5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmarks.gsm8k import GSM8KBenchmark, normalize_pred, extract_final_answer
from benchmarks.csqa import CSQABenchmark
from benchmarks.math import MATHBenchmark
from single_agent.reasoning.direct_llm_planning_test import (
    CONFIG,
    BASE_CONFIG,
    sanitize_filename_component,
    solve_with_direct_planning,
)

ANSWER_MODE = {
    "gsm8k": "gsm8k_number",
    "csqa": "csqa_mcq",
    "math": "math_expression",
}


def result_path(results_dir: str, bench: str, model: str) -> Path:
    llm_tag = sanitize_filename_component(model)
    return Path(results_dir) / f"direct_planning_{bench}_planning_{llm_tag}.json"


def make_benchmark(bench: str):
    if bench == "gsm8k":
        return GSM8KBenchmark(split="test", n=CONFIG.get("n_gsm8k"))
    if bench == "csqa":
        return CSQABenchmark(split="validation", n=CONFIG.get("n_csqa"))
    if bench == "math":
        return MATHBenchmark(root="data/MATH/test", n=CONFIG.get("n_math", 100))
    raise ValueError(bench)


def score_question(bench_name: str, gold: str, pred: str, benchmark) -> bool:
    if bench_name == "gsm8k":
        g = normalize_pred(extract_final_answer(str(gold)))
        p = normalize_pred(str(pred))
        if p == g:
            return True
        try:
            return abs(float(p) - float(g)) < 1e-9
        except Exception:
            return False
    # Use benchmark equivalence for CSQA / MATH
    return benchmark.is_equiv(benchmark.normalize(gold), benchmark.normalize(pred))


def recompute_metrics(questions: list[dict]) -> dict:
    total = len(questions)
    correct = sum(1 for q in questions if q.get("correct"))
    times = [q["time_used"] for q in questions if q.get("time_used") is not None]
    tokens = [q["tokens_out"] for q in questions if q.get("tokens_out") is not None]
    metrics = {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
    }
    if times:
        metrics.update(
            time_total=sum(times),
            time_mean=sum(times) / len(times),
            time_min=min(times),
            time_max=max(times),
        )
    if tokens:
        metrics.update(
            tokens_total=sum(tokens),
            tokens_mean=sum(tokens) / len(tokens),
            tokens_min=min(tokens),
            tokens_max=max(tokens),
        )
    return metrics


def save(path: Path, data: dict) -> None:
    data["metrics"] = recompute_metrics(data["questions"])
    path.write_text(json.dumps(data, indent=2))


def rerun_wrongs(
    bench: str,
    model: str,
    results_dir: str,
    limit: int | None = None,
    save_every: int = 5,
    qids: set[str] | None = None,
) -> None:
    path = result_path(results_dir, bench, model)
    if not path.exists():
        raise FileNotFoundError(path)

    backup = path.with_suffix(path.suffix + ".bak_before_wrong_rerun")
    if not backup.exists():
        backup.write_text(path.read_text())
        print(f"🧾 Backup → {backup}")

    data = json.loads(path.read_text())
    questions = data["questions"]
    by_qid = {str(q["qid"]): q for q in questions}

    wrong = [q for q in questions if not q.get("correct")]
    if qids is not None:
        wrong = [q for q in wrong if str(q["qid"]) in qids]
    if limit is not None:
        wrong = wrong[:limit]

    old_acc = data["metrics"]["accuracy"]
    old_correct = data["metrics"]["correct"]
    print(
        f"\n=== Rerun wrongs: {bench} | {model} ===\n"
        f"file={path}\n"
        f"current={100 * old_acc:.1f}% ({old_correct}/{data['metrics']['total']})  "
        f"wrongs_to_rerun={len(wrong)}  answer_mode={ANSWER_MODE[bench]}\n"
        + (f"qid_filter={len(qids)} ids\n" if qids is not None else "")
    )

    benchmark = make_benchmark(bench)
    # Map qid → live Question for text (gold already in JSON)
    live = {str(q.qid): q for q in benchmark.questions}

    flipped = 0
    still_wrong = 0
    errors = 0

    for i, qrow in enumerate(wrong, 1):
        qid = str(qrow["qid"])
        live_q = live.get(qid)
        question_text = live_q.question if live_q else qrow.get("question", "")
        gold = qrow.get("gold", live_q.gold if live_q else "")

        print(f"🔹 [{i}/{len(wrong)}] qid={qid}")

        # Keep prior wrong answer for audit
        if "pred_before_wrong_rerun" not in qrow:
            qrow["pred_before_wrong_rerun"] = qrow.get("pred")
            qrow["llm_response_before_wrong_rerun"] = qrow.get("llm_response")

        q_start = time.perf_counter()
        try:
            plan, answer, tokens_out = solve_with_direct_planning(
                question_text,
                model,
                model,
                answer_mode=ANSWER_MODE[bench],
            )
            elapsed = time.perf_counter() - q_start

            if isinstance(answer, str) and answer.startswith(("Failed:", "LLM_ERROR")):
                errors += 1

            # Score with benchmark helpers
            if bench == "math":
                # MATH judge needs set_pred on a Question object
                if live_q is None:
                    raise RuntimeError(f"MATH qid {qid} missing from benchmark load")
                benchmark.set_pred(
                    live_q,
                    answer,
                    time_used=elapsed,
                    tokens_out=tokens_out,
                    llm_response=f"PLAN:\n{plan}\n\nANSWER:\n{answer}",
                )
                pred = live_q.pred
                correct = bool(live_q.correct)
                llm_response = live_q.llm_response
            else:
                pred = answer
                if bench == "gsm8k":
                    pred = normalize_pred(answer)
                elif bench == "csqa":
                    pred = benchmark.normalize(answer)
                correct = score_question(bench, gold, answer, benchmark)
                llm_response = f"PLAN:\n{plan}\n\nANSWER:\n{answer}"

            qrow["pred"] = pred
            qrow["correct"] = correct
            qrow["time_used"] = elapsed
            qrow["tokens_out"] = tokens_out
            qrow["llm_response"] = llm_response
            qrow["wrong_rerun"] = True
            qrow["answer_mode"] = ANSWER_MODE[bench]

            if correct:
                flipped += 1
                print(f"   ✅ {pred} (was {qrow.get('pred_before_wrong_rerun')})")
            else:
                still_wrong += 1
                print(f"   ❌ {pred} gold={gold}")

        except Exception as e:
            errors += 1
            still_wrong += 1
            elapsed = time.perf_counter() - q_start
            print(f"   ⚠️ failed: {e}")
            qrow["pred"] = f"FAILED: {e}"
            qrow["correct"] = False
            qrow["time_used"] = elapsed
            qrow["tokens_out"] = 0
            qrow["llm_response"] = f"LLM_ERROR: {e}"
            qrow["wrong_rerun"] = True

        by_qid[qid] = qrow

        if i % save_every == 0 or i == len(wrong):
            data["questions"] = [by_qid[str(q["qid"])] for q in questions]
            data["metrics"] = recompute_metrics(data["questions"])
            data["metrics"]["wrong_rerun_note"] = (
                f"partial rerun of previously wrong items with answer_mode={ANSWER_MODE[bench]}"
            )
            save(path, data)
            print(
                f"   💾 checkpoint acc={100 * data['metrics']['accuracy']:.1f}% "
                f"({data['metrics']['correct']}/{data['metrics']['total']})"
            )

    data["questions"] = [by_qid[str(q["qid"])] for q in questions]
    save(path, data)
    new = data["metrics"]
    print(
        f"\n=== Done {bench} ===\n"
        f"before: {100 * old_acc:.1f}% ({old_correct}/{new['total']})\n"
        f"after:  {100 * new['accuracy']:.1f}% ({new['correct']}/{new['total']})\n"
        f"flipped_to_correct={flipped}  still_wrong={still_wrong}  errors={errors}\n"
        f"saved → {path}\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", choices=["gsm8k", "csqa", "math"], default="gsm8k")
    parser.add_argument("--model", default="anthropic/claude-opus-5")
    parser.add_argument("--results-dir", default="results/planning_direct")
    parser.add_argument("--limit", type=int, default=None, help="Max wrongs to rerun (smoke)")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--qids-file",
        default=None,
        help="Optional file of qids (one per line) to restrict which wrongs are rerun",
    )
    args = parser.parse_args()

    CONFIG.update(BASE_CONFIG)
    CONFIG["llm"] = args.model
    CONFIG["planning_llm"] = args.model

    qids = None
    if args.qids_file:
        qids = {
            line.strip()
            for line in Path(args.qids_file).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }

    rerun_wrongs(
        bench=args.bench,
        model=args.model,
        results_dir=args.results_dir,
        limit=args.limit,
        save_every=args.save_every,
        qids=qids,
    )


if __name__ == "__main__":
    main()
