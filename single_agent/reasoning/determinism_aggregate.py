"""
Determinism Aggregator
----------------------
Scan results/planning/determinism/ for repeated MATH-100 planning runs and
compute mean/std of accuracy & runtime plus per-question prediction consistency.

Usage:
    source mafenv/bin/activate
    python -m single_agent.reasoning.determinism_aggregate
"""

from __future__ import annotations

import json
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RESULTS_DIR = Path("results/planning/determinism")
SUMMARY_PATH = RESULTS_DIR / "determinism_summary.json"

# crewai_math_planning_{model}_run{N}.json
FILENAME_RE = re.compile(
    r"^crewai_math_planning_(.+)_run(\d+)\.json$"
)


def _safe_pstdev(values: list[float]) -> float:
    """Population stdev; 0.0 when fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def _mean_std(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": statistics.mean(values),
        "std": _safe_pstdev(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def discover_runs(results_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    """Group run files by model tag: {model_tag: [(run_idx, path), ...]}."""
    grouped: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    if not results_dir.is_dir():
        return {}

    for path in sorted(results_dir.glob("crewai_math_planning_*_run*.json")):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        model_tag, run_s = m.group(1), m.group(2)
        grouped[model_tag].append((int(run_s), path))

    for model_tag in grouped:
        grouped[model_tag].sort(key=lambda x: x[0])
    return dict(grouped)


def load_run(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def per_question_consistency(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compare predictions / correctness across runs, matched by qid.

    Returns:
      fully_consistent_frac, mean_modal_agreement, correctness_flip_frac,
      n_questions, n_runs
    """
    # qid -> list of (pred, correct) across runs
    by_qid: dict[str, list[tuple[str | None, bool | None]]] = defaultdict(list)

    for data in runs:
        seen: set[str] = set()
        for q in data.get("questions", []):
            qid = str(q.get("qid"))
            if qid in seen:
                continue
            seen.add(qid)
            by_qid[qid].append((q.get("pred"), q.get("correct")))

    if not by_qid:
        return {
            "fully_consistent_frac": None,
            "mean_modal_agreement": None,
            "correctness_flip_frac": None,
            "n_questions": 0,
            "n_runs": len(runs),
        }

    n_questions = len(by_qid)
    n_runs = len(runs)
    fully_consistent = 0
    modal_agreements: list[float] = []
    correctness_flips = 0

    for qid, entries in by_qid.items():
        preds = [e[0] for e in entries]
        corrects = [e[1] for e in entries]

        # Pad missing runs with None so agreement denom is n_runs
        while len(preds) < n_runs:
            preds.append(None)
            corrects.append(None)

        if len(set(preds)) == 1:
            fully_consistent += 1

        counts = Counter(preds)
        modal_count = counts.most_common(1)[0][1]
        modal_agreements.append(modal_count / n_runs)

        if len(set(corrects)) > 1:
            correctness_flips += 1

    return {
        "fully_consistent_frac": fully_consistent / n_questions,
        "mean_modal_agreement": statistics.mean(modal_agreements),
        "correctness_flip_frac": correctness_flips / n_questions,
        "n_questions": n_questions,
        "n_runs": n_runs,
    }


def aggregate_model(model_tag: str, run_files: list[tuple[int, Path]]) -> dict[str, Any]:
    runs_data: list[dict[str, Any]] = []
    accuracies: list[float] = []
    time_totals: list[float] = []
    time_means: list[float] = []
    per_run: list[dict[str, Any]] = []

    for run_idx, path in run_files:
        data = load_run(path)
        runs_data.append(data)
        metrics = data.get("metrics", {})
        acc = metrics.get("accuracy")
        tt = metrics.get("time_total")
        tm = metrics.get("time_mean")

        if acc is not None:
            accuracies.append(float(acc))
        if tt is not None:
            time_totals.append(float(tt))
        if tm is not None:
            time_means.append(float(tm))

        per_run.append({
            "run": run_idx,
            "file": path.name,
            "accuracy": acc,
            "correct": metrics.get("correct"),
            "total": metrics.get("total"),
            "time_total": tt,
            "time_mean": tm,
        })

    consistency = per_question_consistency(runs_data)

    return {
        "model": model_tag,
        "n_runs": len(run_files),
        "per_run": per_run,
        "accuracy": _mean_std(accuracies),
        "time_total": _mean_std(time_totals),
        "time_mean": _mean_std(time_means),
        "consistency": consistency,
    }


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def print_table(summary: dict[str, Any]) -> None:
    models = summary.get("models", [])
    if not models:
        print("No determinism run files found.")
        return

    header = (
        f"{'Model':<28} {'Acc μ':>8} {'Acc σ':>8} "
        f"{'Ttot μ':>10} {'Ttot σ':>10} "
        f"{'Tmean μ':>10} {'Tmean σ':>10} "
        f"{'FullCons':>9} {'ModalAgr':>9} {'CorrFlip':>9}"
    )
    print("\n" + header)
    print("-" * len(header))

    for m in models:
        print(
            f"{m['model']:<28} "
            f"{_fmt(m['accuracy']['mean']):>8} {_fmt(m['accuracy']['std']):>8} "
            f"{_fmt(m['time_total']['mean'], 2):>10} {_fmt(m['time_total']['std'], 2):>10} "
            f"{_fmt(m['time_mean']['mean'], 2):>10} {_fmt(m['time_mean']['std'], 2):>10} "
            f"{_fmt(m['consistency']['fully_consistent_frac']):>9} "
            f"{_fmt(m['consistency']['mean_modal_agreement']):>9} "
            f"{_fmt(m['consistency']['correctness_flip_frac']):>9}"
        )
    print()
    print("FullCons  = fraction of questions with identical pred across all runs")
    print("ModalAgr  = mean over questions of (modal pred count / n_runs)")
    print("CorrFlip  = fraction of questions whose correct flag flips across runs")


def main():
    grouped = discover_runs(RESULTS_DIR)
    models_out = []
    for model_tag in sorted(grouped.keys()):
        models_out.append(aggregate_model(model_tag, grouped[model_tag]))

    summary = {
        "results_dir": str(RESULTS_DIR),
        "models": models_out,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"📄 Wrote summary to {SUMMARY_PATH}")

    print_table(summary)


if __name__ == "__main__":
    main()
