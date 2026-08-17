"""
Determinism Aggregator
----------------------
Scan results/planning/determinism/ for repeated MATH-100 planning runs.

Correctness-based definitions (R runs, question q matched by qid):
  c[q][r] ∈ {0,1},  k_q = Σ_r c[q][r]

  Accuracy (%)  : per-run share correct; mean ± population SD across runs
  Time (s)      : per-run metrics.time_total; mean ± population SD
  Violations (%) : per-run share of preds starting with "FAILED:"; mean ± pop SD
  Stable        : mean_q [ k_q ∈ {0, R} ]  (same correctness in all runs)
  Modal         : mean_q max(k_q, R−k_q) / R

Usage:
    source mafenv/bin/activate
    python -m single_agent.reasoning.determinism_aggregate
"""

from __future__ import annotations

import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


RESULTS_DIR = Path("results/planning/determinism")
SUMMARY_PATH = RESULTS_DIR / "determinism_summary.json"
TEX_PATH = RESULTS_DIR / "determinism_table.tex"

# crewai_math_planning_{model}_run{N}.json
FILENAME_RE = re.compile(r"^crewai_math_planning_(.+)_run(\d+)\.json$")

DISPLAY_NAME = {
    "gpt-5.6-luna": "GPT-5.6-Luna",
    "gpt-5.6-terra": "GPT-5.6-Terra",
    "groq_llama-3.1-8b-instant": "Llama-3.1-8B (Groq)",
    "groq_openai_gpt-oss-20b": "GPT-OSS-20B (Groq)",
}


def _safe_pstdev(values: list[float]) -> float:
    """Population stdev; 0.0 when fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


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


def is_schema_violation(pred: Any) -> bool:
    return str(pred).startswith("FAILED:")


def aggregate_model(model_tag: str, run_files: list[tuple[int, Path]]) -> dict[str, Any]:
    runs = [load_run(path) for _, path in run_files]
    R = len(runs)

    correct: dict[str, list[int]] = defaultdict(list)
    viol: dict[str, list[int]] = defaultdict(list)

    per_run: list[dict[str, Any]] = []
    for run_idx, ((_, path), run) in enumerate(zip(run_files, runs)):
        seen: set[str] = set()
        for q in run.get("questions", []):
            qid = str(q.get("qid"))
            if qid in seen:
                continue
            seen.add(qid)
            correct[qid].append(1 if q.get("correct") else 0)
            viol[qid].append(1 if is_schema_violation(q.get("pred")) else 0)

        metrics = run.get("metrics", {})
        per_run.append({
            "run": run_files[run_idx][0],
            "file": path.name,
            "accuracy": metrics.get("accuracy"),
            "correct": metrics.get("correct"),
            "total": metrics.get("total"),
            "time_total": metrics.get("time_total"),
            "time_mean": metrics.get("time_mean"),
            "violation_rate": (
                sum(1 for q in run.get("questions", []) if is_schema_violation(q.get("pred")))
                / max(1, len(run.get("questions", [])))
            ),
        })

    # Require full coverage across all R runs
    qids = [q for q, v in correct.items() if len(v) == R]
    if len(qids) != len(correct):
        missing = sorted(set(correct) - set(qids))
        raise AssertionError(
            f"{model_tag}: runs do not cover the same questions "
            f"({len(qids)}/{len(correct)} complete); e.g. {missing[:5]}"
        )

    # Per-run accuracy / violations as percentages
    acc = [100.0 * statistics.mean(correct[q][r] for q in qids) for r in range(R)]
    vio = [100.0 * statistics.mean(viol[q][r] for q in qids) for r in range(R)]
    time = [
        float(run["metrics"]["time_total"])
        for run in runs
        if run.get("metrics", {}).get("time_total") is not None
    ]

    k = {q: sum(correct[q]) for q in qids}
    stable = statistics.mean(1.0 if k[q] in (0, R) else 0.0 for q in qids)
    modal = statistics.mean(max(k[q], R - k[q]) / R for q in qids)

    always_correct = statistics.mean(1.0 if k[q] == R else 0.0 for q in qids)
    always_incorrect = statistics.mean(1.0 if k[q] == 0 else 0.0 for q in qids)

    return {
        "model": model_tag,
        "display_name": DISPLAY_NAME.get(model_tag, model_tag),
        "n_runs": R,
        "n_questions": len(qids),
        "per_run": per_run,
        "accuracy_pct": {
            "mean": statistics.mean(acc),
            "std": _safe_pstdev(acc),
            "values": acc,
        },
        "time_total": {
            "mean": statistics.mean(time) if time else None,
            "std": _safe_pstdev(time) if time else None,
            "values": time,
        },
        "violations_pct": {
            "mean": statistics.mean(vio),
            "std": _safe_pstdev(vio),
            "values": vio,
        },
        "stable": stable,
        "modal": modal,
        "always_correct": always_correct,
        "always_incorrect": always_incorrect,
    }


def print_table(summary: dict[str, Any]) -> None:
    models = summary.get("models", [])
    if not models:
        print("No determinism run files found.")
        return

    header = (
        f"{'Model':<28} {'Acc %':>14} {'Time s':>14} "
        f"{'Viol %':>14} {'Stable':>8} {'Modal':>8}"
    )
    print("\n" + header)
    print("-" * len(header))

    for m in models:
        acc = m["accuracy_pct"]
        tt = m["time_total"]
        vio = m["violations_pct"]
        print(
            f"{m['display_name']:<28} "
            f"{acc['mean']:6.1f} ± {acc['std']:<5.1f} "
            f"{tt['mean']:7.0f} ± {tt['std']:<5.0f} "
            f"{vio['mean']:6.1f} ± {vio['std']:<5.1f} "
            f"{m['stable']:8.2f} "
            f"{m['modal']:8.2f}"
        )
    print()
    print("Accuracy / Time / Violations use population SD (statistics.pstdev).")
    print("Stable = share of questions with identical correctness on all runs.")
    print("Modal  = mean_q max(k_q, R-k_q) / R  (majority-label agreement).")
    print("Violations = share of questions whose pred starts with 'FAILED:'.")


def write_tex(summary: dict[str, Any], path: Path) -> None:
    rows = []
    for m in summary.get("models", []):
        acc = m["accuracy_pct"]
        tt = m["time_total"]
        vio = m["violations_pct"]
        rows.append(
            f"{m['display_name']} & "
            f"${acc['mean']:.1f} \\pm {acc['std']:.1f}$ & "
            f"${tt['mean']:.0f} \\pm {tt['std']:.0f}$ & "
            f"${vio['mean']:.1f} \\pm {vio['std']:.1f}$ & "
            f"${m['stable']:.2f}$ & "
            f"${m['modal']:.2f}$ \\\\"
        )

    tex = r"""\begin{table}[t]
\centering
\footnotesize
\caption{Determinism of CrewAI planning on MATH-100 over five independent runs.
Accuracy, total wall-clock time, and schema-violation rate (share of predictions
starting with \texttt{FAILED:}) are reported as mean~$\pm$~population standard
deviation across runs. Stable is the share of questions whose correctness label
is identical in all runs; Modal is the mean, over questions, of the fraction of
runs that agree with that question's majority correctness label.}
\label{tab:determinism-math100}
\begin{tabular}{lccccc}
\toprule
\textbf{Model} &
\textbf{Accuracy (\%)} &
\textbf{Time (s)} &
\textbf{Violations (\%)} &
\textbf{Stable} &
\textbf{Modal} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    path.write_text(tex, encoding="utf-8")
    print(f"📄 Wrote LaTeX table to {path}")


def main():
    grouped = discover_runs(RESULTS_DIR)
    models_out = []
    for model_tag in sorted(grouped.keys()):
        models_out.append(aggregate_model(model_tag, grouped[model_tag]))

    summary = {
        "results_dir": str(RESULTS_DIR),
        "std": "population (statistics.pstdev)",
        "definitions": {
            "accuracy_pct": "per-run mean_q c[q][r]*100; mean ± pstdev across runs",
            "time_total": "metrics.time_total; mean ± pstdev across runs",
            "violations_pct": "per-run share pred.startswith('FAILED:')*100; mean ± pstdev",
            "stable": "mean_q [k_q in {0,R}]",
            "modal": "mean_q max(k_q, R-k_q)/R",
        },
        "models": models_out,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"📄 Wrote summary to {SUMMARY_PATH}")

    print_table(summary)
    write_tex(summary, TEX_PATH)


if __name__ == "__main__":
    main()
