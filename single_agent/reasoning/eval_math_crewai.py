#!/usr/bin/env python
"""
Evaluate CrewAI predictions on MATH benchmark.
- Reads all JSON problems under data/MATH/test
- Runs CrewAI agent (math_task)
- Extracts gold answers with last_boxed_only_string + remove_boxed
- Compares with predictions using math_equivalence.is_equiv
- Aggregates overall, per-level, and per-subject accuracies
"""

import os, json, time
from single_agent.reasoning.crew_math import SingleAgentCrewMATH
from single_agent.reasoning.config import CONFIG

# helpers from original MATH repo
from single_agent.reasoning.math_benchmark_code.math_equivalence import is_equiv
from single_agent.reasoning.math_benchmark_code.evaluate_gpt3 import remove_boxed
from single_agent.reasoning.math_benchmark_code.dataset.util import last_boxed_only_string


def load_math_dataset(rootdir):
    """Yield (filepath, problem_data) for each JSON in dataset."""
    for subdir, _, files in os.walk(rootdir):
        for file in files:
            if not file.endswith(".json"):
                continue
            path = os.path.join(subdir, file)
            with open(path, "r") as fp:
                yield path, json.load(fp)


def normalize_answer(ans: str) -> str:
    """Apply the same normalization as in original eval (last_boxed + remove_boxed)."""
    ans = last_boxed_only_string(ans)
    ans = remove_boxed(ans)
    return ans


def evaluate_math(rootdir, n=None):
    """Run CrewAI agent on MATH problems and evaluate."""
    correct, total = 0, 0
    level_cors, subject_cors = {}, {}

    crew = SingleAgentCrewMATH().crew()

    start_all = time.perf_counter()

    for i, (path, problem_data) in enumerate(load_math_dataset(rootdir), 1):
        if n and i > n:
            break

        problem = problem_data["problem"]
        gold = normalize_answer(problem_data["solution"])

        inputs = {"question": problem, "planning": CONFIG["planning"]}
        result = crew.kickoff(inputs=inputs)
        pred = str(result.get("math_task") if isinstance(result, dict) else result).strip()
        pred = normalize_answer(pred)

        try:
            equiv = is_equiv(pred, gold)
        except Exception:
            equiv = False

        # normalize metadata
        level = problem_data.get("level")
        try:
            level = int(level.split("Level ")[1]) if isinstance(level, str) else int(level)
        except Exception:
            pass
        subject = problem_data.get("type")

        if level:
            level_cors.setdefault(level, []).append(equiv)
        if subject:
            subject_cors.setdefault(subject, []).append(equiv)

        if equiv:
            correct += 1
        total += 1

        print(f"[{i}] {subject} Level {level} | Gold: {gold} | Pred: {pred} | Correct: {equiv}")

    elapsed = time.perf_counter() - start_all

    # print results
    print("\n=== Results ===")
    print(f"Overall Accuracy = {correct}/{total} = {correct/total:.3f}")
    print(f"Time elapsed: {elapsed:.2f} sec")

    for lvl, vals in sorted(level_cors.items()):
        print(f"Level {lvl} Accuracy = {sum(vals)}/{len(vals)} = {sum(vals)/len(vals):.3f}")
    for subj, vals in sorted(subject_cors.items()):
        print(f"{subj} Accuracy = {sum(vals)}/{len(vals)} = {sum(vals)/len(vals):.3f}")

    # also save results
    os.makedirs("results", exist_ok=True)
    out_file = os.path.join("results", f"math_eval_crewai_{'planning' if CONFIG['planning'] else 'no_planning'}.txt")
    with open(out_file, "w") as f:
        f.write(f"Overall Accuracy = {correct}/{total} = {correct/total:.3f}\n")
        f.write(f"Time elapsed: {elapsed:.2f} sec\n\n")
        for lvl, vals in sorted(level_cors.items()):
            f.write(f"Level {lvl} Accuracy = {sum(vals)}/{len(vals)} = {sum(vals)/len(vals):.3f}\n")
        for subj, vals in sorted(subject_cors.items()):
            f.write(f"{subj} Accuracy = {sum(vals)}/{len(vals)} = {sum(vals)/len(vals):.3f}\n")
    print(f"\n📄 Results saved to {out_file}")


if __name__ == "__main__":
    rootdir = "data/MATH/test"
    evaluate_math(rootdir, n=5)   # set n=None for full benchmark
