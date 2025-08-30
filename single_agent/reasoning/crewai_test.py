#!/usr/bin/env python
"""
CrewAI Benchmark Runner
-----------------------
Unified benchmark runner for GSM8K, CSQA, and MATH with CrewAI.
Supports toggling planning on/off automatically.

Usage:
    source mafenv/bin/activate
    python -m single_agent.reasoning.crewai_test
"""

import os
import re
import time
import json
import glob
from datasets import load_dataset
from single_agent.reasoning.crew_gsm8k import SingleAgentCrewGSM8K
from single_agent.reasoning.crew_csqa import SingleAgentCrewCSQA
from single_agent.reasoning.crew_math import SingleAgentCrewMATH   # 🚀 add math crew
from single_agent.reasoning.config import CONFIG


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def extract_final_answer_gsm8k(answer_str: str) -> str:
    """Extract final numeric/string answer from GSM8K answers."""
    match = re.search(r"####\s*(\S+)", answer_str)
    return match.group(1).strip() if match else answer_str.strip()


def normalize_pred_numeric(pred: str) -> str:
    """Normalize numeric prediction by extracting the last number."""
    pred_norm = re.findall(r"-?\d+\.?\d*", pred)
    return pred_norm[-1] if pred_norm else pred.strip()


def normalize_pred_letter(pred: str) -> str:
    """Normalize letter prediction by extracting last a–e char."""
    pred_letter = re.findall(r"[abcde]", pred.lower())
    return pred_letter[-1] if pred_letter else pred.strip().lower()


def result_file(name: str) -> str:
    """Build filename with planning tag."""
    planning_tag = "planning" if CONFIG["planning"] else "no_planning"
    results_dir = os.path.join(os.getcwd(), CONFIG["results_dir"])
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, f"crewai_{name}_{planning_tag}.txt")


def char_token_count(text: str) -> int:
    """Approximate token count by characters / 4."""
    return max(1, len(text) // 4)


def analyze_tokens_from_logs(log_file):
    """Analyze tokens (approx) from CrewAI log file."""
    if not os.path.exists(log_file):
        print("No log file found:", log_file)
        return None

    with open(log_file, "r", encoding="utf-8") as f:
        logs = json.load(f)

    total_prompt, total_completion, total_all, tasks_count = 0, 0, 0, 0
    for event in logs:
        if event.get("status") == "completed" and "output" in event:
            prompt_tokens = char_token_count(event["task"])
            completion_tokens = char_token_count(event["output"])
            total = prompt_tokens + completion_tokens

            total_prompt += prompt_tokens
            total_completion += completion_tokens
            total_all += total
            tasks_count += 1

    avg_prompt = total_prompt / tasks_count if tasks_count else 0
    avg_completion = total_completion / tasks_count if tasks_count else 0
    avg_total = total_all / tasks_count if tasks_count else 0

    return {
        "tasks": tasks_count,
        "prompt_total": total_prompt,
        "completion_total": total_completion,
        "all_total": total_all,
        "prompt_avg": avg_prompt,
        "completion_avg": avg_completion,
        "all_avg": avg_total,
    }


# ---------------------------------------------------------------------
# Generic Benchmark Runner
# ---------------------------------------------------------------------

def run_benchmark(name, dataset, crew_cls, gold_fn, pred_norm_fn, log_file):
    """Generic benchmark runner."""
    correct, total = 0, len(dataset)
    results_file = result_file(f"reasoning_{name}")

    if os.path.exists(log_file):
        os.remove(log_file)

    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"=== {name.upper()} Benchmark (CrewAI Reasoning) ===\n")
        f.write(f"Planning: {CONFIG['planning']}\n")
        f.write(f"Total samples: {total}\n\n")

        start_total = time.perf_counter()
        for i, item in enumerate(dataset, 1):
            q_start = time.perf_counter()

            # prepare inputs
            if name == "gsm8k":
                question = item["question"]
                gold = gold_fn(item["answer"])
                inputs = {"question": question, "planning": CONFIG["planning"]}
            elif name == "csqa":
                question = item["question"]
                choices = item["choices"]["text"]
                labels = item["choices"]["label"]
                gold = item["answerKey"].lower()
                formatted_choices = "\n".join(f"{lbl}. {txt}" for lbl, txt in zip(labels, choices))
                inputs = {
                    "question": f"{question}\nOptions:\n{formatted_choices}\nReturn only the letter (a,b,c,d,e).",
                    "planning": CONFIG["planning"],
                }
            elif name == "math":   # 🚀 new
                question = item["problem"]
                gold = gold_fn(item["solution"])
                inputs = {"question": question, "planning": CONFIG["planning"]}
            else:
                raise ValueError(f"Unknown benchmark {name}")

            # run Crew
            result = crew_cls().crew().kickoff(inputs=inputs)
            task_key = f"{name}_task"
            pred = (str(result.get(task_key)) if isinstance(result, dict) else str(result)).strip()
            pred_final = pred_norm_fn(pred)

            if pred_final == gold:
                correct += 1

            q_end = time.perf_counter()
            f.write(f"--- Item {i} ---\n")
            f.write(f"Q: {question}\n")
            if name == "csqa":
                f.write(f"Choices:\n{formatted_choices}\n")
            f.write(f"Expected: {gold}\n")
            f.write(f"Predicted: {pred_final}\n")
            f.write(f"⏱️ Time: {q_end - q_start:.2f} sec\n\n")

        end_total = time.perf_counter()
        accuracy = 100 * correct / total
        f.write(f"\n✅ Accuracy: {accuracy:.2f}%\n")
        f.write(f"⏱️ Total time: {end_total - start_total:.2f} sec "
                f"(avg {(end_total-start_total)/total:.2f} sec/sample)\n")

        if os.path.exists(log_file):
            token_stats = analyze_tokens_from_logs(log_file)
            if token_stats:
                f.write("\n📊 Token Statistics (approx, chars/4):\n")
                f.write(f"- Prompt total: {token_stats['prompt_total']}, avg: {token_stats['prompt_avg']:.2f}\n")
                f.write(f"- Completion total: {token_stats['completion_total']}, avg: {token_stats['completion_avg']:.2f}\n")
                f.write(f"- Overall total: {token_stats['all_total']}, avg: {token_stats['all_avg']:.2f}\n")

    print(f"\n📄 {name.upper()} results saved to {results_file}")


# ---------------------------------------------------------------------
# Specific Benchmark Wrappers
# ---------------------------------------------------------------------

def run_gsm8k(n=None):
    split = "test" if n is None else f"test[:{n}]"
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    run_benchmark("gsm8k", dataset, SingleAgentCrewGSM8K,
                  extract_final_answer_gsm8k, normalize_pred_numeric,
                  "logs/SingleAgentCrewGSM8K.json")


def run_csqa(n=None):
    split = "validation" if n is None else f"validation[:{n}]"  # test split has no answers
    dataset = load_dataset("tau/commonsense_qa", split=split)
    run_benchmark("csqa", dataset, SingleAgentCrewCSQA,
                  lambda a: a.lower(), normalize_pred_letter,
                  "logs/SingleAgentCrewCSQA.json")


def run_math(n=None, root="data/MATH/test"):   # 🚀 new
    """Run MATH benchmark (local JSON files)."""
    files = glob.glob(os.path.join(root, "*", "*.json"))
    if n:
        files = files[:n]

    dataset = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            dataset.append(json.load(f))

    run_benchmark("math", dataset, SingleAgentCrewMATH,
                  lambda sol: normalize_pred_numeric(sol), normalize_pred_numeric,
                  "logs/SingleAgentCrewMATH.json")


def run_all_benchmarks():
    # if "gsm8k" in CONFIG["benchmarks"]:
    #     print("\n--- Running GSM8K Benchmark ---")
    #     run_gsm8k(n=CONFIG.get("n_gsm8k"))
    # if "csqa" in CONFIG["benchmarks"]:
    #     print("\n--- Running CSQA Benchmark ---")
    #     run_csqa(n=CONFIG.get("n_csqa"))
    if "math" in CONFIG["benchmarks"]:   # 🚀
        print("\n--- Running MATH Benchmark ---")
        run_math(n=CONFIG.get("n_math"))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    for planning in [False, True]:
        CONFIG["planning"] = planning
        run_all_benchmarks()


if __name__ == "__main__":
    main()
