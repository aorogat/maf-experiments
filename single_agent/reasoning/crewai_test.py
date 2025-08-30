#!/usr/bin/env python
"""
CrewAI Benchmark Runner
-----------------------
Run multiple benchmark datasets (e.g., GSM8K, CSQA) with CrewAI.
Each benchmark is tested by a separate function.

Usage:
    source mafenv/bin/activate
    python -m single_agent.reasoning.crewai_test
"""

import re
import os
import time
from datasets import load_dataset
from single_agent.reasoning.crew_gsm8k import SingleAgentCrewGSM8K
from single_agent.reasoning.crew_csqa import SingleAgentCrewCSQA
from single_agent.reasoning.config import CONFIG


def extract_final_answer(answer_str: str) -> str:
    """Extract final numeric/string answer from GSM8K answers."""
    match = re.search(r"####\s*(\S+)", answer_str)
    return match.group(1).strip() if match else answer_str.strip()


def result_file(name: str) -> str:
    """Build filename with planning tag."""
    planning_tag = "planning" if CONFIG["planning"] else "no_planning"
    results_dir = os.path.join(os.getcwd(), CONFIG["results_dir"])
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, f"crewai_{name}_{planning_tag}.txt")


def run_gsm8k(n=None):
    """Evaluate gsm8k_task on GSM8K dataset. n=None → use full test split."""
    split = "test" if n is None else f"test[:{n}]"
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    correct, total = 0, len(dataset)

    results_file = result_file("reasoning_gsm8k")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("=== GSM8K Benchmark (CrewAI Reasoning) ===\n")
        f.write(f"Planning: {CONFIG['planning']}\n")
        f.write(f"Total samples: {total}\n\n")

        start_total = time.perf_counter()
        for i, item in enumerate(dataset, 1):
            q_start = time.perf_counter()

            question = item["question"]
            gold = extract_final_answer(item["answer"])

            inputs = {"question": question, "planning": CONFIG["planning"]}
            result = SingleAgentCrewGSM8K().crew().kickoff(inputs=inputs)

            pred = (str(result.get("gsm8k_task"))
                    if isinstance(result, dict) else str(result)).strip()
            pred_norm = re.findall(r"-?\d+\.?\d*", pred)
            pred_final = pred_norm[-1] if pred_norm else pred

            if pred_final == gold:
                correct += 1

            q_end = time.perf_counter()
            f.write(f"--- Item {i} ---\n")
            f.write(f"Q: {question}\n")
            f.write(f"Expected: {gold}\n")
            f.write(f"Predicted: {pred_final}\n")
            f.write(f"⏱️ Time: {q_end - q_start:.2f} sec\n\n")

        end_total = time.perf_counter()
        accuracy = 100 * correct / total
        f.write(f"\n✅ Accuracy: {accuracy:.2f}%\n")
        f.write(f"⏱️ Total time: {end_total - start_total:.2f} sec "
                f"(avg {(end_total-start_total)/total:.2f} sec/sample)\n")

    print(f"\n📄 GSM8K results saved to {results_file}")


def run_csqa(n=None):
    """Evaluate csqa_task on CommonsenseQA dataset. n=None → use full test split."""
    split = "validation" if n is None else f"validation[:{n}]" #the test split does not have a key answer
    dataset = load_dataset("tau/commonsense_qa", split=split)
    correct, total = 0, len(dataset)

    results_file = result_file("reasoning_csqa")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("=== CommonsenseQA Benchmark (CrewAI Reasoning) ===\n")
        f.write(f"Planning: {CONFIG['planning']}\n")
        f.write(f"Total samples: {total}\n\n")

        start_total = time.perf_counter()
        for i, item in enumerate(dataset, 1):
            q_start = time.perf_counter()

            question = item["question"]
            choices = item["choices"]["text"]
            labels = item["choices"]["label"]
            gold = item["answerKey"].lower()

            formatted_choices = "\n".join(
                f"{lbl}. {txt}" for lbl, txt in zip(labels, choices)
            )

            inputs = {
                "question": f"{question}\nOptions:\n{formatted_choices}\n"
                            f"Return only the letter (a,b,c,d,e).",
                "planning": CONFIG["planning"],
            }
            result = SingleAgentCrewCSQA().crew().kickoff(inputs=inputs)

            pred = (str(result.get("csqa_task"))
                    if isinstance(result, dict) else str(result)).strip().lower()
            pred_letter = re.findall(r"[abcde]", pred)
            pred_final = pred_letter[-1] if pred_letter else pred

            if pred_final == gold:
                correct += 1

            q_end = time.perf_counter()
            f.write(f"--- Item {i} ---\n")
            f.write(f"Q: {question}\n")
            f.write(f"Choices:\n{formatted_choices}\n")
            f.write(f"Expected: {gold}\n")
            f.write(f"Predicted: {pred_final}\n")
            f.write(f"⏱️ Time: {q_end - q_start:.2f} sec\n\n")

        end_total = time.perf_counter()
        accuracy = 100 * correct / total
        f.write(f"\n✅ Accuracy: {accuracy:.2f}%\n")
        f.write(f"⏱️ Total time: {end_total - start_total:.2f} sec "
                f"(avg {(end_total-start_total)/total:.2f} sec/sample)\n")

    print(f"\n📄 CSQA results saved to {results_file}")

def run_all_benchmarks():
    if "gsm8k" in CONFIG["benchmarks"]:
        print("\n--- Running GSM8K Benchmark ---")
        run_gsm8k(n=CONFIG.get("n_gsm8k"))  # set in config
    if "csqa" in CONFIG["benchmarks"]:
        print("\n--- Running CSQA Benchmark ---")
        run_csqa(n=CONFIG.get("n_csqa"))  # set in config

def main():
    CONFIG["planning"] = False
    run_all_benchmarks()
    CONFIG["planning"] = True
    run_all_benchmarks()
    

if __name__ == "__main__":
    main()
