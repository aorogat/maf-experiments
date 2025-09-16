import os, json
from .base import Benchmark, Question
from llms.local_llm import LocalOllamaLLM
from llms.remote_llm import OpenAILLM
from single_agent.reasoning.config import CONFIG


import re

def normalize_answer(sol: str) -> str:
    """Extract final boxed answer from MATH solution string, or fallback to last line."""
    text = str(sol).strip()

    # Manually parse to handle nested braces in \boxed{...}
    start = text.rfind(r"\boxed{")
    if start != -1:
        i = start + len(r"\boxed{")
        depth = 1
        result = []
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            result.append(text[i])
            i += 1
        return "".join(result).strip()

    # Fallback: take last non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        return lines[-1]
    return text




class MATHBenchmark(Benchmark):
    def __init__(self, root="data/MATH/test", split="test", n=None, use_remote=True):
        self.root = root

        # Choose judge LLM backend
        model_name = CONFIG.get("judge_llm", "gpt-4o-mini" if use_remote else "gpt-oss:20b")
        if use_remote:
            self.llm = OpenAILLM(model_name)
        else:
            self.llm = LocalOllamaLLM(model_name)

        super().__init__("math")
        self.load_data(split, n)

    def load_data(self, split: str, n=None) -> None:
        """Load MATH problems from JSON files."""
        files = []
        for subdir, _, fs in os.walk(self.root):
            for f in fs:
                if f.endswith(".json"):
                    files.append(os.path.join(subdir, f))
        if n:
            files = files[:n]

        self.questions = []
        for i, path in enumerate(files, 1):
            with open(path, "r", encoding="utf-8") as f:
                prob = json.load(f)
            gold = normalize_answer(prob["solution"])
            self.questions.append(
                Question(
                    qid=str(i),
                    question=prob["problem"],
                    gold=gold
                )
            )

    def normalize(self, text: str) -> str:
        return str(text).strip()

    def is_equiv(self, gold: str, pred: str) -> bool:
        """Judge if prediction is mathematically equivalent to gold."""
        if not gold or not pred:
            return False

        gold_norm = self.normalize(gold)
        pred_norm = self.normalize(pred)

        # Quick exact match
        if gold_norm == pred_norm:
            return True

        prompt = (
            f"Gold answer: {gold_norm}\n"
            f"Predicted answer: {pred_norm}\n\n"
            "Are these mathematically equivalent?\n"
            "Reply with exactly one word: YES or NO."
        )

        try:
            resp = self.llm.generate(prompt).strip().lower()
            if resp in {"yes", "y", "equivalent", "true"}:
                return True
            if resp in {"no", "n", "different", "false"}:
                return False
            if "yes" in resp or "equivalent" in resp:
                return True
            if "no" in resp or "different" in resp:
                return False
            return False
        except Exception as e:
            print("⚠️ LLM equivalence check failed:", e)
            return False


# 🔹 Run a quick self-test if called directly
if __name__ == "__main__":
    bench = MATHBenchmark(root="data/MATH/test", n=3, use_remote=True)
    print("✅ Loaded MATH benchmark")
    

    print("\n=== Detailed Results ===")
    for q in bench.questions:
        bench.set_pred(q, q.gold)  # test with perfect predictions

        # Pretty print results
        gold_preview = (q.gold[:320] + "...") if len(q.gold) > 120 else q.gold
        pred_preview = (q.pred[:320] + "...") if len(q.pred) > 120 else q.pred

        status = "✅ Correct" if q.correct else "❌ Incorrect"
        print(f"\nQ{q.qid}: {q.question[:200].replace('\n',' ')}...")
        print(f"   Gold: {gold_preview}")
        print(f"   Pred: {pred_preview}")
        print(f"   Result: {status}")

    bench.print_summary()