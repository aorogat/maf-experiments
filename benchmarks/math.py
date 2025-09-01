import os, json
from .base import Benchmark, Question
from llms.local_llm import LocalOllamaLLM
from single_agent.reasoning.config import CONFIG

# To test this file: Run,  python -m benchmarks.math

def normalize_answer(sol: str) -> str:
    """Extract final boxed answer from MATH solution string."""
    ans = last_boxed_only_string(sol)
    return remove_boxed(ans)


class MATHBenchmark(Benchmark):
    def __init__(self, root="data/MATH/test", split="test", n=None):
        self.root = root
        # initialize local LLM (Ollama) for equivalence checking
        self.llm = LocalOllamaLLM(CONFIG.get("judge_llm", "gpt-oss:20b"))
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
                    qid=str(i+1),
                    question=prob["problem"],
                    gold=gold
                )
            )

    def normalize(self, text: str) -> str:
        """No numeric extraction; return raw stripped text."""
        return str(text).strip()

    
    # def is_equiv(self, gold: str, pred: str) -> bool:
    #     if not gold or not pred:
    #         return False
    #     prompt = f"Gold answer: {gold}\nPredicted answer: {pred}\n\nAre these mathematically equivalent? Respond with only 'Yes' or 'No'."
    #     resp = self.llm.generate(prompt).strip().lower()
    #     return "yes" in resp  # safer than just startswith("y")

    def is_equiv(self, gold: str, pred: str) -> bool:
        """Use local LLM to decide if pred ≡ gold."""
        if not gold or not pred:
            return False

        prompt = (
            f"Gold answer: {gold}\n"
            f"Predicted answer: {pred}\n\n"
            "Determine if these are mathematically equivalent.\n"
            "Reply with exactly one word: YES or NO."
        )

        try:
            resp = self.llm.generate(prompt).strip().lower()
            print(f"🔍 Judge LLM response: {resp}")  # Debug log

            # Accept several variants just in case
            if resp in {"yes", "y", "equivalent", "true"}:
                return True
            elif resp in {"no", "n", "different", "false"}:
                return False
            else:
                # If it's verbose, check keywords
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
    bench = MATHBenchmark(root="data/MATH/test", n=3)
    print("✅ Loaded MATH benchmark")
    bench.print_summary()
    # Fake predictions
    for q in bench.questions:
        bench.set_pred(q, q.gold)  # pretend model predicted perfectly
        q.correct = bench.is_equiv(q.gold, q.pred)
        print(f"Q{q.qid}: {q.question[:50]}... | Pred: {q.pred} | Gold: {q.gold} | Correct: {q.correct}")
