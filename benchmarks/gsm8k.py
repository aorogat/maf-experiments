import re
from datasets import load_dataset
from .base import Benchmark, Question

# To test this file: Run,  python -m benchmarks.gsm8k

def extract_final_answer(ans: str) -> str:
    """Extract the final answer from GSM8K string (after ####)."""
    match = re.search(r"####\s*(\S+)", ans)
    return match.group(1).strip() if match else ans.strip()

def normalize_pred(pred: str) -> str:
    """Normalize numeric prediction by extracting the last number."""
    pred_norm = re.findall(r"-?\d+\.?\d*", pred)
    return pred_norm[-1] if pred_norm else pred.strip()


class GSM8KBenchmark(Benchmark):
    def __init__(self, split="test", n=None):
        super().__init__("gsm8k")
        self.load_data(split, n)

    def load_data(self, split: str, n=None) -> None:
        """Load GSM8K data into self.questions as Question objects."""
        ds = load_dataset("openai/gsm8k", "main", split=split if n is None else f"{split}[:{n}]")
        self.questions = [
            Question(
                qid=str(i+1),
                question=row["question"],
                gold=extract_final_answer(row["answer"])
            )
            for i, row in enumerate(ds)
        ]

    def normalize(self, text: str) -> str:
        return normalize_pred(text)

    def is_equiv(self, gold: str, pred: str) -> bool:
        return self.normalize(pred) == self.normalize(gold)



# 🔹 Run a quick self-test if called directly
if __name__ == "__main__":
    bench = GSM8KBenchmark(split="test", n=5)
    print("✅ Loaded GSM8K benchmark")
    bench.print_summary()
    # print the first few questions for clarity
    for q in bench.questions:
        bench.set_pred(q, "20")  # Fake value
        print(
            f"Q{q.qid}: {q.question[:50]}... "
            f"| Pred: {q.pred} \t| Gold: {q.gold} \t| Correct: {q.correct}"
        )
