from dataclasses import dataclass
from typing import List, Optional
import json, os
from abc import ABC, abstractmethod

@dataclass
class Question:
    """Represents a single benchmark question."""
    qid: str
    question: str
    gold: str
    pred: Optional[str] = None
    correct: Optional[bool] = None


class Benchmark(ABC):
    """Abstract base class for all benchmarks."""
    def __init__(self, name: str):
        self.name = name
        self.questions: List[Question] = []

    @abstractmethod
    def load_data(self, *args, **kwargs):
        """Populate self.questions with Question objects."""
        pass

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize gold and prediction for comparison."""
        pass

    @abstractmethod
    def is_equiv(self, gold: str, pred: str) -> bool:
        """Judge correctness of prediction vs gold."""
        pass

    def set_pred(self, q: Question, pred: str):
        q.pred = self.normalize(pred)
        q.correct = self.is_equiv(q.gold, q.pred)


    def evaluate(self):
        total = len(self.questions)
        correct = sum(1 for q in self.questions if q.correct)
        return {
            "accuracy": correct / total if total else 0,
            "correct": correct,
            "total": total
        }

    def save_results(self, results_dir: str, filename: str):
        """Save benchmark results to results_dir/filename.json"""
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, filename)

        data = {
            "benchmark": self.name,
            "metrics": self.evaluate(),
            "questions": [q.__dict__ for q in self.questions]
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"📄 Saved results to {out_path}")

    def print_summary(self):
        m = self.evaluate()
        print(f"=== {self.name.upper()} ===")
        print(f"Accuracy: {m['accuracy']:.3f} ({m['correct']}/{m['total']})")
