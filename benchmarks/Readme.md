# 📊 Benchmarks Framework

This folder provides a **unified interface** for working with multiple benchmarks (e.g., GSM8K, CSQA, MATH) under a consistent structure.  
The goal is to make it easy to:
- Load benchmark data into structured objects
- Run predictions with CrewAI (or any model)
- Store predictions and automatically evaluate correctness
- Save results to JSON files for reproducibility

---

## 🔹 Core Interface

### `Question` Dataclass
Each benchmark question is represented as a `Question` object:
```python
@dataclass
class Question:
    qid: str              # Unique question ID
    question: str         # Input question text
    gold: str             # Gold (reference) answer
    pred: Optional[str]   # Model’s predicted answer
    correct: Optional[bool] # Whether pred ≡ gold
```

### `Benchmark` Base Class
All benchmarks inherit from `Benchmark`:
```python
class Benchmark(ABC):
    def __init__(self, name: str):
        self.name = name
        self.questions: List[Question] = []
```

#### Abstract methods every benchmark must implement:
- `load_data(...)` → Load dataset and populate `self.questions`
- `normalize(text: str)` → Normalize answers for comparison  
  (e.g., strip whitespace, extract numbers/letters)
- `is_equiv(gold: str, pred: str)` → Define correctness checking  
  (exact match or LLM-based equivalence for math)

#### Provided utilities:
- `set_pred(q: Question, pred: str)` → Assigns prediction, marks `correct`
- `evaluate()` → Returns accuracy, total, correct counts
- `save_results(results_dir, filename)` → Saves evaluation results to JSON
- `print_summary()` → Prints benchmark name + accuracy summary

---

## 🔹 Example Benchmarks

### GSM8K (Math Word Problems)
- Loads from HuggingFace `openai/gsm8k`
- Normalization extracts final number from gold/pred
- Correctness = string equality of normalized values

### CSQA (CommonsenseQA)
- Loads from HuggingFace `tau/commonsense_qa`
- Questions are formatted with multiple-choice options
- Correctness = matching the gold letter (a–e)

### MATH (Proof-style Math Problems)
- Loads from local JSON files (`data/MATH/test`)
- Gold answers extracted from LaTeX `\boxed{}` solutions
- Correctness judged by a **local LLM** (Ollama) that decides if prediction ≡ gold

---

## 🔹 Usage

### 1. Quick self-test
Each benchmark file can be run directly:
```bash
python -m benchmarks.gsm8k
python -m benchmarks.csqa
python -m benchmarks.math
```

### 2. Inside an experiment
```python
from benchmarks.gsm8k import GSM8KBenchmark

bench = GSM8KBenchmark(split="test", n=10)
for q in bench.questions:
    pred = "42"  # replace with model output
    bench.set_pred(q, pred)

bench.print_summary()
bench.save_results("results", "gsm8k_test.json")
```

### 3. With CrewAI
The `crewai_test.py` runner integrates CrewAI agents and evaluates all benchmarks in both **planning** and **no-planning** modes.

---

## 🔹 Results Format

Saved results look like:
```json
{
  "benchmark": "gsm8k",
  "metrics": {
    "accuracy": 0.85,
    "correct": 85,
    "total": 100
  },
  "questions": [
    {
      "qid": "1",
      "question": "John has 3 apples...",
      "gold": "6",
      "pred": "6",
      "correct": true
    },
    ...
  ]
}
```

---

## 🔹 Adding a New Benchmark

1. Create a new file under `benchmarks/` (e.g., `newbench.py`)  
2. Subclass `Benchmark` and implement:
   - `load_data()`
   - `normalize()`
   - `is_equiv()`
3. Populate `self.questions` with `Question` objects
4. Add a quick test under `if __name__ == "__main__":`

Example skeleton:
```python
from .base import Benchmark, Question

class NewBenchmark(Benchmark):
    def __init__(self, split="test", n=None):
        super().__init__("newbench")
        self.load_data(split, n)

    def load_data(self, split, n=None):
        self.questions = [
            Question(qid="1", question="...", gold="...")
        ]

    def normalize(self, text: str) -> str:
        return text.strip()

    def is_equiv(self, gold: str, pred: str) -> bool:
        return self.normalize(gold) == self.normalize(pred)
```

---

✅ With this structure, all benchmarks behave consistently, and new ones can be added easily.
