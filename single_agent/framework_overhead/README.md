# Framework Overhead Experiment

This experiment measures the **baseline orchestration overhead** introduced by different multi-agent frameworks (CrewAI, LangGraph, Concordia) compared to a direct LLM call.  

The setup uses a trivial prompt (`"What is 2+2?"`) and disables all optional features (no memory, no planning, no tools). This isolates **framework cost** from model behavior.

---

## 📂 Structure

```
single_agent/framework_overhead/
├── direct_llm.py        # Direct Ollama LLM baseline runner
├── crewai_runner.py     # CrewAI runner (1 agent, 1 task, no planning/memory)
├── langgraph_runner.py  # LangGraph runner (single-node graph)
├── concordia_runner.py  # Concordia runner (pass-through Game Master)
└── run_overhead.py      # Master script: runs all frameworks & saves results
```

---

## ⚙️ How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Requires `ollama` running locally, plus CrewAI, LangGraph, and Concordia.)

2. Run the full overhead experiment:
   ```bash
   python -m single_agent.framework_overhead.run_overhead
   ```

3. Results will be saved as JSON in:
   ```
   results/framework_overhead/framework_overhead.json
   ```

---

## 📊 Metrics

For each framework, we report:

- **Latency (p50 / p95)**  
  - Time (ms) to return the first token.  
  - p50 = median, p95 = 95th percentile.  

- **Throughput (req/s)**  
  - Requests completed per second with concurrency (default 4 workers).  

---

## 🔍 Example Output

```bash
=== Running Direct LLM for 20 trials ===
Direct LLM: p50=22.13 ms | p95=34.88 ms | Throughput=72.41 req/s

=== Running CrewAI for 20 trials ===
CrewAI: p50=52.41 ms | p95=75.23 ms | Throughput=48.31 req/s

=== Running LangGraph for 20 trials ===
LangGraph: p50=60.22 ms | p95=88.55 ms | Throughput=46.17 req/s

=== Running Concordia for 20 trials ===
Concordia: p50=71.93 ms | p95=102.11 ms | Throughput=40.92 req/s
```

Saved in:
```
results/framework_overhead/framework_overhead.json
```

## 📝 Notes
- This experiment isolates framework **overhead only**.  
- All advanced features are **disabled**.  
- Default: 20 trials × concurrency=4. Increase to 1000 trials for final results.  
