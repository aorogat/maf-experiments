# ⚙️ Setting Up Python Environment and Running Multi-Agent Framework Experiments

This guide walks you through setting up the **MAF Experiments** project. You’ll create a Python virtual environment, install required frameworks (**LangGraph**, **CrewAI**, **Concordia**), and configure LLM keys for experiments.

---

## 1) Get the Project

Clone the project from GitHub:

```bash
git clone https://github.com/aorogat/maf-experiments.git
cd maf-experiments
```

---

## 2) Python Environment Setup

Install Python and venv (once):

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
```

Create and activate a virtual environment **inside the project**:

```bash
python3 -m venv mafenv
source mafenv/bin/activate
```

You should see `(mafenv)` in your prompt.

---

## 3) Install Dependencies

With `(mafenv)` active:

```bash
pip install -U pip
pip install -r requirements.txt || (sed -i 's/gdm-concordia/concordia/' requirements.txt && pip install -r requirements.txt)
```

The repo uses:

- `langgraph` — orchestration framework  
- `crewai` — multi-agent collaboration toolkit  
- `gdm-concordia` — Concordia framework (falls back to `concordia` if needed)  
- `openai`, `anthropic`, `google-generativeai`, `ollama` — LLM connectors  
- `python-dotenv` — for reading `.env` API keys  

---

## 4) Configure LLM API Keys

All frameworks use the **centralized `llms/` module**. You don’t need to edit code to configure keys.  

Instead, create a `.env` file in the project root (`maf-experiments/.env`) with your credentials:

```bash
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-your-anthropic-key
GEMINI_API_KEY=sk-your-gemini-key
OLLAMA_MODEL=deepseek-llm:7b
```

- `OpenAILLM` reads from `OPENAI_API_KEY`  
- `LocalOllamaLLM` reads the default model from `OLLAMA_MODEL`  
- Other wrappers (Anthropic, Gemini) use their corresponding keys  

If a key is missing, the framework will raise a **clear error** instead of failing silently.

---

## 5) Project Structure

```
maf-experiments/
├── llms/                  # Shared LLM interfaces
│   ├── base_llm.py
│   ├── local_llm.py
│   ├── remote_llm.py
│   ├── test_llms.py
│   └── README.md
├── single_agent/          # Single-agent experiments
│   ├── memory/
│   ├── reasoning/
│   └── tool-use/
├── multi_agent/           # Multi-agent experiments
│   ├── communication/
│   ├── coordination/
│   ├── environment/
│   └── topology/
├── scripts/               # Setup and runners
├── configs/
├── data/
├── results/
├── logs/
├── requirements.txt
├── .env                   # API keys (ignored by Git)
└── README.md
```

Each experiment subfolder contains:
- `langgraph_test.py` → run experiment using LangGraph  
- `crewai_test.py` → run experiment using CrewAI  
- `concordia_test.py` → run experiment using Concordia  
- `README.md` → description of the experiment  

---

## 6) Run Tests

Activate your environment:

```bash
source mafenv/bin/activate
```

### Test LLM connectivity
```bash
python -m llms.test_llms
```

### Run placeholder experiments
```bash
python scripts/run_single_agent_tool_use.py
python scripts/run_multi_agent_coordination.py
```

Outputs/logs will appear under `results/` and `logs/`.

---

## 7) Optional: Local LLMs with Ollama

Install Ollama in your environment:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run deepseek-llm:7b
```

Update `.env` with your preferred local model:

```bash
OLLAMA_MODEL=deepseek-llm:7b
```

---

## ✅ Tips

- Add new experiments under `single-agent/` or `multi-agent/`.  
- All frameworks import from `llms/`, so you configure keys only once.  
- Track configs, requirements, and `.env.example` in Git for reproducibility (but **never commit `.env`**).  
