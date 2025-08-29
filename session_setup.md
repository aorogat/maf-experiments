For remote push
```bash
git_token: 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
```


```bash
Chatgpt: https://chatgpt.com/c/68b1a74b-7cec-8325-8b5b-d4c6a82f93c3?model=gpt-5-instant
```



# ⚙️ Setting Up WSL, Python Environment, and Running Multi-Agent Framework Experiments on Windows

This guide walks you through setting up the **MAF Experiments** project inside **WSL (Windows Subsystem for Linux)**. You’ll create a Python virtual environment, install required frameworks (**LangGraph**, **CrewAI**, **Concordia**), and prepare for running single-agent and multi-agent experiments.

---

## 1) Install and Launch WSL (Ubuntu)

If you haven’t already, install WSL from a Windows terminal (PowerShell or CMD):

```bash
wsl --install
```

Then launch Ubuntu:

```bash
wsl.exe -d Ubuntu
```

---

## 2) Create the Project Folder (WSL)

```bash
mkdir -p ~/maf-experiments
cd ~/maf-experiments
```

Optionally initialize a Git repo or clone your own:

```bash
git init
# or
# git clone https://github.com/your-repo/maf-experiments.git .
```


## 3) Python Environment Setup

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

## 4) Create Project Structure (one-time)

Run the setup script to scaffold folders and baseline files:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

This creates:

```
maf-experiments/
├── configs/
├── data/
├── logs/
├── models/
├── multi-agent/
│   ├── communication/
│   │   ├── langgraph_test.py
│   │   ├── crewai_test.py
│   │   ├── concordia_test.py
│   │   └── README.md
│   ├── coordination/
│   │   ├── langgraph_test.py
│   │   ├── crewai_test.py
│   │   ├── concordia_test.py
│   │   └── README.md
│   ├── environment/
│   │   ├── langgraph_test.py
│   │   ├── crewai_test.py
│   │   ├── concordia_test.py
│   │   └── README.md
│   └── topology/
│       ├── langgraph_test.py
│       ├── crewai_test.py
│       ├── concordia_test.py
│       └── README.md
├── results/
├── scripts/
│   ├── setup.sh
│   ├── run_multi_agent_coordination.py
│   └── run_single_agent_tool_use.py
├── single-agent/
│   ├── memory/
│   │   ├── langgraph_test.py
│   │   ├── crewai_test.py
│   │   ├── concordia_test.py
│   │   └── README.md
│   ├── reasoning/
│   │   ├── langgraph_test.py
│   │   ├── crewai_test.py
│   │   ├── concordia_test.py
│   │   └── README.md
│   └── tool-use/
│       ├── langgraph_test.py
│       ├── crewai_test.py
│       ├── concordia_test.py
│       └── README.md
├── requirements.txt
├── .gitignore
└── README.md
```

Each **experiment subfolder** contains:
- `langgraph_test.py` → experiment using LangGraph
- `crewai_test.py` → experiment using CrewAI
- `concordia_test.py` → experiment using Concordia
- `README.md` → description of the experiment setup

---

## 5) Install Dependencies

With `(mafenv)` active:

```bash
pip install -U pip
pip install -r requirements.txt || (sed -i 's/gdm-concordia/concordia/' requirements.txt && pip install -r requirements.txt)
```

The repo uses:

- `langgraph` — orchestration framework
- `crewai` — agent/collaboration toolkit
- `gdm-concordia` — Concordia framework (falls back to `concordia` if needed)

---

## 6) Run Example Placeholders

```bash
source mafenv/bin/activate
python scripts/run_single_agent_tool_use.py
python scripts/run_multi_agent_coordination.py
```

Outputs/logs should appear under `results/` and `logs/` as you add real experiments.

---

## 7) (Optional) Local LLMs via Ollama inside WSL

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run deepseek-llm:7b
```

To test the API quickly:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "deepseek-llm:7b",
  "prompt": "Summarize how our multi-agent experiment is orchestrated."
}'
```

---

## 8) Tips

- Add new experiments under `single-agent/` or `multi-agent/` and corresponding runners in `scripts/`.
- Version configs and requirements in Git for reproducibility.
- Keep outputs in `results/` and logs in `logs/`.

**You’re ready to build and compare Single-Agent and Multi-Agent experiments in WSL!**
