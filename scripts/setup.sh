#!/usr/bin/env bash
set -euo pipefail

echo "📂 Creating folders..."
mkdir -p   single-agent/tool-use   single-agent/memory   single-agent/reasoning   multi-agent/communication   multi-agent/coordination   multi-agent/environment   multi-agent/topology   configs   data   logs   models   results

# Python package init files (optional, helpful for imports)
touch single-agent/__init__.py
touch single-agent/tool-use/__init__.py
touch single-agent/memory/__init__.py
touch single-agent/reasoning/__init__.py
touch multi-agent/__init__.py
touch multi-agent/communication/__init__.py
touch multi-agent/coordination/__init__.py
touch multi-agent/environment/__init__.py
touch multi-agent/topology/__init__.py

# Requirements
cat > requirements.txt << 'REQ'
langgraph
crewai
gdm-concordia
REQ

# .gitignore
cat > .gitignore << 'IGN'
.venv/
__pycache__/
*.pyc
.env
logs/
results/
IGN

# Placeholder runners
cat > scripts/run_single_agent_tool_use.py << 'PY'
print("Run: single-agent tool-use experiment (placeholder).")
PY

cat > scripts/run_multi_agent_coordination.py << 'PY'
print("Run: multi-agent coordination experiment (placeholder).")
PY

echo "✅ Structure created. Next:"
echo "   python3 -m venv .venv && source .venv/bin/activate"
echo "   pip install -U pip"
echo "   pip install -r requirements.txt || (sed -i 's/gdm-concordia/concordia/' requirements.txt && pip install -r requirements.txt)"
