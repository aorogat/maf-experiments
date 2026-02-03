[![Python 3.12.3](https://img.shields.io/badge/python-3.12.3-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/status-research-orange.svg)]()

![MASBench](slogan.png)

MASBench is a unified benchmark suite for systematically analyzing architectural design choices in LLM-based agent frameworks — spanning orchestration, memory, planning, specialization, and multi-agent coordination under controlled execution. The suite isolates framework-level effects from model capabilities and task complexity, enabling controlled evaluation of single-agent and multi-agent architectural design choices.

## Why MASBench?

Existing benchmarks primarily test isolated agent capabilities (reasoning, tool use, memory) without addressing how framework architecture governs performance and scalability. MASBench fills this gap by:

- Providing controlled evaluation of architectural design decisions under fixed models and tasks
- Isolating framework-level effects from model capabilities and task complexity
- Enabling systematic comparison across orchestration patterns, memory architectures, and coordination mechanisms
- Supporting reproducible analysis of scalability and resource utilization characteristics

## How MASBench Works

- **Unified execution pipeline**: Standardized interfaces normalize execution across diverse frameworks
- **Standardized configuration & logging**: Consistent measurement and artifact collection
- **Controlled architectural isolation**: Framework behavior evaluated independently of model and task variations
- **Cost-aware backend routing**: Abstracted LLM backends support efficient, framework-agnostic evaluation

## Architectural Taxonomy

MASBench organizes frameworks along three primary paradigms:

- **Graph-based orchestration**: Workflows modeled as directed graphs with nodes representing computational steps and edges defining control flow
- **Role-based agent systems**: Agents structured around specialized roles with coordination mechanisms routing tasks based on role assignments
- **Environment/simulation-mediated systems**: Agents situated within shared environments where interaction occurs through state and action interfaces

The suite evaluates key architectural dimensions:

- **Orchestration & control flow**: How frameworks structure task execution and manage dependencies
- **Memory architecture**: Long-term retention, learning, and forgetting mechanisms
- **Planning interfaces**: Multi-step reasoning under framework constraints
- **Specialization mechanisms**: Role assignment, task routing, and capability distribution
- **Communication topology & coordination**: Information flow patterns, coordination mechanisms, and topology-induced interaction patterns in multi-agent settings

## Benchmark Modules

### Single-Agent Evaluation

- [Memory](single_agent/memory/readme.md) — Long-term retention, learning, forgetting
- [Planning](single_agent/reasoning/README.md) — Multi-step reasoning under interface constraints
- [Specialization](single_agent/specialization/readme.md) — Role assignment and capability distribution
- [Framework Overhead](single_agent/framework_overhead/README.md) — Orchestration and execution efficiency
- [Tool Use](single_agent/tool_use/README.md) — Architectural integration patterns

### Multi-Agent Evaluation

- [Coordination & Topology](multi_agent/topology/README.md) — Communication patterns and coordination outcomes

## Reproducibility & Artifacts

MASBench enforces reproducibility through:

- Fixed Python version (3.12.3) and pinned dependencies (`requirements.lock`)
- Unified execution pipeline with standardized configuration and logging
- Backend abstraction supporting cost-aware, framework-agnostic evaluation
- Experimental results preserved in `results/` for transparency

Analysis and interpretation are documented within individual experiment directories and associated publications.

## Citation

If you use MASBench in academic work, please cite:

```bibtex
@article{orogat2026mafbench,
  title={Understanding Multi-Agent LLM Frameworks: A Unified Benchmark and Experimental Analysis},
  author={Orogat, Abdelghny and Rostam, Ana and Mansour, Essam},
  journal={arXiv preprint arXiv:submit/7225627},
  year={2026}
}
```

## Contact

Abdelghny Orogat — Concordia University  
Email: Abdelghny.Orogat@concordia.ca
