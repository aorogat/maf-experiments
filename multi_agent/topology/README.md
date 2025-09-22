# Network Topology Experiments

A scalability study examining how different network topologies affect multi-agent system performance across various distributed computing tasks.

## 🎯 Overview

This experiment evaluates multi-agent system performance across different network topologies and agent counts using the **AGENTSNET** benchmark. By systematically varying graph structures and team sizes, we measure how network connectivity patterns impact coordination efficiency and task success rates.

### Experimental Design

**Tasks Evaluated:**
- **Coloring**: Distributed graph coloring with conflict avoidance
- **Matching**: Maximum matching algorithm coordination  
- **VertexCover**: Collaborative minimum vertex cover finding
- **LeaderElection**: Distributed consensus for leader selection
- **Consensus**: Byzantine fault-tolerant agreement

**Network Topologies:**
- **Small-World (`ws`)**: High clustering + short paths (Watts-Strogatz)
- **Scale-Free (`ba`)**: Hub-dominated networks (Barabási-Albert) 
- **Delaunay (`dt`)**: Geometric proximity-based connections
- **Star (`star`)**: Centralized hub architecture

**Scaling Parameters:**
- Agent counts: 4, 8, 16 agents
- Communication rounds: 1-10 rounds (task-dependent)
- Multiple samples per configuration for statistical significance

## 🚀 Quick Start

From the project root (`maf-experiments/`):

```bash
# Activate environment
source mafenv/bin/activate

# Run complete topology experiment suite
python -m multi_agent.topology.run_all_experiments

# Or run individual configuration
python -m multi_agent.topology.runner \
  --task coloring \
  --graph_models ws \
  --graph_size 8 \
  --rounds 4 \
  --framework langgraph
```

## 📊 Results Structure

Results are saved in the project's standardized format:

```
results/
└── topology/
    ├── coloring_ws_n8_r4.json       # Individual experiment results
    ├── matching_ba_n16_r6.json
    ├── ...
    ├── scalability_summary.csv      # Aggregated analysis data
    └── scalability_summary.json
```

### Output Schema

Each experiment generates metrics following the project's result format:

| Field | Description | Type |
|-------|-------------|------|
| `task` | Algorithm being tested | string |
| `framework` | Network topology used | string |  
| `n` | Number of agents | integer |
| `rounds` | Communication rounds executed | integer |
| `score` | Task success metric (0.0-1.0) | float |
| `runtime` | Execution time (seconds) | float |
| `success` | Completion status | boolean |
| `error` | Error details if failed | string |

## 🔧 Configuration

### Command Line Options

| Parameter | Description | Options | Default |
|-----------|-------------|---------|---------|
| `--task` | Distributed algorithm | `coloring`, `matching`, `vertexcover`, `leaderelection`, `consensus` | Required |
| `--graph_models` | Network topology | `ws`, `ba`, `dt`, `star` | Required |
| `--graph_size` | Agent team size | `4`, `8`, `16` | Required |
| `--rounds` | Max communication rounds | Integer (1-20) | `4` |
| `--samples_per_graph_model` | Repetitions per config | Integer | `1` |
| `--framework` | Agent orchestration | `langgraph` | `langgraph` |
| `--model` | LLM backend | See project LLM config | `gpt-4o-mini` |

### LLM Configuration

Uses the project's centralized LLM configuration from `llms/`. Ensure your `.env` file contains:

```bash
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-your-anthropic-key
```

## 📈 Expected Outcomes

This experiment investigates several key hypotheses:

**Topology Impact:**
- **Small-world networks** should excel at tasks requiring both local clustering and global coordination
- **Scale-free networks** may suffer from hub bottlenecks but benefit from efficient broadcast
- **Star topologies** should show fast consensus but potential central point failures
- **Delaunay graphs** should demonstrate balanced communication costs

**Scalability Patterns:**
- Performance degradation rates as agent count increases
- Communication round requirements across topologies
- Framework overhead comparison

## 🔬 Implementation Notes

### Framework Integration

- **LangGraph**: Full implementation with custom graph builders for each topology
- **CrewAI**: Planned integration (sequential and hierarchical patterns)
- **Concordia**: Simulated using LangGraph star topology

### Topology Generation

Each network topology is generated using established algorithms:
- `ws`: NetworkX Watts-Strogatz with rewiring probability 0.3
- `ba`: NetworkX Barabási-Albert with preferential attachment  
- `dt`: Delaunay triangulation of random point distributions
- `star`: Central hub connected to all peripheral nodes

### Task Implementations

All tasks follow the same interface pattern:
1. **Initialization**: Each agent receives local problem state
2. **Communication**: Structured message passing per topology
3. **Coordination**: Distributed algorithm execution
4. **Evaluation**: Global solution quality assessment

## 📊 Analysis Examples

### Basic Performance Comparison

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load aggregated results  
df = pd.read_csv('results/topology/scalability_summary.csv')

# Performance by topology
df.groupby(['framework', 'task'])['score'].mean().unstack().plot(kind='bar')
plt.title('Task Success Rate by Network Topology')
plt.ylabel('Success Rate')
plt.show()
```

### Scalability Analysis

```python
# Runtime scaling
for task in df['task'].unique():
    task_data = df[df['task'] == task]
    task_data.pivot(index='n', columns='framework', values='runtime').plot()
    plt.title(f'{task.title()} Runtime Scaling')
    plt.xlabel('Number of Agents')
    plt.ylabel('Runtime (seconds)')
    plt.show()
```

## 🚧 Current Status

- ✅ LangGraph implementation complete
- ✅ All topology generators working
- ✅ Result aggregation and analysis
- 🚧 CrewAI integration in progress
- 📋 Advanced network metrics planned

## 📚 Related Work

This experiment extends research in:
- Multi-agent coordination algorithms
- Network topology effects on distributed computing
- Scalability analysis of LLM-based agent systems
- Graph-theoretic approaches to multi-agent communication

## 🤝 Contributing

This experiment is part of the larger MAF experiments project. To contribute:

1. Follow the project's contribution guidelines in the main README
2. Test new topologies using the existing framework interface
3. Add new distributed algorithms following the task template
4. Ensure results follow the standardized output format

---

*Part of the [MAF Experiments](https://github.com/aorogat/maf-experiments) project - a comprehensive evaluation of multi-agent framework capabilities.*