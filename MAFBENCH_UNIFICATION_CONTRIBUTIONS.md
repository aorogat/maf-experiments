# MAFBench Unification Contributions

This document systematically identifies, extracts, and summarizes all infrastructure, interface, and execution-layer contributions introduced by MAFBench that unify heterogeneous benchmarks under a standardized evaluation pipeline.

---

## Core Infrastructure

### 1. Unified Backend Routing & Model Abstraction

**File**: `single_agent/memory/router_groq.py`, `single_agent/memory/router_local.py`

**Contribution**: MAFBench introduces transparent API routing layers that decouple frameworks from provider-specific APIs, enabling cost-aware execution and framework-agnostic evaluation.

**Key Mechanisms**:
- **OpenAI-compatible proxy servers** that intercept framework API calls
- **Model mapping layer** (`MODEL_MAP`) that translates OpenAI model names to alternative providers (e.g., `gpt-4o-mini` → `openai/gpt-oss-20b` on Groq)
- **Request normalization** that maintains OpenAI API schema while routing to different backends
- **Embedding routing** that intelligently routes embeddings to appropriate providers (Groq doesn't support embeddings, so they route to OpenAI)
- **Streaming support** for frameworks that require SSE (Server-Sent Events) responses

**Code Snippet**:
```python
# single_agent/memory/router_groq.py:309-322
MODEL_MAP = {
    "gpt-4o": "openai/gpt-oss-20b",
    "gpt-4o-mini": "openai/gpt-oss-20b",
    "gpt-4o-mini-high": "openai/gpt-oss-20b",
    "gpt-3.5-turbo": "openai/gpt-oss-20b",
}

def map_model(name: str | None) -> str:
    if not name:
        return DEFAULT_MODEL
    return MODEL_MAP.get(name, DEFAULT_MODEL)
```

**Why this didn't exist originally**: Original benchmarks assume direct API access to specific providers. MAFBench's router enables switching between OpenAI, Groq, and local models without modifying framework code, reducing costs by 10× while maintaining compatibility.

---

### 2. Centralized Configuration System

**File**: `single_agent/memory/config.py`, `single_agent/reasoning/config.py`

**Contribution**: MAFBench introduces unified configuration modules that enforce consistent parameters across all frameworks, ensuring fair comparison and reproducibility.

**Key Mechanisms**:
- **Shared LLM parameters** (temperature, max_tokens) across all frameworks
- **Benchmark-specific limits** (e.g., `max_sessions_per_subtask`, `max_questions_per_session`) for cost control
- **Memory chunking parameters** (chunk_max_tokens, chunk_overlap) for consistent preprocessing
- **Evaluation model configuration** (eval_llm_model, batch sizes) for standardized scoring

**Code Snippet**:
```python
# single_agent/memory/config.py:19-48
llm_max_tokens = 1500
llm_temperature = 0.1
max_sessions_per_subtask = 10
eval_llm_model = "gpt-4o-mini"
eval_small_batch_size = 10
chunk_max_tokens = 4096
chunk_overlap = 200
```

**Why this didn't exist originally**: Each benchmark had its own configuration scattered across files. MAFBench centralizes configuration to ensure identical experimental conditions across frameworks.

---

### 3. Unified Result Aggregation & Logging

**File**: `multi_agent/topology/results.py`, `single_agent/memory/benchmark/memory_agent_bench.py`

**Contribution**: MAFBench standardizes result collection, storage, and aggregation across all benchmarks, enabling cross-framework comparison.

**Key Mechanisms**:
- **Consistent result schema** (score, runtime, token usage, error messages, commit hash)
- **Automatic result aggregation** from multiple experiment runs
- **Git commit tracking** for reproducibility
- **Structured JSON output** with per-question, per-session, and aggregate scores

**Why this didn't exist originally**: Original benchmarks used ad-hoc result formats. MAFBench enforces a unified schema that enables systematic analysis.

---

## Memory Benchmarks

### 1. Unified Interface Abstraction (`reset()`, `ingest()`, `query()`)

**File**: `single_agent/memory/benchmark/memory_agent_bench.py`, `single_agent/memory/readme.md`

**Contribution**: MAFBench enforces a standardized three-method interface (`reset()`, `ingest()`, `query()`) that all memory-enabled frameworks must implement, enabling framework-agnostic evaluation.

**Original Benchmark**: MemoryAgentBench (HuggingFace dataset) provides sessions with context and questions, but no standardized agent interface.

**MAFBench Addition**:
- **Interface specification** that abstracts away framework-specific memory implementations
- **Session controller** (`evaluate_agent()`) that orchestrates the reset-ingest-query cycle
- **Framework adapters** (CrewAI, OpenAI SDK, Agno, LangGraph) that wrap framework-specific APIs to match the interface

**Code Snippet**:
```python
# single_agent/memory/benchmark/memory_agent_bench.py:203-214
agent.reset()
if not ignore_ingest:
    agent.ingest(sess["context"])

preds = []
for idx, q in enumerate(sess["questions"], start=1):
    if max_questions_per_session and idx > max_questions_per_session:
        break
    preds.append(agent.query(q))
```

**Why this didn't exist originally**: MemoryAgentBench expects direct dataset access. MAFBench's interface allows any framework to be evaluated without modifying the benchmark code.

---

### 2. Unified Scoring Wrapper (LLM-Based Semantic Evaluation)

**File**: `single_agent/memory/benchmark/metric_eval_gpt.py`

**Contribution**: MAFBench replaces string-matching evaluation with LLM-based semantic metrics that ensure fairness across different answer formats.

**Original Benchmark**: MemoryAgentBench uses string matching, which fails when frameworks produce differently formatted answers.

**MAFBench Addition**:
- **Semantic exact match** evaluation using GPT-4o-mini
- **Summary fact-level F1** scoring for long-form answers
- **Semantic Recall@5** for recommendation tasks
- **Batch evaluation** for cost efficiency (configurable batch sizes)
- **Normalized answer pair format** that handles both string and list responses

**Code Snippet**:
```python
# single_agent/memory/benchmark/metric_eval_gpt.py:50-64
def evaluate_exact_match(answers_pairs, model="gpt-4o-mini", batch_size=10):
    """Return list of 0/1 semantic exact-match judgments."""
    # Batch processing with semantic evaluation
    # Handles normalization of gold answers (removes duplicates, normalizes meaning)
    # Evaluates system answers for semantic equivalence, not literal matching
```

**Why this didn't exist originally**: Original benchmarks used exact string matching, which penalizes frameworks that format answers differently. MAFBench's semantic evaluation ensures fair comparison.

---

### 3. Session Length Normalization & Cost Control

**File**: `single_agent/memory/config.py`, `single_agent/memory/benchmark/memory_agent_bench.py`

**Contribution**: MAFBench introduces per-subtask session limits and question limits to control evaluation cost while maintaining statistical validity.

**Original Benchmark**: MemoryAgentBench evaluates all sessions, which can cost thousands of dollars.

**MAFBench Addition**:
- **Per-subtask session limiting** (`max_sessions_per_subtask = 10`) that ensures balanced evaluation across subtasks
- **Per-session question limiting** (`max_questions_per_session`) for debugging and cost control
- **Subtask-aware counting** that prevents over-sampling of easy subtasks

**Code Snippet**:
```python
# single_agent/memory/benchmark/memory_agent_bench.py:194-200
subtask = sess["subtask"]
if subtask_session_count[subtask] >= max_sessions_per_subtask:
    continue
subtask_session_count[subtask] += 1
```

**Why this didn't exist originally**: Original benchmark runs all sessions without cost control. MAFBench enables affordable evaluation while maintaining coverage across all subtasks.

---

## Planning Benchmarks

### 1. Planning Stage Injection & Schema Control

**File**: `single_agent/reasoning/config.py`, `single_agent/reasoning/README.md`

**Contribution**: MAFBench introduces controlled planning mode toggles that isolate the effect of planning interface design (schema-constrained vs. free-form) from model capabilities.

**Original Benchmarks**: GSM8K, CSQA, MATH-100 are reasoning benchmarks without planning mechanisms.

**MAFBench Addition**:
- **Planning toggle** (`planning: True/False`) that enables/disables framework-enforced planning
- **Schema-constrained planning** (Crew-Plan) that requires rigid two-stage plan format
- **Free-form planning** (Direct-LLM-Plan) that allows natural plan generation
- **Model decoupling** (separate `llm`, `planning_llm`, `math_judge_llm`) to isolate planning effects

**Code Snippet**:
```python
# single_agent/reasoning/config.py:1-9
CONFIG = {
    "planning": False,      # toggle planning
    "llm": "gpt-4o-mini",              # Backbone inference model
    "planning_llm": "gpt-4o-mini",     # Planning model (used when planning=True)
    "math_judge_llm": "gpt-4o-mini",   # Mathematical judgment model
}
```

**Why this didn't exist originally**: Original benchmarks evaluate direct question-to-answer generation. MAFBench adds planning as a controlled variable to study its impact.

---

### 2. Unified Benchmark Loading & Execution

**File**: `single_agent/reasoning/crewai_test.py`, `single_agent/reasoning/direct_llm_planning_test.py`

**Contribution**: MAFBench provides unified benchmark classes that load GSM8K, CSQA, and MATH-100 with consistent interfaces, enabling framework-agnostic evaluation.

**Original Benchmarks**: Each benchmark has its own loading mechanism and format.

**MAFBench Addition**:
- **Unified benchmark loader** that normalizes dataset formats
- **Consistent execution pipeline** (load → run → evaluate → save)
- **Formatting failure tracking** to distinguish parsing errors from reasoning failures
- **Token usage tracking** for cost analysis

**Why this didn't exist originally**: Benchmarks were evaluated independently. MAFBench unifies them under a single execution pipeline.

---

## Specialization Benchmarks

### 1. Conditioning Strategies Controller

**File**: `single_agent/specialization/crew.py`, `single_agent/specialization/readme.md`

**Contribution**: MAFBench introduces controlled conditioning strategies (role-based, planning-based, expert-guided) that isolate the effect of textual conditioning on agent behavior.

**Original Benchmark**: ML datasets (Utility, Wifi, EU-IT, Yelp, Volkert) from CatDB/OpenML without conditioning mechanisms.

**MAFBench Addition**:
- **Role-based conditioning** via CrewAI agent configurations (`agents.yaml`) with role-specific goals and backstories
- **Planning-based conditioning** via `planning=True` flag that adds explicit step-by-step planning
- **Expert-guided conditioning** via detailed task descriptions in `task.yaml` with methodological workflows
- **Controlled isolation** that holds model, data, and task fixed while varying only conditioning strategy

**Code Snippet**:
```python
# single_agent/specialization/crew.py:94-106
@crew
def crew(self) -> Crew:
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        process=Process.sequential,
        verbose=True,
        # process=Process.hierarchical,  # Alternative orchestration
    )
```

**Why this didn't exist originally**: Original datasets are just ML tasks. MAFBench adds conditioning as a controlled variable to study specialization effects.

---

### 2. Unified Code Generation Pipeline

**File**: `single_agent/specialization/readme.md`

**Contribution**: MAFBench standardizes code generation evaluation across all datasets with consistent naming conventions and execution requirements.

**MAFBench Addition**:
- **Naming conventions** (`{Dataset}_{role}.py`, `{Dataset}_{role}planning.py`, `{Dataset}_{role}exp.py`)
- **Unified task structure** (load CSV → preprocess → train → evaluate → output results)
- **Consistent evaluation metrics** (MAE for regression, Accuracy/Precision/Recall/F1 for classification)

**Why this didn't exist originally**: Each dataset was evaluated separately. MAFBench unifies them under a single evaluation protocol.

---

## Tool Use Benchmarks

### 1. Centralized Tool Selection System

**File**: `single_agent/tool_use/tool_selection/selector.py`, `single_agent/tool_use/TOOL_SELECTION_ARCHITECTURE.md`

**Contribution**: MAFBench introduces a centralized tool selection layer that runs before framework binding, ensuring all frameworks operate on the same tool set for fair comparison.

**Original Benchmark**: StableToolBench provides ~1500 tools but no tool selection mechanism.

**MAFBench Addition**:
- **LLM-based tool selection** that selects top-k most relevant tools per query
- **Caching system** that ensures reproducibility (same query → same tool set)
- **Tool budget control** (`max_tools = 120`) that respects LLM provider limits (OpenAI's 128-tool constraint)
- **Pre-framework execution** that selects tools before any framework binds them

**Code Snippet**:
```python
# single_agent/tool_use/tool_selection/selector.py:323-339
def select_tools(self, query: str, all_tools: List[BaseTool]) -> List[BaseTool]:
    # Check cache first
    cached_tool_names = self._load_from_cache(query)
    if cached_tool_names is not None:
        return self._load_tools_from_cache(cached_tool_names, all_tools)
    
    # If not in cache, use LLM to select
    selected_tool_names = self._select_tools_with_llm(query, tool_metadata)
    self._save_to_cache(query, selected_tool_names)
    return selected_tools
```

**Why this didn't exist originally**: Frameworks would select tools differently, making comparison unfair. MAFBench ensures identical tool sets across frameworks.

---

### 2. Unified Benchmark Runner & Evaluation Wrapper

**File**: `single_agent/tool_use/run_benchmark.py`, `single_agent/tool_use/evaluation/`

**Contribution**: MAFBench wraps StableToolBench's evaluation with a unified runner that handles tool selection, gold answer generation, and result aggregation.

**Original Benchmark**: StableToolBench requires manual setup and framework-specific integration.

**MAFBench Addition**:
- **Unified `run_benchmark()` function** that orchestrates the entire evaluation pipeline
- **Gold answer generator** that executes gold APIs via server to generate reference answers
- **Evaluation wrapper** (`StableToolBenchEvaluator`) that provides a clean interface to StableToolBench's evaluator
- **Result aggregation** that saves structured JSON results to `results/tools/`

**Code Snippet**:
```python
# single_agent/tool_use/run_benchmark.py:270-283
def run_benchmark(
    agent: Callable[[str], Dict[str, Any]],
    test_set: str = "G1_instruction",
    use_tool_selector: bool = True,
    tool_selector_model: str = "gpt-4o-mini",
    max_tools: int = 120,
    ...
) -> Dict[str, Any]:
    # 1. Load queries
    # 2. Initialize tool selector (if enabled)
    # 3. For each query: select tools → bind to agent → run → evaluate
    # 4. Aggregate and return results
```

**Why this didn't exist originally**: StableToolBench requires framework-specific integration. MAFBench provides a unified interface that any framework can use.

---

### 3. Tool Budget Controls & Prefilter Layer

**File**: `single_agent/tool_use/run_benchmark.py`, `single_agent/tool_use/tool_selection/selector.py`

**Contribution**: MAFBench enforces tool budget limits (max_tools) and provides a prefilter layer that reduces the tool space before framework binding.

**MAFBench Addition**:
- **Tool budget enforcement** that limits tools to 120 (under OpenAI's 128 limit)
- **Prefilter layer** that selects relevant tools before frameworks see them
- **Scalability testing** that measures framework performance with varying tool set sizes

**Why this didn't exist originally**: Frameworks would handle tool filtering differently. MAFBench standardizes tool budgets for fair comparison.

---

## Coordination Benchmarks

### 1. Topology Rewriting Engine

**File**: `multi_agent/topology/graph_builder.py`, `multi_agent/topology/frameworks/crewai_runner.py`

**Contribution**: MAFBench introduces a topology rewriting system that transforms communication graphs to simulate different orchestration patterns (sequential, hierarchical, fully-connected) while maintaining the same node set.

**Original Benchmark**: AgentsNet provides graphs from HuggingFace dataset but no topology rewriting.

**MAFBench Addition**:
- **Sequential topology** (`build_graph_from_hf_sequential`) that rewires graphs into path structures (chain of agents)
- **Hierarchical topology** (`build_graph_from_hf_hierarchical`) that rewires graphs into balanced 4-ary trees
- **Fully-connected topology** (`build_graph_from_hf_all_connected`) that simulates Concordia's relay hub behavior
- **Graph abstraction layer** (`get_graph()`) that supports multiple graph sources (HF, framework-native)

**Code Snippet**:
```python
# multi_agent/topology/graph_builder.py:116-131
def build_graph_from_hf_sequential(graph_model: str, graph_size: int, num_sample: int) -> nx.Graph:
    G = build_graph_from_hf(graph_model, graph_size, num_sample)
    # Remove all edges
    G.remove_edges_from(list(G.edges()))
    # Add path edges
    for i in range(graph_size - 1):
        G.add_edge(i, i + 1)
    return G
```

**Why this didn't exist originally**: Original benchmark uses fixed graph topologies. MAFBench enables controlled topology variation to study orchestration effects.

---

### 2. Synchronous Protocol Runner

**File**: `multi_agent/topology/frameworks/langgraph_runner.py`, `multi_agent/topology/frameworks/crewai_runner.py`

**Contribution**: MAFBench provides framework-specific runners that execute the same tasks under different orchestration protocols, enabling fair comparison.

**Original Benchmark**: AgentsNet provides `LiteralMessagePassing` framework but only one execution mode.

**MAFBench Addition**:
- **LangGraph runner** that uses LangGraph's `StateGraph` for agent coordination
- **CrewAI runner** that simulates sequential/hierarchical orchestration patterns
- **Concordia runner** that simulates relay hub behavior (fully-connected)
- **Unified execution interface** that maintains consistent result schema across frameworks

**Code Snippet**:
```python
# multi_agent/topology/frameworks/crewai_runner.py:45-67
async def run_framework(args, commit_hash):
    # Build CrewAI graph (sequential or hierarchical) from HF
    if args.framework == "sequential":
        graph = get_graph(source="hf_sequential", ...)
    elif args.framework == "hierarchical":
        graph = get_graph(source="hf_hierarchical", ...)
    # Execute with same task and model provider
    lmp_model = task_class(graph=graph, ...)
    await lmp_model.bootstrap()
    answers = await lmp_model.pass_messages()
```

**Why this didn't exist originally**: Original benchmark only supports one execution mode. MAFBench enables multi-framework comparison.

---

### 3. Batch Experiment Execution & Result Aggregation

**File**: `multi_agent/topology/run_all_experiments.py`, `multi_agent/contribution.md`

**Contribution**: MAFBench introduces automated batch execution that runs experiments across multiple tasks, frameworks, graph models, and sizes, with automatic result aggregation.

**Original Benchmark**: AgentsNet requires manual execution of individual experiments.

**MAFBench Addition**:
- **Automated experiment sweeps** that generate the product of tasks × frameworks × graph models × sizes
- **Asynchronous execution** using `asyncio` for efficient subprocess management
- **Result aggregation** that collects all JSON results into CSV/JSON summaries
- **Framework-aware execution** that handles framework-specific requirements

**Why this didn't exist originally**: Original benchmark requires manual setup for each experiment. MAFBench automates large-scale evaluation.

---

## Fairness & Standardization Mechanisms

### 1. Prompt Alignment & Normalization

**File**: `single_agent/memory/config.py`, `single_agent/reasoning/config.py`

**Contribution**: MAFBench enforces consistent prompt parameters (temperature, max_tokens) across all frameworks to ensure fair comparison.

**MAFBench Addition**:
- **Shared LLM configuration** that all frameworks read from centralized config files
- **Temperature normalization** (default: 0.1 for memory, 0.0 for tool selection)
- **Token limit enforcement** that prevents frameworks from using excessive context

**Why this didn't exist originally**: Each framework would use different parameters. MAFBench standardizes them.

---

### 2. Scoring Harmonization

**File**: `single_agent/memory/benchmark/metric_eval_gpt.py`, `single_agent/tool_use/run_benchmark.py`

**Contribution**: MAFBench uses consistent evaluation metrics and scoring functions across all benchmarks, ensuring comparable results.

**MAFBench Addition**:
- **Semantic evaluation** for memory benchmarks (instead of string matching)
- **API call scoring** for tool use benchmarks (proportion of gold APIs called)
- **Formatting failure tracking** for planning benchmarks (distinguishes parsing from reasoning failures)

**Why this didn't exist originally**: Each benchmark used different evaluation methods. MAFBench harmonizes them.

---

### 3. Reproducibility Controls

**File**: `multi_agent/topology/utils.py`, `single_agent/tool_use/tool_selection/selector.py`

**Contribution**: MAFBench enforces reproducibility through seed management, git commit tracking, and caching systems.

**MAFBench Addition**:
- **Git commit tracking** in all result files
- **Random seed management** for graph generation and experiment execution
- **Tool selection caching** that ensures identical tool sets across runs
- **Deterministic graph loading** from HuggingFace dataset

**Code Snippet**:
```python
# multi_agent/topology/frameworks/crewai_runner.py:47
random.seed(args.seed)

# single_agent/tool_use/tool_selection/selector.py:65-79
def _get_query_hash(self, query: str) -> str:
    cache_key = f"{query}|max_tools={self.max_tools}|model={self.model}"
    return hashlib.sha256(cache_key.encode('utf-8')).hexdigest()
```

**Why this didn't exist originally**: Original benchmarks didn't track reproducibility. MAFBench ensures experiments can be replicated.

---

## Summary: What MAFBench Engineered

MAFBench did not simply reuse existing benchmarks. Instead, it engineered a **framework-level evaluation system** that:

1. **Unified interfaces** across heterogeneous benchmarks (reset/ingest/query for memory, planning toggles for reasoning, tool selection for tool use, topology rewriting for coordination)

2. **Execution orchestration** that controls execution independently of benchmark assumptions (backend routing, configuration management, batch execution, result aggregation)

3. **Fairness mechanisms** that ensure cross-framework comparability (tool budgets, session limits, prompt alignment, scoring harmonization, reproducibility controls)

4. **Architectural isolation** that enables controlled experiments (planning variants, role modes, topology modes, conditioning strategies)

5. **Cost-aware execution** that makes evaluation affordable (backend routing to Groq, session limits, batch evaluation)

These contributions transform MAFBench from a collection of benchmarks into a **unified evaluation infrastructure** that enables systematic, fair, and reproducible comparison of agent frameworks.

