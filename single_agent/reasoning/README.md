# Planning Evaluation Experiment

This experiment evaluates the impact of planning mechanisms on reasoning task performance, comparing schema-constrained framework planning (CrewAI) against free-form direct LLM planning and baseline non-planning approaches. The evaluation isolates planning as a capability independent of model size, instruction tuning, or prompting style, enabling systematic comparison of structured versus unstructured plan generation.

## Research Question

How do different planning interface designs affect reasoning accuracy, robustness, and runtime? Specifically, does schema-constrained planning (as implemented by frameworks like CrewAI) provide benefits over free-form planning, or does the rigid format requirement introduce fragility that outweighs potential gains?

## Experimental Design

The experiment uses three established reasoning benchmarks that collectively capture different facets of planning:

- **GSM8K**: Short-chain arithmetic reasoning requiring multi-step numerical computation, making it sensitive to whether plan generation meaningfully scaffolds the computation
- **CSQA**: Multiple-choice commonsense inference emphasizing symbolic elimination and structured choice among alternatives, enabling analysis of how planning influences multi-step decision-making
- **MATH-100**: A carefully selected 100-sample subset of competition-level mathematics problems requiring complex algebraic and geometric reasoning, where multi-step symbolic manipulation is required and deviations from expected plan structures can cause extraction failures

Each benchmark is evaluated under three execution modes:

1. **NoPlan**: Direct question-to-answer generation without intermediate planning (baseline)
2. **Crew-Plan**: Framework-enforced rigid two-stage planning mechanism with schema-constrained plan that must be parsed and executed by the orchestration layer
3. **Direct-LLM-Plan**: Free-form unconstrained plan generation where the model generates a plan naturally before producing the final answer, without framework-imposed schema requirements

This design isolates the effect of planning interface design rather than model capability, as all three modes use the same underlying model and decoding configuration.

## Key Findings

Empirical evaluation reveals consistent patterns across models and datasets:

- **Crew-Plan frequently reduces accuracy**, sometimes sharply, particularly for smaller and medium-sized models. Schema-constrained planning forces models to obey a format they may not reliably reproduce, leading to systematic failures when the model deviates from the required schema.

- **Direct-LLM-Plan often preserves or improves accuracy** relative to NoPlan. Free-form planning allows models to express intermediate reasoning naturally without syntactic constraints.

- **Crew-Plan exhibits extremely high formatting failure rates** for local models (often exceeding 50-80% depending on model and dataset), while Direct-LLM-Plan maintains 0% failure rates. These failures stem primarily from parsing and orchestration errors rather than limitations in the underlying model's reasoning capabilities.

- **Crew-Plan introduces substantial computational overhead**, with runtime multipliers ranging from 2× to over 33× depending on model and dataset. Direct-LLM-Plan introduces only modest overhead, generally between 1.2× and 3×.

These findings suggest that effective planning interfaces should minimize syntactic constraints and allow models to express intermediate reasoning naturally. The success of planning depends not simply on whether a model is encouraged to plan, but on how the planning process is orchestrated.

## Setup

### Prerequisites

Ensure you have the MASBench environment configured with Python 3.12.3 and dependencies from `requirements.lock`. The experiment requires:

- CrewAI framework (for framework planning condition)
- OpenAI API access or Ollama (for local models)
- Benchmark datasets in `data/` directory

### Configuration

Planning evaluation introduces additional computational overhead beyond direct answer generation, as planning-enabled executions require explicit intermediate reasoning steps and, in some cases, separate judging models for structured or numerical correctness. The configuration system decouples three model roles to enable systematic analysis:

- **Backbone inference model** (`llm`): Primary model used for answer generation
- **Planning model** (`planning_llm`): Auxiliary model invoked to generate intermediate decompositions (used when planning is enabled)
- **Mathematical judgment model** (`math_judge_llm`): Specialized model for evaluating mathematical correctness in benchmarks like MATH, where surface-form matching alone is insufficient

This separation enables systematic analysis of whether gains arise from improved planning quality, stronger base reasoning, or more reliable evaluation, while avoiding confounding effects caused by monolithic model choices.

Edit `config.py` to set:

- **Planning toggle**: Set `planning` to `True` or `False` to enable/disable CrewAI's schema-constrained planning mechanism. When enabled, the framework enforces a rigid two-stage planning process.
- **Model decoupling**: Configure `llm`, `planning_llm`, and `math_judge_llm` independently to assess the contribution of each component
- **Benchmark selection**: Specify which benchmarks to run in `benchmarks` list
- **Subset sizes**: Control dataset size with `n_gsm8k`, `n_csqa`, `n_math` (set to `None` for full datasets). The MATH benchmark uses a 100-sample subset by default for cost-aware evaluation.

Example configuration:

```python
CONFIG = {
    "planning": False,  # Toggle: True enables Crew-Plan, False uses NoPlan
    "llm": "gpt-4o-mini",              # Backbone inference model
    "planning_llm": "gpt-4o-mini",     # Planning model (used when planning=True)
    "math_judge_llm": "gpt-4o-mini",   # Mathematical judgment model
    "results_dir": "results/planning",
    "benchmarks": ["csqa", "math", "gsm8k"],
    "n_gsm8k": None,   # Full dataset
    "n_csqa": None,    # Full dataset
    "n_math": 100,     # 100-sample subset (as used in evaluation)
}
```

### Agent and Task Definitions

CrewAI agent and task configurations are defined in `config/`:

- `agents.yaml`: Defines the reasoning agent role, goal, and backstory
- `tasks.yaml`: Specifies task descriptions and expected output formats for each benchmark

These YAML files are loaded by the CrewAI framework classes (`crew_gsm8k.py`, `crew_csqa.py`, `crew_math.py`).

## Execution

### Running CrewAI Experiments (Crew-Plan and NoPlan)

The main runner executes both planning and non-planning modes sequentially:

```bash
python -m single_agent.reasoning.crewai_test
```

This script:
1. Loads benchmarks using the unified benchmark classes from `benchmarks/`
2. Instantiates CrewAI crews for each benchmark
3. Runs questions through the framework with planning enabled/disabled based on `CONFIG["planning"]`
4. When planning is enabled, CrewAI enforces a schema-constrained plan format that must be parsed by the orchestration layer
5. Records predictions, correctness, timing, token usage, and formatting failures
6. Saves results to `results/planning/` with filenames like `crewai_csqa_planning_gpt-4o-mini.json` (planning mode) or `crewai_csqa_noplanning_gpt-4o-mini.json` (no planning mode)

### Running Direct LLM Planning (Direct-LLM-Plan)

To evaluate free-form planning without framework-imposed schema constraints:

```bash
python -m single_agent.reasoning.direct_llm_planning_test
```

This script implements unconstrained plan generation:
1. Generates a plan via direct LLM call with a simple prompt requesting step-by-step reasoning
2. Trims plans exceeding character limits (default 1500 chars) to prevent context overflow
3. Sends plan + question to LLM for final answer generation
4. Applies lightweight post-processing for benchmark-specific formats (e.g., single letter extraction for CSQA, numeric extraction for GSM8K)
5. Saves results to `results/planning_direct/` with filenames like `direct_planning_csqa_planning_gpt-4o-mini.json`

Unlike Crew-Plan, Direct-LLM-Plan does not enforce rigid schema requirements, allowing the model to express plans naturally. This implementation supports both OpenAI API and Ollama local models via the `call_llm()` function.

### Result Correction Scripts

If evaluation logic is updated, re-evaluate existing results:

```bash
# Re-evaluate GSM8K results with improved normalization
python -m single_agent.reasoning.correct_gsm8k --dir results/planning

# Re-evaluate CSQA results
python -m single_agent.reasoning.correct_csqa --dir results/planning
```

These scripts preserve original predictions in `pred_original` while updating normalized predictions and correctness flags.

## Analysis

### Comparing Planning Modes

Generate difference analysis between planning modes:

```bash
python -m single_agent.reasoning.results_analysis
```

This script performs two types of comparisons:
1. **Crew-Plan vs. NoPlan**: Pairs planning and non-planning CrewAI result files by benchmark and model, identifying questions where correctness differs between conditions
2. **Direct-LLM-Plan vs. NoPlan**: Compares direct LLM planning against non-planning CrewAI runs to assess whether free-form planning provides benefits over direct answer generation

The analysis saves detailed comparisons to `results/planning/analysis/`, highlighting cases where planning helps, harms, or introduces failure modes unrelated to the underlying model's reasoning ability.

### Generating LaTeX Tables

Convert results to publication-ready tables matching the paper's reporting format:

```bash
python -m single_agent.reasoning.generate_reasoning_table
```

Generates three LaTeX tables:
- **Accuracy table**: Reports accuracy across GSM8K, CSQA, and MATH-100 for each model under NoPlan, Crew-Plan, and Direct-LLM-Plan conditions
- **Failure rates table**: Identifies formatting failure frequencies for Crew-Plan and Direct-LLM-Plan, revealing when schema constraints cause parsing errors
- **Runtime multipliers table**: Shows computational overhead introduced by planning modes relative to NoPlan, quantifying the cost of planning mechanisms

Output files are saved in `results/planning/` with `.tex` extensions. These tables enable systematic comparison of how planning interface design affects accuracy, robustness, and efficiency across different model sizes and capabilities.

## Result Structure

Each result JSON file contains:

```json
{
  "benchmark": "csqa",
  "model": "gpt-4o-mini",
  "metrics": {
    "correct": 1234,
    "total": 2000,
    "accuracy": 0.617,
    "avg_time": 2.34,
    "total_time": 4680.0
  },
  "questions": [
    {
      "qid": "question_001",
      "question": "...",
      "gold": "A",
      "pred": "A",
      "pred_original": "The answer is A",
      "correct": true,
      "time_used": 2.1,
      "tokens_out": 45,
      "llm_response": "..."
    }
  ]
}
```

## Extending the Experiment

### Adding a New Benchmark

1. **Create benchmark class**: Implement a benchmark class in `benchmarks/` following the interface used by `GSM8KBenchmark`, `CSQABenchmark`, and `MATHBenchmark`

2. **Add CrewAI crew**: Create a new crew class (e.g., `crew_newbench.py`) following the pattern in `crew_gsm8k.py`:
   ```python
   @CrewBase
   class SingleAgentCrewNewBench():
       @agent
       def reasoner(self) -> Agent:
           return Agent(config=self.agents_config['reasoner'], llm=CONFIG["llm"])
       
       @task
       def newbench_task(self) -> Task:
           return Task(config=self.tasks_config['newbench_task'])
       
       @crew
       def crew(self) -> Crew:
           # ... planning configuration
   ```

3. **Add task definition**: Add task description to `config/tasks.yaml`

4. **Update runners**: Add benchmark case to `crewai_test.py` and `direct_llm_planning_test.py`

5. **Add format handling**: If the benchmark requires specific answer formats, extend the format constraints in `direct_llm_planning_test.py`'s `solve_with_direct_planning()` function

### Supporting Additional Frameworks

To evaluate planning in other frameworks (e.g., LangGraph, AutoGen):

1. Create framework-specific crew/agent classes similar to the CrewAI implementations
2. Implement planning toggle mechanism (if the framework supports it)
3. Add runner function to `crewai_test.py` or create a new test file
4. Ensure result file naming follows the pattern: `{framework}_{benchmark}_{mode}_{model}.json`

### Customizing Planning Behavior

The direct LLM planning implementation allows customization:

- **Plan length limits**: Adjust `MAX_PLAN` in `direct_llm_planning_test.py` (default 1500 characters)
- **Planning prompts**: Modify system prompts in `solve_with_direct_planning()` to change plan generation style
- **Answer format enforcement**: Extend the `answer_mode` parameter to add new format constraints

For CrewAI planning, behavior is controlled by the framework's internal planning mechanism. Refer to CrewAI documentation for planning customization options.

## Notes on Model Support

The experiment supports both cloud-based (OpenAI API) and local (Ollama) models. For Ollama models, use the `ollama/{model_name}` format in configuration. The `call_llm()` function in `direct_llm_planning_test.py` handles model routing automatically.

When using local models, ensure Ollama is running and the specified model is available. Token counting for local models uses character-based estimation (characters/4) rather than actual tokenizer counts.

## Troubleshooting

**Crew-Plan shows high failure rates**: Schema-constrained planning requires models to follow a rigid format. When models deviate from the required schema, CrewAI cannot parse the plan or extract the final answer, resulting in systematic failures. This is particularly common for smaller and medium-sized local models. Consider using Direct-LLM-Plan for models that struggle with format constraints, or verify that `planning_llm` is set correctly in `config.py`.

**CrewAI planning fails silently**: Check that `planning_llm` is set in `config.py` when `planning=True`. Some models may not reliably produce schema-compliant plans; failures may appear as empty predictions or "FAILED" entries in results.

**Direct-LLM-Plan produces invalid formats**: The format enforcement in `solve_with_direct_planning()` uses regex extraction as a lightweight post-processing step. If answers are consistently malformed, adjust the extraction patterns or increase plan trimming. Unlike Crew-Plan, Direct-LLM-Plan failures are rare (typically 0%) because it doesn't enforce rigid schema requirements.

**Results show inconsistent correctness**: Run the correction scripts (`correct_gsm8k.py`, `correct_csqa.py`) to re-evaluate with updated normalization logic. Some benchmarks require careful normalization of both predictions and gold answers, especially for MATH where symbolic expressions may have multiple valid representations.

**High computational overhead with Crew-Plan**: Crew-Plan's schema-constrained planning introduces substantial runtime overhead (2× to 33× depending on model and dataset). This is expected behavior due to the framework's requirement to generate structured plans and parse them. Direct-LLM-Plan provides a more efficient alternative with modest overhead (1.2× to 3×).

**Ollama models timeout**: Increase the timeout in `call_llm()` (default 200 seconds) or use smaller model subsets for testing. Local models may require more time for plan generation, especially in Crew-Plan mode where schema compliance adds complexity.
