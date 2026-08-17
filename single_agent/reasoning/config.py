CONFIG = {
    "planning": False,      # toggle planning For True, we need to determine the planning_llm. defauld is openai
    # "llm": "ollama/gpt-oss:20b",
    # "planning_llm": "ollama/gpt-oss:20b",
    # "math_judge_llm":  "gpt-4o-mini",

    # "llm": "gpt-4o-mini",
    # "planning_llm": "gpt-4o-mini",
    # "math_judge_llm":  "gpt-4o-mini",

    # "llm": "gpt-4.1",
    # "planning_llm": "gpt-4.1",
    # "math_judge_llm":  "gpt-4.1",

    # Claude Opus 5 (Anthropic via LiteLLM / CrewAI)
    "llm": "anthropic/claude-opus-5",
    "planning_llm": "anthropic/claude-opus-5",
    "math_judge_llm": "gpt-4o-mini",

    # Gemini 3.1 Pro (Google via LiteLLM / CrewAI)
    # "llm": "gemini/gemini-3.1-pro-preview",
    # "planning_llm": "gemini/gemini-3.1-pro-preview",
    # "math_judge_llm": "gpt-4o-mini",

    # GPT-5.6 Terra
    # "llm": "gpt-5.6-terra",
    # "planning_llm": "gpt-5.6-terra",
    # "math_judge_llm": "gpt-4o-mini",

    # GPT-5.6 Luna
    # "llm": "gpt-5.6-luna",
    # "planning_llm": "gpt-5.6-luna",
    # "math_judge_llm": "gpt-4o-mini",

    # "llm": "ollama/deepseek-llm:7b",
    # "planning_llm": "ollama/deepseek-llm:7b",
    # "math_judge_llm":  "gpt-4o-mini",

    # "llm": "ollama/llama3.1:8b",
    # "planning_llm": "ollama/llama3.1:8b",
    # "math_judge_llm":  "gpt-4o-mini",

    # "llm": "ollama/qwen:7b",
    # "planning_llm": "ollama/qwen:7b",
    # "math_judge_llm":  "gpt-4o-mini",

    # "llm": "ollama/phi4:14b",
    # "planning_llm": "ollama/phi4:14b",
    # "math_judge_llm":  "gpt-4o-mini",

    "results_dir": "results/planning",
    # Full Crew-Plan Opus 5 rerun (overwrites broken planning_* files).
    "benchmarks": ["csqa", "math", "gsm8k"],
    # "benchmarks": ["math"],
    # "benchmarks": ["gsm8k", "csqa"],
    # "benchmarks": ["csqa"],
    "n_gsm8k": None,       # full test set
    "n_csqa": None,        # full validation set
    "n_math": 100,         # MATH-100
    "result_suffix": "",   # overwrite normal experiment filenames
}

# Sweep used by crewai_test when running multiple frontier models in one go.
# CrewAI: resolve_llm() applies the same no_thinking_kwargs as Direct-LLM-Plan.
MODEL_CONFIGS = [
    {
        "llm": "anthropic/claude-opus-5",
        "planning_llm": "anthropic/claude-opus-5",
        "math_judge_llm": "gpt-4o-mini",
    },
    # {
    #     "llm": "gemini/gemini-3.1-pro-preview",
    #     "planning_llm": "gemini/gemini-3.1-pro-preview",
    #     "math_judge_llm": "gpt-4o-mini",
    # },
    # {
    #     "llm": "gpt-5.6-terra",
    #     "planning_llm": "gpt-5.6-terra",
    #     "math_judge_llm": "gpt-4o-mini",
    # },
    # {
    #     "llm": "gpt-5.6-luna",
    #     "planning_llm": "gpt-5.6-luna",
    #     "math_judge_llm": "gpt-4o-mini",
    # },
]


def no_thinking_kwargs(model: str) -> dict:
    """
    Extra kwargs to disable thinking/reasoning for planning runners
    (CrewAI via resolve_llm, and Direct-LLM-Plan via litellm.completion).

    Provider differences:
    - GPT-5.6: reasoning_effort="none" (default otherwise is medium)
    - Claude Opus 5: thinking={"type": "disabled"} (thinking on by default;
      Anthropic has no effort level "none")
    """
    m = str(model).lower()
    if m.startswith("gpt-5.6"):
        return {"reasoning_effort": "none"}
    if "claude-opus" in m:
        return {"thinking": {"type": "disabled"}}
    return {}


def resolve_llm(model: str | None = None):
    """
    CrewAI LLM handle for reasoner and planning_llm.

    Applies the same no_thinking_kwargs as Direct-LLM-Plan so NoPlan / Crew-Plan
    and Direct-LLM-Plan share thinking/reasoning overrides. CrewAI maps
    reasoning_effort as a first-class field; other keys (e.g. thinking) go through
    LLM(**kwargs) -> additional_params -> LiteLLM.
    """
    from crewai import LLM

    model_id = model if model is not None else CONFIG["llm"]
    return LLM(model=model_id, **no_thinking_kwargs(model_id))
