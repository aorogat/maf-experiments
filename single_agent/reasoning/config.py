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
    # Smoke test: few questions, planning only. Does NOT overwrite full experiment
    # files (result_suffix="_smoke"). Restore full list / n_* / empty suffix later.
    # "benchmarks": ["csqa","math","gsm8k"],
    "benchmarks": ["math"],
    # "benchmarks": ["gsm8k", "csqa"],
    # "benchmarks": ["csqa"],
    "n_gsm8k": 3,         # set None for full test set, a number for a subset
    "n_csqa": 3,        # set None for full test set, a number for a subset
    "n_math": 3,         # set None for full test set, a number for a subset
    "result_suffix": "_smoke",  # set "" for normal experiment filenames
}

# Sweep used by crewai_test when running multiple frontier models in one go.
# CrewAI: pass model IDs only; let CrewAI/provider defaults decide thinking/reasoning.
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


def resolve_llm(model: str | None = None):
    """
    CrewAI model handle: return the model ID string and let CrewAI decide
    thinking / reasoning settings (provider defaults).
    """
    return model if model is not None else CONFIG["llm"]


def no_thinking_kwargs(model: str) -> dict:
    """
    Extra LiteLLM kwargs to disable thinking/reasoning for non-CrewAI runners
    (e.g. direct_llm_planning_test).

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
