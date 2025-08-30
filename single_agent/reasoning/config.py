CONFIG = {
    "planning": False,      # toggle planning For True, we need to determine the planning_llm. defauld is openai
    "planning_llm": "ollama/gpt-oss:20b",
    "results_dir": "results",
    "benchmarks": ["gsm8k", "csqa"],
    "n_gsm8k": 3,         # set None for full test set, a number for a subset
    "n_csqa": 3,        # set None for full test set, a number for a subset
}

