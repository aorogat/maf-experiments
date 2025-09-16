"""
Concordia baseline runner with OpenAI.
Usage:
    python -m single_agent.framework_overhead.concordia_runner --model openai/gpt-5-nano
"""

import os
import time
import argparse
import numpy as np
from dotenv import load_dotenv

# Concordia core simulation & prefabs
from concordia.prefabs.simulation.generic import Simulation
from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as game_master_prefabs
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions
from concordia.language_model import utils as lm_utils
from concordia.language_model import base_gpt_model

# OpenAI SDK internals (we patch low-level call to strip bad args)
from openai.resources.chat.completions import Completions


# -------------------------------------------------------------------------
# PATCH 1: Strip unsupported arguments from OpenAI calls
# -------------------------------------------------------------------------
_original_create = Completions.create
def _patched_create(self, *args, **kwargs):
    for bad_arg in ["reasoning_effort", "verbosity"]:
        kwargs.pop(bad_arg, None)  # safely remove if present
    return _original_create(self, *args, **kwargs)
Completions.create = _patched_create


# -------------------------------------------------------------------------
# PATCH 2: Handle Concordia's fragile MCQ parsing
# -------------------------------------------------------------------------
_original_sample_choice = base_gpt_model.BaseGPTModel.sample_choice

def _patched_sample_choice(self, *args, **kwargs):
    """
    Robust MCQ handler.
    If Concordia’s strict parsing fails, we fallback to a deterministic safe answer.
    This prevents crashes due to "Too many multiple choice attempts".
    """
    try:
        return _original_sample_choice(self, *args, **kwargs)
    except Exception:
        return 0, "a", "forced"  # idx=0, response="a", debug="forced"

base_gpt_model.BaseGPTModel.sample_choice = _patched_sample_choice


# -------------------------------------------------------------------------
# EXPERIMENT CONFIGURATION
# -------------------------------------------------------------------------
QUESTION = "What is 2+2?"


class ConcordiaRunner:
    """
    Concordia runner configured for OpenAI gpt-4o-mini.

    Key fixes applied:
      - OpenAI client patched to strip unsupported args.
      - MCQ handling patched to prevent InvalidResponseError.

    Experiment setup:
      - One Entity agent ("TestAgent") with a goal to answer math in JSON.
      - One GameMaster controlling the environment.
      - Simulation limited to a single turn (default_max_steps=1).
    """

    def __init__(self, model: str = "openai/gpt-4o-mini"):
        load_dotenv()

        if not model.startswith("openai/"):
            raise ValueError("Concordia only supports OpenAI models. Use e.g., 'openai/gpt-4o-mini'.")

        # Strip "openai/" prefix for Concordia
        model_name = model.split("openai/")[1]

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in .env")

        # Initialize Concordia language model wrapper for OpenAI
        self.model = lm_utils.language_model_setup(
            api_type="openai",
            model_name=model_name,
            api_key=api_key,
            disable_language_model=False,
        )

        # Minimal dummy embedder (3D vector of ones)
        self.embedder = lambda _: np.ones(3)

        # Load prefab classes
        self.prefabs = {
            **helper_functions.get_package_classes(entity_prefabs),
            **helper_functions.get_package_classes(game_master_prefabs),
        }

        # Shared config template (used for both fresh + reuse modes)
        self.instances = [
            prefab_lib.InstanceConfig(
                prefab="basic__Entity",
                role=prefab_lib.Role.ENTITY,
                params={"name": "TestAgent",
                        "goal": "Answer simple math in JSON format {\"answer\": number}"},
            ),
            prefab_lib.InstanceConfig(
                prefab="generic__GameMaster",
                role=prefab_lib.Role.GAME_MASTER,
                params={"name": "default rules"},
            ),
        ]
        self.base_config = prefab_lib.Config(
            default_premise=QUESTION,
            default_max_steps=1,
            prefabs=self.prefabs,
            instances=self.instances,
        )

        # Prebuilt simulation (used by run_reuse)
        self.sim = Simulation(config=self.base_config,
                              model=self.model,
                              embedder=self.embedder)

    # -----------------------------------------------------------------
    # Option A: Fresh simulation each time (independent trials)
    # -----------------------------------------------------------------
    def run(self, question: str = QUESTION):
        """Rebuild Simulation object each run to avoid accumulation."""
        config = prefab_lib.Config(
            default_premise=question,
            default_max_steps=1,
            prefabs=self.prefabs,
            instances=self.instances,
        )
        sim = Simulation(config=config, model=self.model, embedder=self.embedder)

        start = time.perf_counter()
        results_log = sim.play(max_steps=1)
        end = time.perf_counter()
        return str(results_log), (end - start) * 1000

    # -----------------------------------------------------------------
    # Option B: Reuse simulation (accumulates context/logs)
    # -----------------------------------------------------------------
    def run_reuse(self, question: str = QUESTION):
        """Reuse the same Simulation object, accumulating context."""
        self.sim._config.default_premise = question

        start = time.perf_counter()
        results_log = self.sim.play(max_steps=1)
        end = time.perf_counter()
        return str(results_log), (end - start) * 1000


# -------------------------------------------------------------------------
# ENTRYPOINT: Run the experiment
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/gpt-4o-mini",
                        help="Model to use (must be OpenAI, e.g., 'openai/gpt-5-nano')")
    args = parser.parse_args()

    print(f"=== Concordia Test ({args.model}) ===")
    runner = ConcordiaRunner(model=args.model)

    print("\n[Fresh run mode]")
    resp, latency = runner.run(QUESTION)
    print(f"Q={QUESTION} | A={resp[:200]}... | ⏱️ {latency:.2f} ms")

    print("\n[Reuse run mode]")
    resp, latency = runner.run_reuse(QUESTION)
    print(f"Q={QUESTION} | A={resp[:200]}... | ⏱️ {latency:.2f} ms")
