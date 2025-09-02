"""
Concordia baseline runner with Ollama.
Usage:
    python -m single_agent.framework_overhead.concordia_runner
"""

import time
import numpy as np

from concordia.prefabs.simulation.generic import Simulation
from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as game_master_prefabs
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions

from single_agent.framework_overhead.ollama_adapter import OllamaAdapter


QUESTION = "What is 2+2?"


class ConcordiaRunner:
    """Reusable Concordia runner for overhead experiments using Ollama."""

    def __init__(self, model_name="deepseek-llm:7b"):
        # === Use our adapter instead of OpenAI ===
        self.model = OllamaAdapter(model_name)

        # Dummy embedder (no sentence_transformers required)
        self.embedder = lambda _: np.ones(3)

        # Prefabs
        self.prefabs = {
            **helper_functions.get_package_classes(entity_prefabs),
            **helper_functions.get_package_classes(game_master_prefabs),
        }

        # Minimal simulation setup
        instances = [
            prefab_lib.InstanceConfig(
                prefab="basic__Entity",
                role=prefab_lib.Role.ENTITY,
                params={"name": "TestAgent", "goal": "Answer simple math"},
            ),
            prefab_lib.InstanceConfig(
                prefab="generic__GameMaster",
                role=prefab_lib.Role.GAME_MASTER,
                params={"name": "default rules", "extra_event_resolution_steps": ""},
            ),
        ]

        config = prefab_lib.Config(
            default_premise=QUESTION,  # initial task prompt
            default_max_steps=1,       # just 1 step for overhead
            prefabs=self.prefabs,
            instances=instances,
        )

        self.sim = Simulation(
            config=config,
            model=self.model,
            embedder=self.embedder,
        )

    def run(self, question: str = QUESTION):
        """Run Concordia once with Ollama."""
        # Update premise for new input
        self.sim._config.default_premise = question

        start = time.perf_counter()
        results_log = self.sim.play(max_steps=1)
        end = time.perf_counter()

        return str(results_log), (end - start) * 1000


if __name__ == "__main__":
    runner = ConcordiaRunner(model_name="deepseek-llm:7b")
    print("=== Concordia Test (Ollama) ===")
    for i in range(3):
        resp, latency = runner.run(QUESTION)
        print(f"Run {i+1}: Q={QUESTION} | A={resp} | ⏱️ {latency:.2f} ms")
