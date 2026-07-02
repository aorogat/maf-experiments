"""Minimal Generative-Agent-Based-Modeling skeleton framework."""

from frameworks.gabm_skeleton.metrics import Metrics, RoundTrace, RunResult

__all__ = ["GABMSkeletonRunner", "Metrics", "RoundTrace", "RunResult"]


def __getattr__(name: str):
    if name == "GABMSkeletonRunner":
        from frameworks.gabm_skeleton.runner import GABMSkeletonRunner
        return GABMSkeletonRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
