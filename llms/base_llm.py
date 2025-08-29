from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """Abstract interface for LLMs usable by any framework (LangGraph, CrewAI, Concordia)."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text given a prompt."""
        pass
