"""
graph_builder.py
----------------
Utilities to build agent graphs for experiments.

Supports:
- Loading graphs from HuggingFace datasets (legacy).
- Future: building graphs dynamically using multi-agent frameworks
  like LangGraph, CrewAI, or custom generators.
"""

import json
import networkx as nx
from networkx.readwrite import json_graph
import pandas as pd
from datasets import load_dataset


def build_graph_from_hf(graph_model: str, graph_size: int, num_sample: int) -> nx.Graph:
    """
    Build a graph from the HuggingFace dataset (legacy mode).

    Parameters
    ----------
    graph_model : str
        Graph generator type (e.g., "ws", "ba", "dt").
    graph_size : int
        Number of nodes in the graph.
    num_sample : int
        Index of the sample graph to load.

    Returns
    -------
    nx.Graph
        The constructed NetworkX graph.
    """
    dataset = load_dataset("disco-eth/AgentsNet", split="train")
    _loaded_hf_df = pd.DataFrame(dataset)

    row = _loaded_hf_df[
        (_loaded_hf_df["graph_generator"] == graph_model)
        & (_loaded_hf_df["num_nodes"] == graph_size)
        & (_loaded_hf_df["index"] == num_sample)
    ]

    if len(row) == 0:
        raise ValueError(
            f"Graph not found in HF dataset: {graph_model}_{graph_size}_{num_sample}"
        )

    graph_dict = json.loads(row.iloc[0]["graph"])
    print(f"[GraphBuilder] Loaded HF graph: {graph_model}_{graph_size}_{num_sample}")
    return json_graph.node_link_graph(graph_dict["graph"], edges="links")


def build_graph_from_framework(framework: str, config: dict) -> nx.Graph:
    """
    Placeholder: Build a graph using a multi-agent framework
    (LangGraph, CrewAI, etc.).

    Parameters
    ----------
    framework : str
        Name of the framework ("langgraph", "crewai", etc.).
    config : dict
        Configuration for building the graph.

    Returns
    -------
    nx.Graph
        The constructed NetworkX graph.
    """
    print(f"[GraphBuilder] Building graph using framework={framework}, config={config}")
    # TODO: Implement actual integration with chosen framework.
    # For now, just return an empty graph as placeholder.
    return nx.Graph()


def get_graph(source: str, **kwargs) -> nx.Graph:
    """
    Generic entry point to build graphs.

    Parameters
    ----------
    source : str
        Graph source type ("hf" for HuggingFace, "framework" for MAS framework).
    kwargs : dict
        Arguments passed to the builder.

    Returns
    -------
    nx.Graph
    """
    if source == "hf":
        return build_graph_from_hf(
            graph_model=kwargs["graph_model"],
            graph_size=kwargs["graph_size"],
            num_sample=kwargs["num_sample"],
        )
    elif source == "framework":
        return build_graph_from_framework(
            framework=kwargs["framework"], config=kwargs.get("config", {})
        )
    else:
        raise ValueError(f"Unknown graph source: {source}")
