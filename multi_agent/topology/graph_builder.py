"""
graph_builder.py
----------------
Utilities to build agent graphs for experiments.

Supports two main modes:
1. HuggingFace ("hf"): loads pre-generated graphs from the AgentsNet dataset.
2. Framework ("framework"): allows integration with orchestration frameworks
   like LangGraph, CrewAI, or custom MAS graph generators.

Notes on LangGraph:
-------------------
For LangGraph runs, we currently *reuse the same HF graphs* as input.
The difference lies not in how graphs are built but in how agents
communicate during execution (handled in `langgraph_runner.py`).
Thus, when `framework="langgraph"` is passed, the system behaves
identically to "hf" mode in graph construction.
"""

import json
import networkx as nx
from networkx.readwrite import json_graph
import pandas as pd
from datasets import load_dataset


def build_graph_from_hf(graph_model: str, graph_size: int, num_sample: int) -> nx.Graph:
    """
    Build a graph from the HuggingFace dataset.

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



def build_graph_from_hf_all_connected(graph_model: str, graph_size: int, num_sample: int) -> nx.Graph:
    """
    Build a graph from HF dataset and then enforce full connectivity.

    Equivalent to simulating Concordia's "relay hub" behavior,
    where all agents can reach each other directly.
    """
    G = build_graph_from_hf(graph_model, graph_size, num_sample)

    nodes = list(G.nodes())
    for u in nodes:
        for v in nodes:
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)

    print(f"[GraphBuilder] Modified graph to fully connected: {graph_model}_{graph_size}_{num_sample}")

    return G



def build_graph_from_framework(framework: str, config: dict) -> nx.Graph:
    """
    Build a graph using a multi-agent framework (LangGraph, CrewAI, etc.).

    For LangGraph:
    - No special graph structure is required, we just use HF graphs.
    - The distinction is in execution (runner), not construction.

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
    if framework == "langgraph":
        print("[GraphBuilder] Using LangGraph mode: falling back to HF-style graphs.")
        # Still return an empty placeholder unless HF is explicitly requested
        return nx.Graph()

    print(f"[GraphBuilder] Building graph using framework={framework}, config={config}")
    # TODO: Implement integration with other frameworks (CrewAI, Concordia, etc.)
    return nx.Graph()


def build_graph_from_hf_sequential(graph_model: str, graph_size: int, num_sample: int) -> nx.Graph:
    """
    Build a graph from HF dataset and then rewire as sequential (path graph).
    This simulates CrewAI's sequential orchestration.
    """
    G = build_graph_from_hf(graph_model, graph_size, num_sample)

    # Remove all edges
    G.remove_edges_from(list(G.edges()))

    # Add path edges
    for i in range(graph_size - 1):
        G.add_edge(i, i + 1)

    print(f"[GraphBuilder] Modified graph to sequential path: {graph_model}_{graph_size}_{num_sample}")
    return G


def build_graph_from_hf_hierarchical(graph_model: str, graph_size: int, num_sample: int) -> nx.Graph:
    """
    Build a graph from HF dataset and then rewire as hierarchical (balanced 4-ary tree).
    This simulates CrewAI's hierarchical orchestration.
    """
    G = build_graph_from_hf(graph_model, graph_size, num_sample)

    # Remove all edges
    G.remove_edges_from(list(G.edges()))

    # Build 4-ary tree edges
    for parent in range(graph_size):
        for j in range(1, 5):  # 4 children
            child = 4 * parent + j
            if child < graph_size:
                G.add_edge(parent, child)

    print(f"[GraphBuilder] Modified graph to hierarchical 4-ary tree: {graph_model}_{graph_size}_{num_sample}")
    return G


def get_graph(source: str, **kwargs) -> nx.Graph:
    """
    Generic entry point to build graphs.

    Parameters
    ----------
    source : str
        Graph source type:
          - "hf": load from HuggingFace dataset.
          - "framework": build using a MAS framework (LangGraph, CrewAI, etc.).
    kwargs : dict
        Arguments passed to the builder.

    Returns
    -------
    nx.Graph
    """
    if source == "hf":
        print("[source: hf]")
        return build_graph_from_hf(
            graph_model=kwargs["graph_model"],
            graph_size=kwargs["graph_size"],
            num_sample=kwargs["num_sample"],
        )
    elif source == "hf_all_connected":
        print("[source: hf_all_connected]")
        return build_graph_from_hf_all_connected(
            graph_model=kwargs["graph_model"],
            graph_size=kwargs["graph_size"],
            num_sample=kwargs["num_sample"],
        )
    elif source == "hf_sequential":
        print("[source: hf_sequential]")
        return build_graph_from_hf_sequential(
            graph_model=kwargs["graph_model"],
            graph_size=kwargs["graph_size"],
            num_sample=kwargs["num_sample"],
        )
    elif source == "hf_hierarchical":
        print("[source: hf_hierarchical]")
        return build_graph_from_hf_hierarchical(
            graph_model=kwargs["graph_model"],
            graph_size=kwargs["graph_size"],
            num_sample=kwargs["num_sample"],
        )

    elif source == "framework":
        print("[source: framework]")
        return build_graph_from_framework(
            framework=kwargs["framework"], config=kwargs.get("config", {})
        )
    else:
        raise ValueError(f"Unknown graph source: {source}")
