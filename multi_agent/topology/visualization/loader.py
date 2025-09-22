"""
Loader utilities for multi-agent topology results.

This module provides functions to:
1. Load experiment result JSON files.
2. Extract graph structures into NetworkX objects.
3. Map agent group assignments for visualization.
"""

import json
import networkx as nx
from typing import Dict, Tuple


def load_result_file(filepath: str) -> Dict:
    """Load and parse a result JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def extract_graph_data(result_data: Dict) -> Tuple[nx.Graph, Dict[str, str]]:
    """
    Extract NetworkX graph and group assignments from result data.

    Returns
    -------
    G : nx.Graph
        NetworkX graph object representing the topology.
    group_assignments : dict
        Mapping of agent name -> group label.
    """
    G = nx.Graph()

    # Add nodes
    nodes = result_data["graph"]["nodes"]
    for node in nodes:
        G.add_node(node["id"], name=node["name"])

    # Add edges
    edges = result_data["graph"]["links"]
    for edge in edges:
        G.add_edge(edge["source"], edge["target"])

    # Extract group assignments
    answers = result_data.get("answers", [])
    group_assignments = {}
    for i, node in enumerate(nodes):
        if i < len(answers):
            group_assignments[node["name"]] = answers[i]

    return G, group_assignments
