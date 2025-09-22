"""
Topology analysis utilities for multi-agent experiments.
"""

import networkx as nx
from typing import Dict

from .loader import extract_graph_data


def analyze_topology_properties(result_data: Dict) -> Dict:
    """
    Analyze topological properties of the network.

    Returns
    -------
    properties : dict
        Dictionary of network metrics and statistics.
    """
    G, group_assignments = extract_graph_data(result_data)

    properties = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_connected": nx.is_connected(G) if G.number_of_nodes() > 0 else False,
        "avg_clustering": nx.average_clustering(G) if G.number_of_nodes() > 0 else 0,
        "max_degree": max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0,
        "min_degree": min(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0,
    }

    # Average shortest path length only if graph is connected
    if properties["is_connected"] and G.number_of_nodes() > 1:
        properties["avg_shortest_path"] = nx.average_shortest_path_length(G)
    else:
        properties["avg_shortest_path"] = "N/A"

    return properties
