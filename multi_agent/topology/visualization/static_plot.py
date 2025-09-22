"""
Static plotting utilities using matplotlib for multi-agent topology visualization.
"""

import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .loader import extract_graph_data
from .layouts import create_layout_positions
from .utils import COLOR_PALETTE


def visualize_network(
    result_data: Dict,
    layout: str = "spring",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Create a static matplotlib visualization of the network.

    - Nodes are colored by group assignment.
    - Edges are styled to show constraint violations.
    """

    G, group_assignments = extract_graph_data(result_data)
    pos = create_layout_positions(G, layout)

    fig, ax = plt.subplots(figsize=figsize)

    # Node colors
    node_colors = [
        COLOR_PALETTE.get(group_assignments.get(G.nodes[node]["name"]), "#CCCCCC")
        for node in G.nodes()
    ]

    # Split edges
    correct_edges = []
    incorrect_edges = []
    for u, v in G.edges():
        name_u = G.nodes[u]["name"]
        name_v = G.nodes[v]["name"]
        if group_assignments.get(name_u) != group_assignments.get(name_v):
            correct_edges.append((u, v))
        else:
            incorrect_edges.append((u, v))

    # Draw edges
    if correct_edges:
        nx.draw_networkx_edges(G, pos, edgelist=correct_edges, ax=ax,
                               edge_color="#2E8B57", width=2.0, alpha=0.8)
    if incorrect_edges:
        nx.draw_networkx_edges(G, pos, edgelist=incorrect_edges, ax=ax,
                               edge_color="#DC143C", width=2.0, alpha=0.8, style="dashed")

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=800, edgecolors="black", linewidths=1.5)

    # Labels
    labels = {node: G.nodes[node]["name"] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10, font_weight="bold")

    ax.set_title(
        f"{result_data.get('task','Unknown').title()} Task - "
        f"{result_data.get('graph_generator','Unknown').upper()} Topology",
        fontsize=14,
    )
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()

    return fig


def create_comparison_plot(
    result_files: List[str],
    layout: str = "spring",
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    Create a side-by-side comparison plot of multiple experiments.
    """
    n_files = len(result_files)
    cols = min(3, n_files)
    rows = (n_files + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    from .loader import load_result_file  # avoid circular import

    for i, filepath in enumerate(result_files):
        if i >= len(axes):
            break
        result_data = load_result_file(filepath)
        G, group_assignments = extract_graph_data(result_data)
        pos = create_layout_positions(G, layout)
        ax = axes[i]

        # Reuse visualize_network core logic
        node_colors = [
            COLOR_PALETTE.get(group_assignments.get(G.nodes[node]["name"]), "#CCCCCC")
            for node in G.nodes()
        ]
        nx.draw(G, pos, with_labels=True, labels={n: G.nodes[n]["name"] for n in G.nodes()},
                node_color=node_colors, node_size=500, ax=ax)
        ax.set_title(Path(filepath).stem, fontsize=10)

    # Hide extra axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig
