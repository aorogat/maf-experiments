"""
Interactive visualization using Plotly for multi-agent topology results.
"""

import networkx as nx
import plotly.graph_objects as go
from typing import Dict

from .loader import extract_graph_data
from .layouts import create_layout_positions
from .utils import COLOR_PALETTE


def create_interactive_network(result_data: Dict, layout: str = "spring") -> go.Figure:
    """
    Create an interactive Plotly network visualization.
    - Nodes are black circles.
    - Edges are green if neighbors agreed, red if they disagreed.
    - Hover shows agent info, degree, and group assignment.
    """
    G, group_assignments = extract_graph_data(result_data)
    pos = create_layout_positions(G, layout)

    # Node info
    node_names = [G.nodes[n]["name"] for n in G.nodes()]
    node_groups = [group_assignments.get(name, "Unknown") for name in node_names]
    x_nodes = [pos[n][0] for n in G.nodes()]
    y_nodes = [pos[n][1] for n in G.nodes()]

    # Get consensus answers (if available)
    answers = result_data.get("answers", None)

    # Edge traces
    correct_x, correct_y, incorrect_x, incorrect_y = [], [], [], []
    for u, v in G.edges():
        u_name, v_name = G.nodes[u]["name"], G.nodes[v]["name"]
        x0, y0, x1, y1 = pos[u][0], pos[u][1], pos[v][0], pos[v][1]

        # Curved edges: add mid-point for smooth arc
        mid_x = (x0 + x1) / 2 + 0.1 * (y1 - y0)
        mid_y = (y0 + y1) / 2 - 0.1 * (x1 - x0)

        if answers:
            # Agreement vs violation based on answers
            if answers[u] == answers[v]:
                correct_x += [x0, mid_x, x1, None]
                correct_y += [y0, mid_y, y1, None]
            else:
                incorrect_x += [x0, mid_x, x1, None]
                incorrect_y += [y0, mid_y, y1, None]
        else:
            # Fallback: all edges treated as correct if no answers
            correct_x += [x0, mid_x, x1, None]
            correct_y += [y0, mid_y, y1, None]

    fig = go.Figure()

    # --- Agreement edges (solid green) ---
    fig.add_trace(go.Scatter(
        x=correct_x, y=correct_y,
        mode="lines",
        line=dict(color="#2E8B57", width=1),
        opacity=0.5,
        name="Agreement",
        hoverinfo="none"
    ))

    # --- Violation edges (solid red) ---
    fig.add_trace(go.Scatter(
        x=incorrect_x, y=incorrect_y,
        mode="lines",
        line=dict(color="#DC143C", width=1),
        opacity=0.9,
        name="Violations",
        hoverinfo="none"
    ))

    # --- Nodes (black circles) ---
    hover_text = [
        f"Agent: {name}<br>Group: {group}<br>Degree: {G.degree(n)}"
        for n, name, group in zip(G.nodes(), node_names, node_groups)
    ]

    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes,
        mode="markers",
        marker=dict(
            size=10,
            color="black",
            symbol="circle"
        ),
        hovertext=hover_text,
        hoverinfo="text",
        showlegend=False
    ))

    # Layout
    fig.update_layout(
        title=f"{result_data.get('task','Unknown').title()} - "
              f"{result_data.get('graph_generator','Unknown').upper()} Topology",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="white",
        showlegend=True
    )

    return fig
