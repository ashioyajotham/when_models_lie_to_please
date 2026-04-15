"""
Circuit diagram generation.

Produces publication-quality visualizations of attribution graphs.
Nodes are colored by feature label (from characterization.py).
Edge width encodes contribution magnitude.
Layer position is used as the x-axis; feature activation strength as y-axis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from src.circuits.attribution import AttributionGraph

LABEL_COLORS = {
    "correctness_signal": "#2166ac",       # Blue
    "user_agreement_signal": "#d6604d",    # Red
    "confidence_modulator": "#f4a582",     # Light orange
    "reasoning_step_marker": "#92c5de",    # Light blue
    "override_trigger": "#b2182b",         # Dark red
    "output_formatter": "#999999",         # Grey
    "unknown": "#dddddd",                  # Light grey
}


def draw_attribution_graph(
    graph: AttributionGraph,
    feature_labels: dict[tuple[int, int], str] | None = None,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (14, 8),
    title: str | None = None,
) -> plt.Figure:
    """
    Draw an attribution graph as a layered directed graph.

    Args:
        graph: AttributionGraph instance
        feature_labels: Optional dict mapping (layer, feature_index) → label string
                        (classification from characterization.py)
        output_path: If provided, save the figure to this path
        title: Optional figure title

    Returns:
        matplotlib Figure
    """
    G = graph.to_networkx()
    if G.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Empty graph", ha="center", va="center")
        return fig

    # Assign positions: x = layer, y = feature index (normalized)
    layers = sorted({data["layer"] for _, data in G.nodes(data=True)})
    layer_to_x = {l: i for i, l in enumerate(layers)}

    layer_node_counts: dict[int, int] = {}
    pos = {}
    for node_id, data in G.nodes(data=True):
        layer = data["layer"]
        layer_node_counts[layer] = layer_node_counts.get(layer, 0) + 1

    layer_counters: dict[int, int] = {l: 0 for l in layers}
    for node_id, data in G.nodes(data=True):
        layer = data["layer"]
        total = layer_node_counts[layer]
        y_pos = layer_counters[layer] / max(total - 1, 1)
        pos[node_id] = (layer_to_x[layer], y_pos)
        layer_counters[layer] += 1

    node_colors = []
    for node_id in G.nodes():
        label = (feature_labels or {}).get(node_id, "unknown")
        node_colors.append(LABEL_COLORS.get(label, LABEL_COLORS["unknown"]))

    edge_weights = [abs(G[u][v].get("weight", 0.1)) for u, v in G.edges()]
    max_weight = max(edge_weights, default=1.0)
    edge_widths = [3.0 * w / max_weight for w in edge_weights]

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, ax=ax)
    nx.draw_networkx_edges(
        G, pos, width=edge_widths, alpha=0.6, arrows=True, arrowsize=12, ax=ax
    )

    # Layer labels on x-axis
    ax.set_xticks(list(range(len(layers))))
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
    ax.set_xlabel("Model Layer", fontsize=11)
    ax.set_ylabel("Feature (normalized position)", fontsize=11)

    if title:
        ax.set_title(title, fontsize=12, pad=12)
    elif graph.prompt_id:
        ax.set_title(
            f"Attribution graph — {graph.condition} — {graph.prompt_id}", fontsize=11
        )

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=label)
        for label, color in LABEL_COLORS.items()
        if label != "unknown"
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.8)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
