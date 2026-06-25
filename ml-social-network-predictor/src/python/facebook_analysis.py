"""Facebook ego-network analysis. Ported from R/facebook_analysis.R;
behavior is unchanged."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import igraph as ig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import safe_diameter, write_csv
from community_detection import (
    community_summary,
    detect_communities,
    embeddedness_scores,
    dispersion_scores,
)


def analyze_facebook_network(graph, core_degree_threshold, figures_dir, results_dir):
    degrees = graph.degree()
    names = graph.vs["name"]
    core_threshold = min(core_degree_threshold, max(degrees))
    core_nodes = [name for name, d in zip(names, degrees) if d >= core_threshold]

    summary = {
        "vertices": graph.vcount(),
        "edges": graph.ecount(),
        "connected": graph.is_connected(),
        "diameter_or_largest_component_diameter": safe_diameter(graph),
        "average_degree": sum(degrees) / len(degrees),
        "max_degree": max(degrees),
        "core_degree_threshold": core_threshold,
        "core_node_count": len(core_nodes),
    }

    plot_degree_distribution(degrees, Path(figures_dir) / "facebook_degree_distribution.png")

    degree_by_name = dict(zip(names, degrees))
    core_rows = sorted(
        ({"vertex": name, "degree": degree_by_name[name]} for name in core_nodes),
        key=lambda row: row["degree"],
        reverse=True,
    )
    write_csv(core_rows, ["vertex", "degree"], Path(results_dir) / "facebook_core_nodes.csv")

    ego_vertex = choose_ego_vertex(graph, names, degrees)
    ego_graph = graph.induced_subgraph(graph.neighborhood(ego_vertex, order=1))

    communities = detect_communities(ego_graph, method="fast_greedy")
    plot_ego_network(ego_graph, ego_vertex, communities, Path(figures_dir) / "facebook_ego_network.png")

    community_rows = community_summary(ego_graph, communities)
    write_csv(community_rows, ["community_id", "size", "density", "clustering_coefficient"],
               Path(results_dir) / "facebook_ego_communities.csv")

    embeddedness_rows = {row["vertex"]: row for row in embeddedness_scores(ego_graph, ego_vertex)}
    dispersion_rows = {row["vertex"]: row for row in dispersion_scores(ego_graph, ego_vertex)}
    social_score_rows = []
    for vertex in sorted(set(embeddedness_rows) | set(dispersion_rows)):
        embeddedness = embeddedness_rows.get(vertex, {}).get("embeddedness")
        dispersion = dispersion_rows.get(vertex, {}).get("dispersion")
        ratio = dispersion / embeddedness if embeddedness else None
        social_score_rows.append({
            "vertex": vertex,
            "embeddedness": embeddedness,
            "dispersion": dispersion,
            "dispersion_to_embeddedness": ratio,
        })
    write_csv(social_score_rows, ["vertex", "embeddedness", "dispersion", "dispersion_to_embeddedness"],
               Path(results_dir) / "facebook_ego_social_scores.csv")

    return {
        "summary": summary,
        "core_nodes": core_rows,
        "ego_communities": community_rows,
        "ego_social_scores": social_score_rows,
    }


def choose_ego_vertex(graph, names, degrees):
    if "1" in names:
        return "1"
    return names[degrees.index(max(degrees))]


def plot_degree_distribution(degrees, path):
    fig, ax = plt.subplots(figsize=(8, 5.33), dpi=150)
    ax.hist(degrees, bins=30, color="#4C78A8", edgecolor="white")
    ax.set_title("Facebook Degree Distribution")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Node Count")
    fig.savefig(path)
    plt.close(fig)


def plot_ego_network(graph, ego_vertex, clustering, path):
    membership = clustering.membership
    is_ego = [name == ego_vertex for name in graph.vs["name"]]
    colors = [max(membership) + 1 if ego else m for ego, m in zip(is_ego, membership)]
    palette = ig.drawing.colors.ClusterColoringPalette(max(colors) + 1)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ig.plot(
        graph,
        target=ax,
        vertex_size=[14 if ego else 8 for ego in is_ego],
        vertex_color=[palette[c] for c in colors],
        vertex_label=None,
        edge_color="#B8B8B8",
    )
    ax.set_title("Ego Network Community Structure")
    fig.savefig(path)
    plt.close(fig)
