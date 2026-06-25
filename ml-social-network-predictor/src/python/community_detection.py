"""Community detection and social-network metrics. Ported from
R/community_detection.R; behavior is unchanged."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import graph_density_safe


def ensure_dirs(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def detect_communities(graph, method="fast_greedy", weights=None):
    """Run igraph community detection, returning a VertexClustering.

    fast_greedy and edge_betweenness run on an undirected copy of the graph;
    infomap and walktrap run on the graph as-is.
    """
    if method == "fast_greedy":
        return graph.as_undirected().community_fastgreedy(weights=weights).as_clustering()
    if method == "edge_betweenness":
        return graph.as_undirected().community_edge_betweenness(weights=weights).as_clustering()
    if method == "infomap":
        return graph.community_infomap(edge_weights=weights)
    if method == "walktrap":
        return graph.community_walktrap(weights=weights).as_clustering()
    raise ValueError(f"Unknown community detection method: {method}")


def community_summary(graph, clustering):
    rows = []
    for comm_id, members in enumerate(clustering):
        subgraph = graph.induced_subgraph(members)
        clustering_coeff = subgraph.transitivity_undirected(mode="zero")
        rows.append({
            "community_id": comm_id + 1,
            "size": subgraph.vcount(),
            "density": graph_density_safe(subgraph),
            "clustering_coefficient": clustering_coeff,
        })
    return rows


def embeddedness_scores(ego_graph, ego_vertex):
    neighbor_ids = ego_graph.neighbors(ego_vertex)
    neighbor_names = {ego_graph.vs[n]["name"] for n in neighbor_ids}

    if not neighbor_ids:
        return []

    scores = []
    for n in neighbor_ids:
        vertex_neighbor_names = {ego_graph.vs[m]["name"] for m in ego_graph.neighbors(n)}
        scores.append({
            "vertex": ego_graph.vs[n]["name"],
            "embeddedness": len(neighbor_names & vertex_neighbor_names),
        })
    return scores


def dispersion_scores(ego_graph, ego_vertex):
    neighbor_ids = ego_graph.neighbors(ego_vertex)
    neighbor_names = {ego_graph.vs[n]["name"] for n in neighbor_ids}

    if not neighbor_ids:
        return []

    scores = []
    for n in neighbor_ids:
        vertex_neighbor_names = {ego_graph.vs[m]["name"] for m in ego_graph.neighbors(n)}
        mutual_names = neighbor_names & vertex_neighbor_names

        if len(mutual_names) < 2:
            scores.append({"vertex": ego_graph.vs[n]["name"], "dispersion": 0})
            continue

        mutual_ids = [ego_graph.vs.find(name=name).index for name in mutual_names]
        mutual_graph = ego_graph.induced_subgraph(mutual_ids)
        distances = mutual_graph.distances()
        # Pairs of mutual neighbors with no path between them inside the
        # mutual subgraph contribute 0 rather than infinity, so they don't
        # dominate the sum.
        total = sum(d for row in distances for d in row if d != float("inf"))
        scores.append({"vertex": ego_graph.vs[n]["name"], "dispersion": total / 2})
    return scores
