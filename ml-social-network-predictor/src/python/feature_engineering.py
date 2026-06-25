"""Joins learned embeddings with the existing graph heuristics
(embeddedness, dispersion, common neighbors, community co-membership) into a
flat feature table for link-prediction classifiers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from community_detection import embeddedness_scores, dispersion_scores

EMBEDDING_FEATURE_COLUMNS = ["embedding_cosine_similarity"]
HEURISTIC_FEATURE_COLUMNS = [
    "common_neighbors",
    "jaccard_coefficient",
    "same_community",
    "embeddedness",
    "dispersion",
]


def embedding_features(u, v, embeddings):
    a, b = embeddings[u], embeddings[v]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    similarity = float(np.dot(a, b) / denom) if denom else 0.0
    return {"embedding_cosine_similarity": similarity}


def _ego_centric_scores(graph, u):
    """Embeddedness/dispersion of every neighbor of u, relative to u as ego.
    Reuses the existing embeddedness_scores()/dispersion_scores() functions
    from community_detection.py, which are ego-vertex-centric and operate on
    a pre-built 1-hop induced subgraph -- the same pattern facebook_analysis.py
    uses for ego-network analysis."""
    ego_subgraph = graph.induced_subgraph(graph.neighborhood(u, order=1))
    embeddedness = {row["vertex"]: row["embeddedness"] for row in embeddedness_scores(ego_subgraph, u)}
    dispersion = {row["vertex"]: row["dispersion"] for row in dispersion_scores(ego_subgraph, u)}
    return embeddedness, dispersion


def heuristic_features(graph, u, v, clustering, ego_cache):
    if u not in ego_cache:
        ego_cache[u] = _ego_centric_scores(graph, u)
    embeddedness_by_neighbor, dispersion_by_neighbor = ego_cache[u]

    u_neighbors = set(graph.neighbors(u))
    v_idx = graph.vs.find(name=v).index
    v_neighbors = set(graph.neighbors(v))
    common = u_neighbors & v_neighbors
    union = u_neighbors | v_neighbors
    jaccard = len(common) / len(union) if union else 0.0

    membership = clustering.membership
    same_community = membership[graph.vs.find(name=u).index] == membership[v_idx]

    return {
        "common_neighbors": len(common),
        "jaccard_coefficient": jaccard,
        "same_community": int(same_community),
        "embeddedness": embeddedness_by_neighbor.get(v, 0),
        "dispersion": dispersion_by_neighbor.get(v, 0),
    }


def build_edge_features(graph, edge_label_df, embeddings, clustering):
    ego_cache = {}
    rows = []
    for _, row in edge_label_df.iterrows():
        u, v, label = row["u"], row["v"], row["label"]
        features = {"u": u, "v": v, "label": label}
        features.update(embedding_features(u, v, embeddings))
        features.update(heuristic_features(graph, u, v, clustering, ego_cache))
        rows.append(features)
    return pd.DataFrame(rows)
