"""Edge dataset construction for link prediction: train/test split of edges
without leaking held-out edges into the training graph, plus negative
(non-edge) sampling."""

import random

import pandas as pd


def sample_negative_edges(graph, n, exclude_pairs=None, seed=42):
    """Sample n non-adjacent vertex-name pairs, rejecting existing edges,
    self-loops, and any pair in exclude_pairs (e.g. held-out test positives)."""
    rng = random.Random(seed)
    names = graph.vs["name"]
    existing_names = {frozenset((names[e.source], names[e.target])) for e in graph.es}
    excluded = {frozenset(pair) for pair in (exclude_pairs or [])}

    negatives = []
    seen = set()
    attempts = 0
    max_attempts = n * 200 + 1000
    while len(negatives) < n and attempts < max_attempts:
        attempts += 1
        u, v = rng.sample(names, 2)
        key = frozenset((u, v))
        if key in existing_names or key in excluded or key in seen:
            continue
        seen.add(key)
        negatives.append((u, v))

    if len(negatives) < n:
        raise ValueError(
            f"Could only sample {len(negatives)} of {n} requested negative edges; "
            "graph may be too small/dense for this many non-edges."
        )
    return negatives


def split_edges_for_link_prediction(
    graph, test_fraction=0.2, seed=42, max_train_edges=4000, max_test_edges=1000
):
    """Hold out test_fraction of edges as positive test edges, removing them
    from a copy of the graph (train_graph) so embeddings/heuristics trained on
    train_graph never see the held-out edges. An edge is only removed if both
    endpoints retain degree >= 1 afterward, so no vertex is isolated.

    The labeled positive sets used for *classifier* training/evaluation are
    capped (max_train_edges / max_test_edges) independently of train_graph's
    actual size: feature engineering re-scores each ego's full neighborhood
    per labeled edge, so capping keeps runtime proportionate on the full
    ~88k-edge dataset without limiting how much structure the embeddings see
    (embeddings are still trained on the complete train_graph).

    Returns (train_graph, test_pos_edges, train_neg_edges, test_neg_edges) as
    vertex-name tuples for the edge lists.
    """
    rng = random.Random(seed)
    names = graph.vs["name"]
    train_graph = graph.copy()

    edge_indices = list(range(train_graph.ecount()))
    rng.shuffle(edge_indices)

    target_test_count = max(
        1, min(max_test_edges, int(round(test_fraction * train_graph.ecount())))
    )
    test_pos_edges = []
    to_delete = []
    for idx in edge_indices:
        if len(test_pos_edges) >= target_test_count:
            break
        edge = train_graph.es[idx]
        u, v = edge.source, edge.target
        if train_graph.degree(u) <= 1 or train_graph.degree(v) <= 1:
            continue
        test_pos_edges.append((names[u], names[v]))
        to_delete.append(idx)

    if not test_pos_edges:
        raise ValueError(
            "Could not hold out any test edges without isolating a vertex; "
            "graph is too small/sparse for link-prediction evaluation."
        )

    train_graph.delete_edges(to_delete)

    remaining_pos_edges = [(names[e.source], names[e.target]) for e in train_graph.es]
    train_pos_count = min(max_train_edges, len(remaining_pos_edges))
    train_pos_edges = rng.sample(remaining_pos_edges, train_pos_count)

    test_neg_edges = sample_negative_edges(
        graph, len(test_pos_edges), exclude_pairs=test_pos_edges, seed=seed
    )
    train_neg_edges = sample_negative_edges(
        graph, len(train_pos_edges),
        exclude_pairs=test_pos_edges + test_neg_edges,
        seed=seed + 1,
    )

    return train_graph, train_pos_edges, test_pos_edges, train_neg_edges, test_neg_edges


def build_edge_label_dataset(pos_edges, neg_edges):
    rows = [{"u": u, "v": v, "label": 1} for u, v in pos_edges]
    rows += [{"u": u, "v": v, "label": 0} for u, v in neg_edges]
    return pd.DataFrame(rows)
