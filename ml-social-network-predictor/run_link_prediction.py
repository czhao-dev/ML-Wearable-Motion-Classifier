#!/usr/bin/env python3
"""Link prediction & node embeddings pipeline.

Trains DeepWalk-style node embeddings (PyTorch) on a held-out training graph,
engineers features combining those embeddings with the existing
embeddedness/dispersion/community heuristics, and compares link-prediction
classifiers (heuristics-only vs. embeddings-only vs. combined).

Usage: python3 run_link_prediction.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "python"))

from community_detection import detect_communities, ensure_dirs
from utils import write_csv
from load_data import load_facebook_graph
from link_prediction_data import split_edges_for_link_prediction, build_edge_label_dataset
from node_embeddings import (
    train_node_embeddings,
    sanity_check_embeddings,
    save_embeddings,
    save_model,
)
from feature_engineering import build_edge_features, HEURISTIC_FEATURE_COLUMNS, EMBEDDING_FEATURE_COLUMNS
from link_prediction_model import run_comparison, plot_roc_curve, save_classifier, train_classifier

ensure_dirs("figures", "results")

graph = load_facebook_graph()

train_graph, train_pos, test_pos, train_neg, test_neg = split_edges_for_link_prediction(graph)
print(
    f"Edge split: {len(train_pos)} train-positive / {len(test_pos)} test-positive "
    f"edges (train_graph: {train_graph.vcount()} vertices, {train_graph.ecount()} edges)."
)

embeddings, embedding_model = train_node_embeddings(train_graph)
sanity_check_embeddings(train_graph, embeddings)
save_embeddings(embeddings, "results/node_embeddings.pkl")
save_model(embedding_model, "results/node_embeddings_model.pt")

clustering = detect_communities(train_graph, method="fast_greedy")

train_label_df = build_edge_label_dataset(train_pos, train_neg)
test_label_df = build_edge_label_dataset(test_pos, test_neg)

train_features_df = build_edge_features(train_graph, train_label_df, embeddings, clustering)
test_features_df = build_edge_features(train_graph, test_label_df, embeddings, clustering)

comparison_df, curves = run_comparison(train_features_df, test_features_df)
write_csv(
    comparison_df.to_dict("records"),
    ["feature_set", "roc_auc", "precision", "recall"],
    "results/link_prediction_metrics.csv",
)
plot_roc_curve(curves, "figures/roc_curve.png")

combined_columns = HEURISTIC_FEATURE_COLUMNS + EMBEDDING_FEATURE_COLUMNS
final_model = train_classifier(train_features_df, combined_columns)
save_classifier(final_model, "results/link_prediction_classifier.pkl")

print(comparison_df.to_string(index=False))
print("Link prediction pipeline complete. Results written to results/ and figures/.")
