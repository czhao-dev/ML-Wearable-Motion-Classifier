#!/usr/bin/env python3
"""Community Detection in Social Networks - main pipeline runner.

Ported from run_analysis.R; behavior is unchanged.

Usage: python3 run_analysis.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "python"))

from community_detection import ensure_dirs
from utils import write_summary_metrics
from load_data import load_facebook_graph, load_google_plus_graph
from facebook_analysis import analyze_facebook_network
from google_plus_analysis import circle_overlap_scores, read_google_plus_circles

ensure_dirs("figures", "results")

facebook_graph = load_facebook_graph()
facebook_results = analyze_facebook_network(
    graph=facebook_graph,
    core_degree_threshold=200,
    figures_dir="figures",
    results_dir="results",
)

write_summary_metrics(facebook_results["summary"], "results/summary_metrics.csv")

google_plus_graph = load_google_plus_graph()
google_plus_circles = read_google_plus_circles("data/sample/google_plus_sample_circles.txt")
circle_scores = (
    circle_overlap_scores(google_plus_graph, google_plus_circles, method="walktrap")
    + circle_overlap_scores(google_plus_graph, google_plus_circles, method="infomap")
)

with open("results/google_plus_circle_scores.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["circle_id", "method", "match_score"])
    writer.writeheader()
    writer.writerows(circle_scores)

print("Analysis complete. Results written to results/ and figures/.")
