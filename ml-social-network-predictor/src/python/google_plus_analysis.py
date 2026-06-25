"""Google+ circle-matching analysis. Ported from R/google_plus_analysis.R;
behavior is unchanged."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from community_detection import detect_communities


def circle_overlap_scores(graph, circles, method):
    clustering = detect_communities(graph, method=method)
    community_sets = [{graph.vs[v]["name"] for v in members} for members in clustering]

    rows = []
    for circle_id, circle in enumerate(circles, start=1):
        circle_set = set(circle)
        overlaps = [len(circle_set & community) for community in community_sets]
        total = sum(overlaps)
        match_score = max(overlaps) / total if total > 0 else None
        rows.append({"circle_id": circle_id, "method": method, "match_score": match_score})
    return rows


def read_google_plus_circles(path):
    circles = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            circles.append(parts[1:])
    return circles
