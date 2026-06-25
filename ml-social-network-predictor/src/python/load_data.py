"""Graph loading utilities. Ported from R/load_data.R; behavior is unchanged."""

from pathlib import Path

import igraph as ig


def read_edge_list_graph(path, directed=False):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Edge list not found: {path}")

    edges = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Expected at least two columns in edge list: {path}")
            edges.append((parts[0], parts[1]))

    graph = ig.Graph.TupleList(edges, directed=directed)
    graph.simplify(multiple=True, loops=True)
    return graph


def load_facebook_graph():
    full_path = Path("data/raw/facebook_combined.txt")
    sample_path = Path("data/sample/facebook_sample_edges.txt")

    if full_path.exists():
        print("Loading full Facebook edge list from data/raw/facebook_combined.txt")
        return read_edge_list_graph(full_path, directed=False)

    print("Full Facebook edge list not found; using sample data.")
    return read_edge_list_graph(sample_path, directed=False)


def load_google_plus_graph(path="data/sample/google_plus_sample_edges.txt"):
    return read_edge_list_graph(path, directed=True)
