"""Small I/O and graph-safety helpers used across the social-graph-analyzer
pipeline. Ported from R/utils.R; behavior is unchanged."""

import csv


def write_csv(rows, fieldnames, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_metrics(summary, path):
    # R's unlist() coerces a mixed logical/numeric list to numeric (TRUE/FALSE
    # -> 1/0), so booleans are coerced the same way here for parity.
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for metric, value in summary.items():
            if isinstance(value, bool):
                value = int(value)
            writer.writerow([metric, value])


def safe_diameter(graph):
    if graph.is_connected():
        return graph.diameter(directed=graph.is_directed())
    components = graph.connected_components()
    largest = max(components, key=len)
    induced = graph.induced_subgraph(largest)
    return induced.diameter(directed=induced.is_directed())


def graph_density_safe(graph):
    if graph.vcount() < 2:
        return 0
    return graph.density(loops=False)
