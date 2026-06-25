# Data

Raw social-network datasets are not committed because they are large and contain user-level identifiers from public datasets.

## Sample Data

The `data/sample/` files are tiny synthetic edge lists used to verify that the analysis pipeline runs.

## Full Data

To reproduce the large-scale analysis, download the public SNAP datasets and place them here:

```text
data/raw/facebook_combined.txt
data/raw/gplus/
```

Expected inputs:

- `facebook_combined.txt`: undirected Facebook edge list.
- Google+ `.edges` and `.circles` files: directed ego networks and user-defined circle memberships.

The combined Google+ edge list is useful for large-scale graph exploration, but circle-matching analysis requires the per-user files that include circle labels.

`facebook_combined.txt` is also the input for the link-prediction/node-embeddings pipeline (`run_link_prediction.py`); it uses the same `load_facebook_graph()` loader and full-data fallback as `run_analysis.py`, so no separate data setup is needed.
