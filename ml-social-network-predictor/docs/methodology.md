# Methodology

## Graph Representation

Each social network is represented as an edge-list graph. Facebook relationships are treated as undirected ties. Google+ relationships are treated as directed ties when ego-network files are available.

## Network-Level Metrics

The analysis computes:

- Number of vertices and edges
- Connectivity
- Diameter, or diameter of the largest connected component when the graph is disconnected
- Average and maximum degree
- Core nodes above a configurable degree threshold

## Ego Networks

An ego network is the induced subgraph containing one focal user and that user's immediate neighbors. Ego networks make it possible to inspect local social structure around highly connected or representative users.

## Community Detection

The pipeline supports several `igraph` community detection methods:

- Fast-Greedy
- Edge-Betweenness
- Infomap
- Walktrap

Fast-Greedy is used as the default for the Facebook ego-network summary because it is efficient on undirected graphs and produces interpretable local communities.

## Embeddedness and Dispersion

Embeddedness measures how many mutual neighbors two connected users share. Dispersion estimates how spread out those mutual neighbors are inside the ego network. Together, the two metrics help distinguish close-knit relationships from bridge-like relationships across social contexts.

## Google+ Circle Matching

When Google+ `.circles` files are available, detected communities can be compared with user-defined circles. For each circle, the match score is the largest overlap with any detected community divided by the total overlap across communities.

## Link Prediction and Node Embeddings

`run_link_prediction.py` extends the Facebook analysis into a predictive task: given the graph with some edges held out, predict which vertex pairs are actually connected.

**Node embeddings.** Each vertex is embedded via DeepWalk-style random walks: `igraph` generates uniform random walks (not the literal p/q-biased Node2Vec transition rule, hence "DeepWalk-style" rather than "Node2Vec"), a sliding window over each walk produces (center, context) pairs, and a small skip-gram model with negative sampling — two `nn.Embedding` tables trained in PyTorch with `BCEWithLogitsLoss` — learns a vector per vertex. This is hand-rolled rather than using `gensim`/`node2vec` because neither has a prebuilt wheel/compatible dependency set for this project's Python version; the hand-rolled version is also a more direct demonstration of the underlying skip-gram mechanics.

**Train/test edge split.** A naive split that holds out edges but still trains embeddings on the full graph leaks information: random walks would already traverse the "held-out" edges, making the held-out edges trivially easy to predict and the evaluation meaningless. Instead, held-out positive test edges are removed from a copy of the graph (`train_graph`) before any walk generation, embedding training, or heuristic computation happens — the model only ever sees `train_graph`. Negative (non-edge) examples are sampled separately for train and test, each excluding the other partition's positive edges so a held-out positive can't accidentally be labeled negative.

**Feature engineering.** Each candidate edge (u, v) is scored with two feature groups: the embedding-based cosine similarity between the learned vectors for u and v, and the existing heuristics — common-neighbor count, Jaccard coefficient, same-community membership (via the existing `detect_communities`), and the existing `embeddedness_scores()`/`dispersion_scores()` functions from `community_detection.py`, applied with u as the ego vertex.

**Evaluation.** Three logistic-regression classifiers are trained on the same train/test split: heuristics-only, embeddings-only, and combined. ROC-AUC measures how well each ranks true edges above non-edges across all thresholds; precision/recall are also reported at a fixed 0.5 probability threshold. Comparing the three answers the central question — do learned embeddings add predictive power beyond the hand-engineered graph heuristics already in this repository?

**Known simplifications.** Negative edges are sampled 1:1 against positives, which is a simplification relative to the real class imbalance (a social graph has far more non-edges than edges) and affects the precision/recall numbers; ROC-AUC is reported alongside specifically because it is less sensitive to this choice.
