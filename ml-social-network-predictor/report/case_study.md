# Case Study: Community Detection in Social Networks

## Overview

This project studies how online social networks organize into local and global communities. The analysis uses graph algorithms to characterize network structure, identify highly connected users, and compare algorithmic communities with user-defined social groups.

The original full-data analysis used public Facebook and Google+ social-network datasets. The current repository includes a cleaned, reproducible workflow with sample data for quick execution and instructions for running the same methods on the full datasets.

## Analysis Questions

This case study focuses on six questions:

- How connected and compact is the overall social graph?
- Which users act as high-degree core nodes?
- What community structure appears inside ego networks?
- Which relationships have high embeddedness or high dispersion?
- How well do algorithmic communities align with user-defined Google+ circles?
- Can learned node embeddings improve link prediction over the hand-engineered graph heuristics above?

## Findings

### 1. Overall Facebook Graph Structure

The full Facebook graph contained 4,039 vertices and 88,234 undirected edges. The network was connected, with a diameter of 8. This indicates a compact social graph: even the most distant users in the dataset were connected by a relatively short path.

The average degree was 43.69, showing that users in the graph had a substantial number of observed connections. The degree distribution was tested against two simple curve fits. An exponential model fit the observed degree distribution much better than a reciprocal model, with a mean squared error of 8.19 compared with 96.30 for the reciprocal fit.

**Answer:** The Facebook graph was connected and compact. Its diameter was 8, and its average degree was about 43.69.

### 2. High-Degree Core Nodes

The analysis defined core nodes as users with more than 200 neighbors. Under this definition, the graph contained 40 core nodes. These users had an average degree of 279.38.

The personal network around node 1 contained 348 nodes and 2,866 edges: node 1, its 347 neighbors, and all edges among those neighbors. This ego network was dense enough to reveal clear local community structure.

**Answer:** The high-degree core users were the 40 nodes with more than 200 neighbors. Their average degree was about 279.38.

### 3. Ego-Network Community Structure

Community detection on node 1's personal network showed that different algorithms produce different levels of granularity:

- Fast-Greedy produced 8 communities, with two dominant communities of size 114 and 112.
- Edge-Betweenness produced 41 communities, including many very small or singleton groups.
- Infomap produced 26 communities, with one large group of size 142 and several medium/small groups.

After removing the core node from its own personal network, the number of communities increased. Many isolated or nearly isolated nodes appeared as singleton communities. This showed that the ego node acted as a connector holding together otherwise weaker local structures.

**Answer:** Ego networks contained recognizable community structure, but the detected structure depended strongly on the algorithm. Removing the core user fragmented the ego network and increased the number of small communities.

### 4. Embeddedness and Dispersion

Embeddedness measured how many mutual friends a node shared with the core node. Dispersion measured how spread out those mutual friends were across the local network.

Across the core personal networks, embeddedness values below 50 occurred most frequently, while values from about 50 to 200 were more evenly distributed. Most nodes had dispersion values below 200.

In many personal networks, the same node had maximum embeddedness, maximum dispersion, and maximum dispersion divided by embeddedness. This happened because having more mutual friends often creates more possible pairwise distances among those mutual friends.

However, the report highlighted network 2267 as an important counterexample. In that network:

- Node 69 had maximum embeddedness.
- Node 102 had maximum dispersion.
- Node 83 had maximum dispersion divided by embeddedness.

This distinction is important. A high-embeddedness node can be connected to many mutual friends who all belong to the same community. A high-dispersion node may have fewer mutual friends, but those friends can span multiple communities. Dispersion therefore helps identify relationships that bridge different social contexts.

**Answer:** High embeddedness identifies relationships with many shared contacts. High dispersion identifies relationships whose shared contacts are spread across communities, making dispersion more useful for detecting cross-community social bridges.

### 5. Recurring Community Types

The original analysis computed community-level features for core ego networks, including size, density, modularity, clustering coefficient, vertex betweenness, and assortativity. Only communities with more than 10 nodes were considered.

Two recurring community types were defined:

- **Type 1:** A relatively dense community with the maximum average vertex betweenness in its ego network. These communities tend to contain highly connected users that sit on many shortest paths.
- **Type 2:** A community with the maximum clustering coefficient and relatively low modularity. These communities tend to be tightly knit groups that are difficult to divide into smaller subcommunities.

**Answer:** The analysis found two recurring patterns: dense, high-betweenness communities and tightly clustered, low-modularity communities.

### 6. Google+ Circle Matching

The Google+ analysis compared algorithm-detected communities with user-defined circles. Unlike Facebook, Google+ relationships were directed, and users could place the same person into multiple circles.

For 57 Google+ ego networks with more than two circles:

- Walktrap match scores averaged 0.8086, ranged from 0.1827 to 1.0000, and exceeded 0.9 for 18 users.
- Infomap match scores averaged 0.9231, ranged from 0.4754 to 1.0000, and exceeded 0.9 for 45 users.

High match scores suggested that a user's circles aligned closely with graph communities. Low match scores suggested that circles reflected personal tagging habits, overlapping categories, or nested labels rather than clean graph structure.

For example, a user's circles might separate "family" and "college friends" in a way that maps cleanly to communities. But circles such as "college friends," "classmates," and "best friends" can overlap heavily, so the same person may belong to multiple user-defined groups even when the graph algorithm assigns them to one structural community.

**Answer:** Algorithmic communities often aligned with Google+ circles, especially under Infomap. Variation across users reflected different tagging habits: some users tagged by real social communities, while others used overlapping or preference-based labels.

### 7. Can Learned Embeddings Improve Link Prediction Over Heuristics?

Running `run_link_prediction.py` against the full Facebook graph (4,039 vertices, 88,234 edges) held out 1,000 real edges as a test set, removed them from the training graph before any embedding training or feature computation, and trained DeepWalk-style node embeddings (random walks + a PyTorch skip-gram model) only on the remaining training graph. A sanity check confirmed the embeddings captured real structure: mean cosine similarity was 0.68 for adjacent vertex pairs versus 0.23 for random non-adjacent pairs.

Three logistic-regression classifiers were trained and compared on the same train/test split:

| Feature set | ROC-AUC | Precision | Recall |
|---|---|---|---|
| Heuristics only (common neighbors, Jaccard, same-community, embeddedness, dispersion) | 0.974 | 0.992 | 0.733 |
| Embeddings only (cosine similarity of learned vectors) | 0.986 | 0.951 | 0.955 |
| Combined | 0.953 | 0.963 | 0.360 |

The existing graph heuristics already discriminate real edges from non-edges remarkably well (AUC 0.974) — embeddedness and common-neighbor counts are, after all, strong structural signals for friendship. Embedding-only features performed slightly better still (AUC 0.986) and with much higher recall, suggesting the learned vectors captured some structural signal beyond what the hand-engineered heuristics encode individually. The combined model's lower recall at the default 0.5 probability threshold (despite a still-high AUC) is consistent with the heuristic and embedding features being correlated — both are derived from the same underlying graph structure — which can shift a single linear model's decision boundary away from 0.5 even while it continues to rank edges well.

**Answer:** Yes, but by a modest margin on this dataset: learned embeddings (AUC 0.986) slightly outperformed the existing graph heuristics (AUC 0.974) at ranking true edges above non-edges, and noticeably improved recall. The practical takeaway is that the repository's original embeddedness/dispersion/common-neighbor heuristics were already strong link-prediction features in their own right — the embeddings add a smaller increment than one might expect from a from-scratch learned representation, which is itself a useful finding about how much structural signal simple heuristics already capture.

## Conclusion

The full-data analysis showed that large online social graphs can be compact globally while still containing rich local community structure. The Facebook graph was connected with a short diameter, and its high-degree core users created ego networks with visible communities. Removing a core user fragmented the local network, showing how influential ego nodes connect otherwise separate groups.

Embeddedness and dispersion provided complementary views of social ties. Embeddedness captured the volume of shared contacts, while dispersion captured whether those shared contacts spanned multiple communities. The distinction matters because socially important bridge relationships are not always the relationships with the most mutual friends.

The Google+ results showed that detected communities can align strongly with user-defined circles, but not uniformly. Infomap produced higher average circle-match scores than Walktrap in the original analysis. Users with low match scores likely used circles as flexible labels rather than strict community boundaries.

The link-prediction extension showed that the project's own hand-engineered heuristics (embeddedness, dispersion, common neighbors) were already strong predictors of real friendships, and that learned node embeddings added only a modest additional improvement on this dataset. This is a useful empirical check on how much of a graph's predictive structure simple, interpretable metrics can already capture before reaching for a learned representation.

Overall, the project demonstrates an end-to-end graph analytics and machine learning workflow: ingesting social-network data, building graph representations, extracting ego networks, comparing community detection algorithms, designing interpretable metrics, training node embeddings and link-prediction classifiers, and turning graph outputs into social-network insight.
