"""Node embeddings via DeepWalk-style uniform random walks + a hand-rolled
skip-gram-with-negative-sampling model in PyTorch.

This is uniform-random-walk based (igraph's random_walk), not the literal
p/q-biased Node2Vec transition rule -- referred to as "Node2Vec-style"
elsewhere in the docs to mean "random walk + skip-gram embeddings" in
general, not the specific biased-walk algorithm.
"""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def generate_random_walks(graph, walks_per_node=10, walk_length=20, seed=42):
    rng = random.Random(seed)
    walk_length = min(walk_length, max(graph.vcount() - 1, 1))
    names = graph.vs["name"]
    walks = []
    for _ in range(walks_per_node):
        order = list(range(graph.vcount()))
        rng.shuffle(order)
        for v in order:
            walk_ids = graph.random_walk(v, walk_length, mode="all", stuck="return")
            walks.append([names[i] for i in walk_ids])
    return walks


def build_skipgram_pairs(walks, window_size=5):
    pairs = []
    for walk in walks:
        for i, center in enumerate(walk):
            lo = max(0, i - window_size)
            hi = min(len(walk), i + window_size + 1)
            for j in range(lo, hi):
                if j != i:
                    pairs.append((center, walk[j]))
    return pairs


class SkipGramModel(torch.nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.center_embeddings = torch.nn.Embedding(vocab_size, dim)
        self.context_embeddings = torch.nn.Embedding(vocab_size, dim)
        torch.nn.init.uniform_(self.center_embeddings.weight, -0.5 / dim, 0.5 / dim)
        torch.nn.init.zeros_(self.context_embeddings.weight)

    def forward(self, center_idx, context_idx, negative_idx):
        center = self.center_embeddings(center_idx)
        context = self.context_embeddings(context_idx)
        negatives = self.context_embeddings(negative_idx)

        pos_logits = (center * context).sum(dim=-1)
        neg_logits = torch.bmm(negatives, center.unsqueeze(-1)).squeeze(-1)
        return pos_logits, neg_logits


def train_node_embeddings(
    graph,
    dim=64,
    epochs=5,
    walks_per_node=10,
    walk_length=20,
    window_size=5,
    num_negative=5,
    batch_size=512,
    seed=42,
):
    dim = max(1, min(dim, graph.vcount() - 1))
    names = graph.vs["name"]
    name_to_idx = {name: i for i, name in enumerate(names)}
    vocab_size = len(names)

    walks = generate_random_walks(graph, walks_per_node, walk_length, seed)
    pairs = build_skipgram_pairs(walks, window_size)
    if not pairs:
        raise ValueError("No skip-gram pairs generated; graph may be too small.")

    centers = torch.tensor([name_to_idx[c] for c, _ in pairs], dtype=torch.long)
    contexts = torch.tensor([name_to_idx[x] for _, x in pairs], dtype=torch.long)
    dataset = TensorDataset(centers, contexts)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(pairs)), shuffle=True)

    torch.manual_seed(seed)
    model = SkipGramModel(vocab_size, dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        for center_batch, context_batch in loader:
            negative_batch = torch.randint(0, vocab_size, (len(center_batch), num_negative))
            pos_logits, neg_logits = model(center_batch, context_batch, negative_batch)
            pos_labels = torch.ones_like(pos_logits)
            neg_labels = torch.zeros_like(neg_logits)
            loss = loss_fn(pos_logits, pos_labels) + loss_fn(neg_logits, neg_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    vectors = model.center_embeddings.weight.detach().numpy()
    embeddings = {name: vectors[idx] for name, idx in name_to_idx.items()}
    return embeddings, model


def sanity_check_embeddings(graph, embeddings, sample_size=500, seed=42):
    """Adjacent vertex pairs should have higher embedding cosine similarity
    than random non-adjacent pairs if the embeddings captured graph structure.
    Prints a pass/fail line and returns True/False."""
    rng = random.Random(seed)
    names = graph.vs["name"]

    def cosine(a, b):
        a, b = embeddings[a], embeddings[b]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0

    edges = [(graph.vs[e.source]["name"], graph.vs[e.target]["name"]) for e in graph.es]
    adjacent_sample = rng.sample(edges, min(sample_size, len(edges)))
    adjacent_sim = sum(cosine(u, v) for u, v in adjacent_sample) / len(adjacent_sample)

    existing = {frozenset(pair) for pair in edges}
    random_sims = []
    attempts = 0
    while len(random_sims) < len(adjacent_sample) and attempts < len(adjacent_sample) * 50:
        attempts += 1
        u, v = rng.sample(names, 2)
        if frozenset((u, v)) in existing:
            continue
        random_sims.append(cosine(u, v))
    random_sim = sum(random_sims) / len(random_sims) if random_sims else 0.0

    passed = adjacent_sim > random_sim
    status = "PASS" if passed else "FAIL"
    print(
        f"Sanity check: adjacent-pair similarity {adjacent_sim:.4f} "
        f"{'>' if passed else '<='} random-pair similarity {random_sim:.4f} ({status})"
    )
    return passed


def save_embeddings(embeddings, path):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)


def load_embeddings(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path):
    torch.save(model.state_dict(), path)
