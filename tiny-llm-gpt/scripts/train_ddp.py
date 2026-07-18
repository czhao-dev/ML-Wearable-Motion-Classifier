"""Multi-process DistributedDataParallel training, additive to scripts/train.py.

Launch with torchrun, e.g. two ranks sharing one physical GPU:

    torchrun --nproc_per_node=2 scripts/train_ddp.py --config configs/small.yaml \
        --max-steps-override 2000

This is a DDP-mechanics-and-throughput demo, not a rerun of the full model-
scaling experiment: it exists to show correct DDP wiring (process group init,
gradient all-reduce via DDP, per-rank data sharding, rank-0-only logging and
checkpointing) and to measure what running two ranks on a single GPU actually
costs, not to reproduce scripts/train.py's full 20,000-step runs. See
scripts/train.py for the single-process baseline this is compared against.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tinyllm.model import GPT
from tinyllm.utils import load_config, save_checkpoint, set_seed


def shard_offset_bounds(usable_len: int, world_size: int, rank: int) -> tuple[int, int]:
    """Split the valid window-start-offset range [0, usable_len) into world_size
    contiguous, non-overlapping shards and return this rank's [start, end).

    Pure integer arithmetic, no I/O -- kept separate from get_batch_sharded so
    it's unit-testable without a real token file (see tests/test_ddp_sharding.py).
    """
    if world_size < 1 or not (0 <= rank < world_size):
        raise ValueError(f"invalid rank/world_size: rank={rank}, world_size={world_size}")
    if usable_len < world_size:
        raise ValueError(f"usable_len={usable_len} too small to shard across world_size={world_size}")
    shard_size = usable_len // world_size
    start = rank * shard_size
    end = start + shard_size if rank < world_size - 1 else usable_len
    return start, end


def get_batch_sharded(bin_path: str, batch_size: int, context_length: int, device, rank: int, world_size: int):
    """Same sampling as tinyllm.dataset.get_batch, restricted to this rank's shard
    of the token stream so ranks never train on the same offsets."""
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    usable_len = len(data) - context_length
    start, end = shard_offset_bounds(usable_len, world_size, rank)
    idxs = np.random.randint(start, end, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i:i + context_length].astype(np.int64)) for i in idxs])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + context_length].astype(np.int64)) for i in idxs])
    return x.to(device), y.to(device)


def get_lr(step: int, warmup_steps: int, max_steps: int, peak_lr: float) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return peak_lr * 0.1
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * 0.1 + coeff * peak_lr * 0.9


def build_optimizer(model, lr: float, weight_decay: float):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


@torch.no_grad()
def estimate_loss(model, data_path, batch_size, context_length, device, eval_iters, rank, world_size):
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch_sharded(data_path, batch_size, context_length, device, rank, world_size)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps-override", type=int, default=None)
    parser.add_argument("--backend", default="nccl", choices=["nccl", "gloo"],
                         help="NCCL is the default for CUDA; fall back to gloo if a "
                              "given NCCL build rejects multiple ranks sharing one GPU.")
    parser.add_argument("--run-suffix", default="-ddp",
                         help="Appended to the config's checkpoint_dir/log_dir so this "
                              "run doesn't overwrite scripts/train.py's single-process logs.")
    args = parser.parse_args()

    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    is_main = rank == 0

    cfg = load_config(args.config)
    m, t, d, o = cfg["model"], cfg["training"], cfg["data"], cfg["output"]

    # Decorrelate per-rank sampling RNG state on top of the shared base seed;
    # shard disjointness already prevents ranks from training on the same
    # offsets, this just avoids any incidental cross-rank RNG correlation.
    set_seed(t["seed"])
    np.random.seed(t["seed"] + rank)

    if torch.cuda.is_available():
        device_index = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device("cpu")

    if is_main:
        print(f"world_size={world_size} backend={args.backend} device={device} "
              f"(torch.cuda.device_count()={torch.cuda.device_count() if torch.cuda.is_available() else 0})")

    model = GPT(**m).to(device)
    if world_size > 1:
        ddp_kwargs = {"device_ids": [device.index], "output_device": device.index} if device.type == "cuda" else {}
        model = DDP(model, **ddp_kwargs)
    raw_model = model.module if isinstance(model, DDP) else model

    optimizer = build_optimizer(model, t["learning_rate"], t["weight_decay"])

    checkpoint_dir = o["checkpoint_dir"].rstrip("/") + args.run_suffix
    log_dir = o["log_dir"].rstrip("/") + args.run_suffix
    log_file = None
    if is_main:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        log_file = open(os.path.join(log_dir, "train_log.jsonl"), "a")

    max_steps = args.max_steps_override or t["max_steps"]
    best_val_loss = float("inf")

    t0 = time.time()
    tokens_processed = 0
    for step in range(max_steps):
        lr = get_lr(step, t["warmup_steps"], max_steps, t["learning_rate"])
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch_sharded(d["train_path"], t["batch_size"], m["context_length"], device, rank, world_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # DDP all-reduces gradients across ranks here
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        optimizer.step()
        # Each rank processes batch_size * context_length tokens per step;
        # world_size ranks running in parallel means world_size times that
        # many tokens/sec in aggregate, same accounting as scripts/train.py's
        # per-process tokens_processed so the two are directly comparable.
        tokens_processed += t["batch_size"] * m["context_length"] * world_size

        if is_main and (step % t["eval_interval"] == 0 or step == max_steps - 1):
            val_loss = estimate_loss(
                model, d["valid_path"], t["batch_size"], m["context_length"], device, t["eval_iters"], rank, world_size
            )
            elapsed = time.time() - t0
            tps = tokens_processed / elapsed if elapsed > 0 else 0.0
            record = {
                "step": step,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "lr": lr,
                "tokens_per_sec": tps,
                "world_size": world_size,
                "timestamp": time.time(),
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            print(f"step {step}: train_loss {loss.item():.4f}, val_loss {val_loss:.4f}, "
                  f"lr {lr:.2e}, tok/s {tps:.0f}, world_size {world_size}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(os.path.join(checkpoint_dir, "best.pt"), raw_model, optimizer, step, cfg)

        if is_main and (step % t["checkpoint_interval"] == 0 or step == max_steps - 1):
            save_checkpoint(os.path.join(checkpoint_dir, "latest.pt"), raw_model, optimizer, step, cfg)

    if is_main:
        log_file.close()
        print("training complete")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
