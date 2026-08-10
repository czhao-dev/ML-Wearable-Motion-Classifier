#pragma once

#include "sdl/generators/generator.hpp"
#include "sdl/queue/bounded_queue.hpp"
#include "sdl/sample.hpp"
#include "sdl/transform/transform_pipeline.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sdl {

enum class OrderingMode {
    kAsReady, // default, fastest: batches formed from whatever completes first
    kOrdered, // strict epoch order, at the cost of reorder-buffer overhead
};

struct WorkerPoolConfig {
    // This rank's ordered index sequence for the epoch (already shuffled
    // and sharded by the caller, e.g. via epoch_permutation + shard_indices;
    // pass identity_sequence(num_samples) for unsharded/unshuffled use).
    std::vector<std::size_t> indices;

    // Resume point within `indices`: on restore this is the checkpoint's
    // consumer_position, so production restarts exactly there rather than
    // from wherever an abandoned run's producers happened to reach. See
    // Checkpoint (checkpoint.hpp) for the full contract.
    std::size_t start_position = 0;

    std::size_t num_workers = 1;

    // Base seed for sample generation. Internally combined with `epoch` (so
    // the same index generates different content across epochs, matching
    // the spec's "derive per-index RNG seeds from a hash of (seed, epoch,
    // index)" determinism contract) before being passed to Generator::sample.
    std::uint64_t seed = 0;
    std::size_t epoch = 0;

    OrderingMode mode = OrderingMode::kAsReady;
    std::size_t reorder_depth = 1; // only used when mode == kOrdered
};

// Runs `config.num_workers` producer threads pulling (position, index)
// pairs from a shared IndexStream over `config.indices`, generating and
// transforming each Sample, and delivering it to `queue`.
//
// kAsReady: samples are pushed as soon as each worker finishes one — the
// delivery order to `queue` is not deterministic across runs (worker
// scheduling dependent), though the *set* of (index, Sample) pairs produced
// always is.
//
// kOrdered: samples pass through a ReorderBuffer of capacity
// `reorder_depth`, keyed by dispensing *position* (not the sample index,
// which may be shuffled), so `queue` receives them in the same order
// `config.indices` lists them — regardless of worker count — at the cost
// of throughput bounded by the slowest in-flight worker producing the next
// needed position.
//
// Closes `queue` once all workers (and, in ordered mode, the reorder relay)
// have finished.
void run_worker_pool(const Generator& generator,
                      const TransformPipeline& transforms,
                      const WorkerPoolConfig& config,
                      BoundedQueue<Sample>& queue);

} // namespace sdl
