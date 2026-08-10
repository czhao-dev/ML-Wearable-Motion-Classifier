#include "sdl/pipeline/epoch_shuffle.hpp"
#include "sdl/pipeline/sharding.hpp"
#include "sdl/pipeline/worker_pool.hpp"

#include "sdl/generators/image_generator.hpp"
#include "sdl/queue/bounded_queue.hpp"

#include <gtest/gtest.h>

#include <thread>
#include <vector>

namespace sdl {
namespace {

// See tests/worker_pool_test.cpp for why draining must happen concurrently
// with run_worker_pool(), not after it returns.
std::vector<Sample> run_and_drain(const Generator& gen,
                                   const TransformPipeline& transforms,
                                   const WorkerPoolConfig& config,
                                   BoundedQueue<Sample>& queue) {
    std::thread pool([&] { run_worker_pool(gen, transforms, config, queue); });
    std::vector<Sample> out;
    while (auto sample = queue.pop()) {
        out.push_back(std::move(*sample));
    }
    pool.join();
    return out;
}

// spec 10.2 "seed reproducibility": same (seed, epoch) -> identical ordered
// sequence, content included, across independent runs.
TEST(P2Pipeline, SeedAndEpochReproducibilityAcrossIndependentRuns) {
    ImageGenerator gen(ImageGenerator::Config{});
    TransformPipeline transforms;

    constexpr std::uint64_t seed = 123;
    constexpr std::size_t epoch = 4;
    constexpr std::size_t num_samples = 2000;

    const auto shard = shard_indices(epoch_permutation(seed, epoch, num_samples), /*rank=*/0, /*world_size=*/1);

    WorkerPoolConfig config;
    config.indices = shard;
    config.num_workers = 4;
    config.seed = seed;
    config.epoch = epoch;
    config.mode = OrderingMode::kOrdered;
    config.reorder_depth = 32;

    BoundedQueue<Sample> queue_a(16);
    std::vector<Sample> run_a = run_and_drain(gen, transforms, config, queue_a);

    BoundedQueue<Sample> queue_b(16);
    std::vector<Sample> run_b = run_and_drain(gen, transforms, config, queue_b);

    ASSERT_EQ(run_a.size(), num_samples);
    EXPECT_EQ(run_a, run_b);
}

// spec 10.2 "worker-count invariance," now with the real shuffle+shard
// pipeline (not just an identity range as in Milestone 5's version): in
// ordered mode, the delivered sequence must equal shard_indices(...) in
// exact order, regardless of worker count.
TEST(P2Pipeline, OrderedModeMatchesCanonicalShuffledShardAcrossWorkerCounts) {
    ImageGenerator gen(ImageGenerator::Config{});
    TransformPipeline transforms;

    constexpr std::uint64_t seed = 7;
    constexpr std::size_t epoch = 1;
    constexpr std::size_t num_samples = 3000;
    constexpr std::size_t rank = 1;
    constexpr std::size_t world_size = 3;

    const auto canonical_shard = shard_indices(epoch_permutation(seed, epoch, num_samples), rank, world_size);

    for (std::size_t num_workers : {1u, 2u, 4u, 8u}) {
        WorkerPoolConfig config;
        config.indices = canonical_shard;
        config.num_workers = num_workers;
        config.seed = seed;
        config.epoch = epoch;
        config.mode = OrderingMode::kOrdered;
        config.reorder_depth = 32;

        BoundedQueue<Sample> queue(16);
        std::vector<Sample> samples = run_and_drain(gen, transforms, config, queue);

        ASSERT_EQ(samples.size(), canonical_shard.size()) << "num_workers=" << num_workers;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            ASSERT_EQ(samples[i].index, canonical_shard[i]) << "num_workers=" << num_workers << " position=" << i;
        }
    }
}

// spec 10.4 resumption, combined with the decision that checkpointing
// records the CONSUMER's observed position, not the producer's cursor:
// samples already generated/queued-but-unconsumed beyond the checkpoint are
// discarded (never replayed), and a fresh run starting at consumer_position
// reproduces exactly the missing suffix — so pre ++ post == uninterrupted.
TEST(P2Pipeline, ResumeAfterInFlightSamplesDiscardsAndRegeneratesExactSuffix) {
    ImageGenerator gen(ImageGenerator::Config{});
    TransformPipeline transforms;

    constexpr std::uint64_t seed = 42;
    constexpr std::size_t epoch = 0;
    constexpr std::size_t num_samples = 2000;
    constexpr std::size_t checkpoint_at = 500;

    const auto canonical = shard_indices(epoch_permutation(seed, epoch, num_samples), /*rank=*/0, /*world_size=*/1);

    WorkerPoolConfig config1;
    config1.indices = canonical;
    config1.start_position = 0;
    config1.num_workers = 4;
    config1.seed = seed;
    config1.epoch = epoch;
    config1.mode = OrderingMode::kOrdered;
    config1.reorder_depth = 16;

    // --- Run 1: consume only the first `checkpoint_at` samples, then stop
    // listening for the "real" result. Production continues in the
    // background (workers may already have generated/queued samples beyond
    // checkpoint_at); everything past checkpoint_at is drained and thrown
    // away here, exactly as a crash would lose it. ---
    BoundedQueue<Sample> queue1(8); // small: encourages genuine in-flight samples
    std::thread pool1([&] { run_worker_pool(gen, transforms, config1, queue1); });

    std::vector<Sample> pre;
    for (std::size_t i = 0; i < checkpoint_at; ++i) {
        auto sample = queue1.pop();
        ASSERT_TRUE(sample.has_value());
        pre.push_back(std::move(*sample));
    }
    while (queue1.pop()) {
        // Discarded: simulates in-flight samples lost at checkpoint time.
    }
    pool1.join();

    // --- Run 2: fresh pipeline resuming at consumer_position. ---
    WorkerPoolConfig config2 = config1;
    config2.start_position = checkpoint_at;

    BoundedQueue<Sample> queue2(8);
    std::vector<Sample> post = run_and_drain(gen, transforms, config2, queue2);

    ASSERT_EQ(pre.size(), checkpoint_at);
    ASSERT_EQ(post.size(), num_samples - checkpoint_at);

    for (std::size_t i = 0; i < checkpoint_at; ++i) {
        EXPECT_EQ(pre[i].index, canonical[i]) << "pre position=" << i;
    }
    for (std::size_t i = 0; i < post.size(); ++i) {
        EXPECT_EQ(post[i].index, canonical[checkpoint_at + i]) << "post position=" << i;
    }
}

} // namespace
} // namespace sdl
