#include "sdl/pipeline/worker_pool.hpp"

#include "sdl/generators/image_generator.hpp"
#include "sdl/queue/bounded_queue.hpp"
#include "sdl/transform/scale_transform.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

namespace sdl {
namespace {

std::vector<std::size_t> identity(std::size_t n) {
    std::vector<std::size_t> v(n);
    std::iota(v.begin(), v.end(), std::size_t{0});
    return v;
}

// run_worker_pool() only returns once every producer (and, in ordered mode,
// the reorder relay) has finished — it does not drain `queue` itself. The
// queue is bounded, so the consumer must drain it *concurrently* on another
// thread; draining only after run_worker_pool() returns would deadlock as
// soon as producers fill the queue to capacity with nobody popping.
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

// P1 DoD: as-ready mode is correct — exactly-once over the epoch, no
// duplicates or drops, regardless of worker count. Delivery order is not
// asserted here (it is explicitly not guaranteed in as-ready mode).
TEST(WorkerPool, AsReadyModeIsExactlyOnceAcrossWorkerCounts) {
    ImageGenerator gen(ImageGenerator::Config{});
    TransformPipeline transforms;
    constexpr std::size_t kNumSamples = 5000;

    for (std::size_t num_workers : {1u, 2u, 4u, 8u}) {
        BoundedQueue<Sample> queue(16);
        WorkerPoolConfig config;
        config.indices = identity(kNumSamples);
        config.num_workers = num_workers;
        config.seed = 42;
        config.mode = OrderingMode::kAsReady;

        std::vector<Sample> samples = run_and_drain(gen, transforms, config, queue);

        ASSERT_EQ(samples.size(), kNumSamples) << "num_workers=" << num_workers;
        std::vector<std::size_t> indices;
        indices.reserve(samples.size());
        for (const auto& s : samples) {
            indices.push_back(s.index);
        }
        std::sort(indices.begin(), indices.end());
        for (std::size_t i = 0; i < kNumSamples; ++i) {
            ASSERT_EQ(indices[i], i) << "num_workers=" << num_workers << " index=" << i;
        }
    }
}

// P1 DoD / spec 10.2 "worker-count invariance": in ordered mode, 1 worker
// and 8 workers must produce the identical ordered sequence. This is the
// direct race detector for the reorder buffer.
TEST(WorkerPool, OrderedModeIsWorkerCountInvariant) {
    ImageGenerator gen(ImageGenerator::Config{});
    TransformPipeline transforms;
    constexpr std::size_t kNumSamples = 3000;

    std::vector<std::vector<std::size_t>> index_sequences;
    std::vector<std::vector<Sample>> sample_sequences;

    for (std::size_t num_workers : {1u, 2u, 4u, 8u}) {
        BoundedQueue<Sample> queue(16);
        WorkerPoolConfig config;
        config.indices = identity(kNumSamples);
        config.num_workers = num_workers;
        config.seed = 42;
        config.mode = OrderingMode::kOrdered;
        config.reorder_depth = 32; // comfortably >= max in-flight indices (bounded by num_workers)

        std::vector<Sample> samples = run_and_drain(gen, transforms, config, queue);
        ASSERT_EQ(samples.size(), kNumSamples) << "num_workers=" << num_workers;

        std::vector<std::size_t> indices;
        indices.reserve(samples.size());
        for (const auto& s : samples) {
            indices.push_back(s.index);
        }
        index_sequences.push_back(std::move(indices));
        sample_sequences.push_back(std::move(samples));
    }

    // Strict ascending order within each run.
    for (const auto& indices : index_sequences) {
        for (std::size_t i = 0; i < indices.size(); ++i) {
            ASSERT_EQ(indices[i], i);
        }
    }

    // Identical content across every worker count, not just identical index
    // order — the generated Sample data itself must match too.
    for (std::size_t i = 1; i < sample_sequences.size(); ++i) {
        ASSERT_EQ(sample_sequences[i], sample_sequences[0]);
    }
}

TEST(WorkerPool, TransformIsAppliedToEverySample) {
    ImageGenerator gen(ImageGenerator::Config{});
    TransformPipeline transforms({std::make_shared<ScaleTransform>(0.0f)});
    BoundedQueue<Sample> queue(16);

    constexpr std::size_t kNumSamples = 200;
    WorkerPoolConfig config;
    config.indices = identity(kNumSamples);
    config.num_workers = 4;
    config.seed = 7;
    config.mode = OrderingMode::kAsReady;

    std::vector<Sample> samples = run_and_drain(gen, transforms, config, queue);

    ASSERT_EQ(samples.size(), kNumSamples);
    for (const auto& s : samples) {
        for (float v : s.data) {
            EXPECT_FLOAT_EQ(v, 0.0f);
        }
    }
}

} // namespace
} // namespace sdl
