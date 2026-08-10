#include "sdl/generators/image_generator.hpp"
#include "sdl/pipeline/epoch_shuffle.hpp"
#include "sdl/pipeline/mock_consumer.hpp"
#include "sdl/pipeline/worker_pool.hpp"
#include "sdl/queue/bounded_queue.hpp"

#include <gtest/gtest.h>

#include <thread>

namespace sdl {
namespace {

ConsumeResult run_pipeline(ImageGenerator::Config gen_config,
                           std::size_t num_workers,
                           std::size_t queue_capacity,
                           std::size_t num_samples,
                           std::chrono::nanoseconds t_step) {
    ImageGenerator gen(gen_config);
    TransformPipeline transforms;
    BoundedQueue<Sample> queue(queue_capacity);

    WorkerPoolConfig config;
    config.indices = identity_sequence(num_samples);
    config.num_workers = num_workers;
    config.seed = 42;

    std::thread pool([&] { run_worker_pool(gen, transforms, config, queue); });
    ConsumeResult result = run_mock_consumer(queue, t_step);
    pool.join();
    return result;
}

TEST(Starvation, IdleTimeNeverExceedsWallTime) {
    ImageGenerator::Config gen_config;
    const ConsumeResult result = run_pipeline(gen_config, /*num_workers=*/2, /*queue_capacity=*/8,
                                               /*num_samples=*/500, std::chrono::microseconds(50));
    EXPECT_LE(result.idle_time, result.wall_time);
}

// Slow single producer + instant consumer: the consumer should spend most
// of its time blocked waiting on an empty queue (high starvation).
TEST(Starvation, SlowProducerYieldsHighIdleFraction) {
    ImageGenerator::Config gen_config;
    gen_config.cost_iterations = 500000; // expensive generation, single worker
    const ConsumeResult result =
        run_pipeline(gen_config, /*num_workers=*/1, /*queue_capacity=*/8, /*num_samples=*/50, std::chrono::nanoseconds(0));

    ASSERT_GT(result.wall_time.count(), 0);
    const double idle_fraction =
        static_cast<double>(result.idle_time.count()) / static_cast<double>(result.wall_time.count());
    EXPECT_GT(idle_fraction, 0.5) << "expected the consumer to be starved most of the time";
}

// Many cheap producers + a slow consumer: the queue should stay populated,
// so the consumer rarely blocks (low starvation).
TEST(Starvation, FastProducersAndSlowConsumerYieldLowIdleFraction) {
    ImageGenerator::Config gen_config; // default: cheap generation
    const ConsumeResult result = run_pipeline(gen_config, /*num_workers=*/8, /*queue_capacity=*/64,
                                               /*num_samples=*/500, std::chrono::milliseconds(2));

    ASSERT_GT(result.wall_time.count(), 0);
    const double idle_fraction =
        static_cast<double>(result.idle_time.count()) / static_cast<double>(result.wall_time.count());
    EXPECT_LT(idle_fraction, 0.2) << "expected the well-fed queue to rarely starve the consumer";
}

} // namespace
} // namespace sdl
