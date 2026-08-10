#include "sdl/generators/image_generator.hpp"
#include "sdl/pipeline/epoch_shuffle.hpp"
#include "sdl/pipeline/mock_consumer.hpp"
#include "sdl/pipeline/worker_pool.hpp"
#include "sdl/queue/bounded_queue.hpp"

#include <benchmark/benchmark.h>

#include <chrono>
#include <cstdint>
#include <thread>

namespace {

// P3 (spec section 11): throughput vs. augmentation cost, sweeping the
// generator's busy-loop cost knob at a fixed worker count, to show where
// the pipeline transitions from synchronization/queue-overhead-bound (cheap
// samples) to compute-bound (expensive samples).
//
// Args: {num_samples, cost_iterations}.
void BM_P3CostSweep(benchmark::State& state) {
    const auto num_samples = static_cast<std::size_t>(state.range(0));
    const auto cost_iterations = static_cast<std::uint64_t>(state.range(1));

    sdl::ImageGenerator::Config gen_config;
    gen_config.channels = 3;
    gen_config.height = 64;
    gen_config.width = 64;
    gen_config.cost_iterations = cost_iterations;
    sdl::ImageGenerator gen(gen_config);
    sdl::TransformPipeline transforms;

    constexpr std::size_t kNumWorkers = 4;

    for (auto _ : state) {
        sdl::BoundedQueue<sdl::Sample> queue(256);
        sdl::WorkerPoolConfig config;
        config.indices = sdl::identity_sequence(num_samples);
        config.num_workers = kNumWorkers;
        config.seed = 42;

        std::thread pool([&] { sdl::run_worker_pool(gen, transforms, config, queue); });
        sdl::ConsumeResult result = sdl::run_mock_consumer(queue, std::chrono::microseconds(0));
        pool.join();
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(num_samples));
}

BENCHMARK(BM_P3CostSweep)
    ->Args({4000, 0})
    ->Args({4000, 1000})
    ->Args({4000, 5000})
    ->Args({4000, 20000})
    ->Args({4000, 100000})
    ->Args({4000, 300000})
    ->Args({4000, 1000000})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

} // namespace
