#include "sdl/generators/image_generator.hpp"
#include "sdl/pipeline/mock_consumer.hpp"
#include "sdl/pipeline/worker_pool.hpp"
#include "sdl/queue/bounded_queue.hpp"

#include <benchmark/benchmark.h>

#include <chrono>
#include <cstdint>
#include <numeric>
#include <thread>
#include <vector>

namespace {

// P1: N-worker pool -> bounded queue -> mock consumer. Measures samples/sec
// vs. worker count and ordering mode, with a non-trivial per-sample
// generation cost (busy-loop cost knob) so the curve reflects real
// parallel scaling rather than thread-overhead noise.
//
// Args: {num_samples, num_workers, mode} where mode: 0 = as-ready,
// 1 = ordered. UseRealTime() because the mock consumer's T_step (here 0,
// but the harness is shared with P0) and thread synchronization are
// wall-clock phenomena, not CPU-time ones.
void BM_P1WorkerPool(benchmark::State& state) {
    const auto num_samples = static_cast<std::size_t>(state.range(0));
    const auto num_workers = static_cast<std::size_t>(state.range(1));
    const auto mode = state.range(2) == 0 ? sdl::OrderingMode::kAsReady : sdl::OrderingMode::kOrdered;

    sdl::ImageGenerator::Config gen_config;
    gen_config.channels = 3;
    gen_config.height = 64;
    gen_config.width = 64;
    gen_config.cost_iterations = 20000;
    sdl::ImageGenerator gen(gen_config);
    sdl::TransformPipeline transforms;

    for (auto _ : state) {
        sdl::BoundedQueue<sdl::Sample> queue(64);
        std::vector<std::size_t> indices(num_samples);
        std::iota(indices.begin(), indices.end(), std::size_t{0});

        sdl::WorkerPoolConfig config;
        config.indices = std::move(indices);
        config.num_workers = num_workers;
        config.seed = 42;
        config.mode = mode;
        config.reorder_depth = num_workers * 4;

        std::thread pool([&] { sdl::run_worker_pool(gen, transforms, config, queue); });
        sdl::ConsumeResult result = sdl::run_mock_consumer(queue, std::chrono::microseconds(0));
        pool.join();
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(num_samples));
}

BENCHMARK(BM_P1WorkerPool)
    ->Args({4000, 1, 0})
    ->Args({4000, 2, 0})
    ->Args({4000, 4, 0})
    ->Args({4000, 8, 0})
    ->Args({4000, 1, 1})
    ->Args({4000, 2, 1})
    ->Args({4000, 4, 1})
    ->Args({4000, 8, 1})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

} // namespace
