#include "sdl/generators/image_generator.hpp"
#include "sdl/pipeline/mock_consumer.hpp"
#include "sdl/pipeline/producer.hpp"
#include "sdl/queue/bounded_queue.hpp"

#include <benchmark/benchmark.h>

#include <chrono>
#include <cstdint>
#include <thread>

namespace {

// P0: single producer -> bounded queue -> mock consumer. Measures
// samples/sec while sweeping queue depth and simulated per-step compute
// cost (T_step). Args: {num_samples, queue_capacity, t_step_us}.
//
// UseRealTime() is required because T_step sleeps rather than spins — the
// benchmark's default CPU-time measurement would exclude that sleeping time
// and report a misleadingly high throughput.
void BM_P0Pipeline(benchmark::State& state) {
    const auto num_samples = static_cast<std::size_t>(state.range(0));
    const auto capacity = static_cast<std::size_t>(state.range(1));
    const auto t_step = std::chrono::microseconds(state.range(2));

    sdl::ImageGenerator gen(sdl::ImageGenerator::Config{});

    for (auto _ : state) {
        sdl::BoundedQueue<sdl::Sample> queue(capacity);
        std::thread producer([&] { sdl::run_single_producer(gen, num_samples, 42, queue); });
        sdl::ConsumeResult result = sdl::run_mock_consumer(queue, t_step);
        producer.join();
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(num_samples));
}

BENCHMARK(BM_P0Pipeline)
    ->Args({2000, 8, 0})
    ->Args({2000, 64, 0})
    ->Args({2000, 8, 100})
    ->Args({2000, 64, 100})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

} // namespace
