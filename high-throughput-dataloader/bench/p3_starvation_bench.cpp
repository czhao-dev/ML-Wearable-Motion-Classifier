#include "sdl/generators/image_generator.hpp"
#include "sdl/pipeline/epoch_shuffle.hpp"
#include "sdl/pipeline/mock_batch_consumer.hpp"
#include "sdl/pipeline/worker_pool.hpp"
#include "sdl/queue/bounded_queue.hpp"

#include <benchmark/benchmark.h>

#include <chrono>
#include <cstdint>
#include <thread>

namespace {

// P3 headline metric (spec section 12): starvation-% (accelerator-idle
// time / wall time) vs. worker count and vs. prefetch depth (queue
// capacity, in samples — batch_size is fixed at 1 here so prefetch_depth
// is directly the sample count), with a simulated per-sample T_step
// standing in for accelerator compute time.
//
// Args: {num_samples, num_workers, prefetch_depth, t_step_us}.
// UseRealTime(): T_step sleeps (see mock_consumer.hpp) and idle_time is
// itself a wall-clock measurement, so CPU-time would be misleading here.
void BM_P3Starvation(benchmark::State& state) {
    const auto num_samples = static_cast<std::size_t>(state.range(0));
    const auto num_workers = static_cast<std::size_t>(state.range(1));
    const auto prefetch_depth = static_cast<std::size_t>(state.range(2));
    const auto t_step = std::chrono::microseconds(state.range(3));
    constexpr std::size_t kBatchSize = 1;

    // cost_iterations is deliberately large: a busy-loop this size dominates
    // over per-sample setup/scheduling noise, so measured idle_pct reflects
    // the real supply/demand balance rather than measurement artifacts.
    // (Calibrated against optimized/release builds — an unoptimized debug
    // build has much higher fixed per-sample overhead, so the same knob
    // value lands in a different regime there; see the project README's
    // benchmarks section for the full discussion.)
    sdl::ImageGenerator::Config gen_config;
    gen_config.channels = 3;
    gen_config.height = 64;
    gen_config.width = 64;
    gen_config.cost_iterations = 20000;
    sdl::ImageGenerator gen(gen_config);
    sdl::TransformPipeline transforms;

    double idle_pct = 0.0;

    for (auto _ : state) {
        sdl::BoundedQueue<sdl::Sample> queue(prefetch_depth * kBatchSize);
        sdl::WorkerPoolConfig config;
        config.indices = sdl::identity_sequence(num_samples);
        config.num_workers = num_workers;
        config.seed = 42;

        std::thread pool([&] { sdl::run_worker_pool(gen, transforms, config, queue); });
        sdl::BatchConsumeResult result = sdl::run_mock_batch_consumer(queue, kBatchSize, t_step);
        pool.join();

        idle_pct = result.wall_time.count() > 0
                       ? 100.0 * static_cast<double>(result.idle_time.count()) /
                             static_cast<double>(result.wall_time.count())
                       : 0.0;
        benchmark::DoNotOptimize(result);
    }

    state.counters["idle_pct"] = idle_pct;
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(num_samples));
}

// t_step_us=0 (an infinitely eager consumer) isolates whether the worker
// pool alone can keep the queue non-empty: on this machine, idle_pct falls
// from ~95% at 1 worker to ~53% by 4 (the knee), per the project README.
// num_samples is large enough (20,000) that the run is long enough for
// worker-thread startup/scheduling jitter not to dominate the measurement.
//
// The prefetch-depth sweep (fixed at 2 workers, still on the starved side
// of that boundary at t_step=0) is included for completeness, but the
// honest finding (see the README) is that depth barely moves idle_pct
// here: this generator's busy-loop cost is deterministic and low-variance,
// so buffering — which smooths *bursty* per-sample timing — has little to
// smooth. Depth mainly pays off against
// variance, not against a sustained supply/demand mismatch; only adding
// workers fixes that.
BENCHMARK(BM_P3Starvation)
    // Starvation vs. worker count, fixed prefetch depth.
    ->Args({20000, 1, 64, 0})
    ->Args({20000, 2, 64, 0})
    ->Args({20000, 4, 64, 0})
    ->Args({20000, 8, 64, 0})
    // Starvation vs. prefetch depth, fixed (still-marginal) worker count.
    ->Args({20000, 2, 1, 0})
    ->Args({20000, 2, 4, 0})
    ->Args({20000, 2, 16, 0})
    ->Args({20000, 2, 64, 0})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

} // namespace
