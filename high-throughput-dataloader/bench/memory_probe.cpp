// Standalone (non-Google-Benchmark) tool: runs ONE pipeline configuration
// to completion and reports peak RSS via getrusage. Deliberately a
// separate process per invocation rather than a Google Benchmark in-process
// loop: peak RSS is a process-lifetime high-water mark, so running several
// configurations in one process would contaminate later readings with
// earlier (possibly larger) ones. Sweep configurations externally (e.g. a
// shell loop) — one process per data point — to get an isolated reading
// for each.
//
// Usage: sdl_memory_probe <num_samples> <num_workers> <prefetch_depth> <batch_size> <cost_iterations> <t_step_us>

#include "sdl/generators/image_generator.hpp"
#include "sdl/pipeline/epoch_shuffle.hpp"
#include "sdl/pipeline/mock_batch_consumer.hpp"
#include "sdl/pipeline/worker_pool.hpp"
#include "sdl/queue/bounded_queue.hpp"

#include <sys/resource.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: " << argv[0]
                   << " <num_samples> <num_workers> <prefetch_depth> <batch_size> <cost_iterations> <t_step_us>\n";
        return 1;
    }

    const auto num_samples = static_cast<std::size_t>(std::stoul(argv[1]));
    const auto num_workers = static_cast<std::size_t>(std::stoul(argv[2]));
    const auto prefetch_depth = static_cast<std::size_t>(std::stoul(argv[3]));
    const auto batch_size = static_cast<std::size_t>(std::stoul(argv[4]));
    const auto cost_iterations = static_cast<std::uint64_t>(std::stoull(argv[5]));
    const auto t_step = std::chrono::microseconds(std::stoll(argv[6]));

    sdl::ImageGenerator::Config gen_config;
    gen_config.channels = 3;
    gen_config.height = 64;
    gen_config.width = 64;
    gen_config.cost_iterations = cost_iterations;
    sdl::ImageGenerator gen(gen_config);
    sdl::TransformPipeline transforms;

    sdl::BoundedQueue<sdl::Sample> queue(prefetch_depth * batch_size);
    sdl::WorkerPoolConfig config;
    config.indices = sdl::identity_sequence(num_samples);
    config.num_workers = num_workers;
    config.seed = 42;

    std::thread pool([&] { sdl::run_worker_pool(gen, transforms, config, queue); });
    const sdl::BatchConsumeResult result = sdl::run_mock_batch_consumer(queue, batch_size, t_step);
    pool.join();

    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
    const double peak_rss_mb = static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0); // bytes on macOS
#else
    const double peak_rss_mb = static_cast<double>(usage.ru_maxrss) / 1024.0; // KB on Linux
#endif

    const double wall_time_ms = std::chrono::duration<double, std::milli>(result.wall_time).count();
    const double idle_time_ms = std::chrono::duration<double, std::milli>(result.idle_time).count();
    const double idle_pct = wall_time_ms > 0 ? 100.0 * idle_time_ms / wall_time_ms : 0.0;

    std::cout << "num_samples=" << num_samples << " num_workers=" << num_workers
              << " prefetch_depth=" << prefetch_depth << " batch_size=" << batch_size
              << " cost_iterations=" << cost_iterations << " t_step_us=" << t_step.count()
              << " samples_consumed=" << result.samples_consumed << " batches_consumed=" << result.batches_consumed
              << " wall_time_ms=" << wall_time_ms << " idle_time_ms=" << idle_time_ms << " idle_pct=" << idle_pct
              << " peak_rss_mb=" << peak_rss_mb << "\n";
    return 0;
}
