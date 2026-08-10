#include "sdl/pipeline/worker_pool.hpp"

#include "sdl/generators/seed_util.hpp"
#include "sdl/pipeline/index_stream.hpp"
#include "sdl/pipeline/reorder_buffer.hpp"

#include <thread>
#include <vector>

namespace sdl {

namespace {

void run_as_ready(const Generator& generator,
                   const TransformPipeline& transforms,
                   const WorkerPoolConfig& config,
                   std::uint64_t generation_seed,
                   IndexStream& index_stream,
                   BoundedQueue<Sample>& queue) {
    std::vector<std::thread> workers;
    workers.reserve(config.num_workers);
    for (std::size_t w = 0; w < config.num_workers; ++w) {
        workers.emplace_back([&] {
            while (auto dispensed = index_stream.next()) {
                Sample sample = generator.sample(dispensed->index, generation_seed);
                sample = transforms.apply(std::move(sample));
                queue.push(std::move(sample));
            }
        });
    }
    for (auto& t : workers) {
        t.join();
    }
    queue.close();
}

void run_ordered(const Generator& generator,
                  const TransformPipeline& transforms,
                  const WorkerPoolConfig& config,
                  std::uint64_t generation_seed,
                  IndexStream& index_stream,
                  BoundedQueue<Sample>& queue) {
    // Keyed by dispensing *position* (not the sample index, which may be
    // shuffled) — see DispensedIndex. start_index aligns the window with
    // config.start_position so resumed runs don't expect position 0 first.
    ReorderBuffer<Sample> reorder(config.reorder_depth, config.start_position);

    std::vector<std::thread> workers;
    workers.reserve(config.num_workers);
    for (std::size_t w = 0; w < config.num_workers; ++w) {
        workers.emplace_back([&] {
            while (auto dispensed = index_stream.next()) {
                Sample sample = generator.sample(dispensed->index, generation_seed);
                sample = transforms.apply(std::move(sample));
                reorder.insert(dispensed->position, std::move(sample));
            }
        });
    }

    std::thread relay([&] {
        while (auto sample = reorder.pop()) {
            queue.push(std::move(*sample));
        }
        queue.close();
    });

    for (auto& t : workers) {
        t.join();
    }
    reorder.close();
    relay.join();
}

} // namespace

void run_worker_pool(const Generator& generator,
                      const TransformPipeline& transforms,
                      const WorkerPoolConfig& config,
                      BoundedQueue<Sample>& queue) {
    IndexStream index_stream(config.indices, config.start_position);
    const std::uint64_t generation_seed = mix_seed(config.seed, config.epoch);

    if (config.mode == OrderingMode::kAsReady) {
        run_as_ready(generator, transforms, config, generation_seed, index_stream, queue);
    } else {
        run_ordered(generator, transforms, config, generation_seed, index_stream, queue);
    }
}

} // namespace sdl
