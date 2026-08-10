#include "sdl/pipeline/mock_batch_consumer.hpp"

#include "sdl/collate/collate.hpp"

#include <thread>
#include <vector>

namespace sdl {

BatchConsumeResult run_mock_batch_consumer(BoundedQueue<Sample>& queue,
                                            std::size_t batch_size,
                                            std::chrono::nanoseconds t_step) {
    const auto start = std::chrono::steady_clock::now();
    BatchConsumeResult result;

    std::vector<Sample> pending;
    pending.reserve(batch_size);

    auto flush = [&] {
        if (pending.empty()) {
            return;
        }
        result.samples_consumed += pending.size();
        ++result.batches_consumed;
        Batch batch = collate(std::move(pending));
        pending.clear();
        (void)batch;
        if (t_step.count() > 0) {
            std::this_thread::sleep_for(t_step);
        }
    };

    while (true) {
        const auto pop_start = std::chrono::steady_clock::now();
        auto sample = queue.pop();
        result.idle_time +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - pop_start);
        if (!sample) {
            break;
        }
        pending.push_back(std::move(*sample));
        if (pending.size() == batch_size) {
            flush();
        }
    }
    flush(); // final, possibly-ragged batch

    const auto end = std::chrono::steady_clock::now();
    result.wall_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    return result;
}

} // namespace sdl
