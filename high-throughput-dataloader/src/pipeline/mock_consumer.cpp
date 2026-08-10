#include "sdl/pipeline/mock_consumer.hpp"

#include <thread>

namespace sdl {

ConsumeResult run_mock_consumer(BoundedQueue<Sample>& queue, std::chrono::nanoseconds t_step) {
    const auto start = std::chrono::steady_clock::now();
    std::size_t count = 0;
    std::chrono::nanoseconds idle{0};

    while (true) {
        const auto pop_start = std::chrono::steady_clock::now();
        auto sample = queue.pop();
        idle += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - pop_start);
        if (!sample) {
            break;
        }
        if (t_step.count() > 0) {
            std::this_thread::sleep_for(t_step);
        }
        ++count;
    }

    const auto end = std::chrono::steady_clock::now();
    return ConsumeResult{count, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start), idle};
}

} // namespace sdl
