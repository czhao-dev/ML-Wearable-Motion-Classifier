#pragma once

#include "sdl/queue/bounded_queue.hpp"
#include "sdl/sample.hpp"

#include <chrono>
#include <cstddef>

namespace sdl {

struct BatchConsumeResult {
    std::size_t samples_consumed = 0;
    std::size_t batches_consumed = 0;
    std::chrono::nanoseconds wall_time{0};

    // See ConsumeResult::idle_time (mock_consumer.hpp) — same approximation,
    // summed over every queue.pop() call made while assembling batches.
    std::chrono::nanoseconds idle_time{0};
};

// Pops samples from `queue`, collates them into batches of `batch_size`
// (the final, possibly-ragged batch is emitted once the queue is closed
// and drained), sleeping `t_step` after each batch to simulate a training
// step's compute time. See mock_consumer.hpp for why this is a sleep, not a
// busy-loop.
BatchConsumeResult run_mock_batch_consumer(BoundedQueue<Sample>& queue,
                                            std::size_t batch_size,
                                            std::chrono::nanoseconds t_step);

} // namespace sdl
