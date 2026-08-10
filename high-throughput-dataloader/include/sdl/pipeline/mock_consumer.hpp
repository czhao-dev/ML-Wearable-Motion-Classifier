#pragma once

#include "sdl/queue/bounded_queue.hpp"
#include "sdl/sample.hpp"

#include <chrono>
#include <cstddef>

namespace sdl {

struct ConsumeResult {
    std::size_t samples_consumed = 0;
    std::chrono::nanoseconds wall_time{0};

    // Approximate total time spent blocked inside queue.pop() waiting for a
    // sample. A successful non-blocking pop costs only lock overhead
    // (near-zero), so summing every pop() call's duration closely
    // approximates true starvation (accelerator-idle) time without needing
    // separate instrumentation inside BoundedQueue itself. idle_time /
    // wall_time is the starvation-% metric from spec section 12.
    std::chrono::nanoseconds idle_time{0};
};

// Pops samples from the queue until it is closed and drained, sleeping
// `t_step` after each pop to simulate a training step's compute time. A
// sleep (not a busy-loop) is deliberate: T_step models time spent on a
// separate accelerator, which must not compete with producer worker threads
// for host CPU the way the generator's busy-loop cost knob does.
ConsumeResult run_mock_consumer(BoundedQueue<Sample>& queue, std::chrono::nanoseconds t_step);

} // namespace sdl
