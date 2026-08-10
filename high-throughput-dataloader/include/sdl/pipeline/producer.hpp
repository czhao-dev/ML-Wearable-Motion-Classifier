#pragma once

#include "sdl/generators/generator.hpp"
#include "sdl/queue/bounded_queue.hpp"
#include "sdl/sample.hpp"

#include <cstddef>
#include <cstdint>

namespace sdl {

// Pulls sequential indices [0, num_samples) on a single producer, runs the
// generator, and pushes each Sample into the queue; closes the queue once
// all indices have been produced. This is the P0 single-worker producer —
// P1 generalizes it to N workers pulling from a shared IndexStream.
void run_single_producer(const Generator& generator,
                          std::size_t num_samples,
                          std::uint64_t seed,
                          BoundedQueue<Sample>& queue);

} // namespace sdl
