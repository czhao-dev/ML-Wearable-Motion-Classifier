#include "sdl/pipeline/producer.hpp"

namespace sdl {

void run_single_producer(const Generator& generator,
                          std::size_t num_samples,
                          std::uint64_t seed,
                          BoundedQueue<Sample>& queue) {
    for (std::size_t index = 0; index < num_samples; ++index) {
        queue.push(generator.sample(index, seed));
    }
    queue.close();
}

} // namespace sdl
