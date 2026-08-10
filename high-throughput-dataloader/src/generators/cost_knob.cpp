#include "sdl/generators/cost_knob.hpp"

namespace sdl {

void busy_spin(std::uint64_t iterations) {
    volatile std::uint64_t sink = 0;
    for (std::uint64_t i = 0; i < iterations; ++i) {
        sink += i;
    }
}

} // namespace sdl
