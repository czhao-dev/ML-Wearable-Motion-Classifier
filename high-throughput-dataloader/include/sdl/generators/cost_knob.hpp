#pragma once

#include <cstdint>

namespace sdl {

// Burns CPU for approximately `iterations` busy-loop steps to simulate an
// expensive decode/augment stage. A busy loop (not a sleep) is used
// deliberately: it competes for CPU with real worker threads, so starvation
// benchmarks reflect genuine compute contention rather than idle sleeping.
// Never affects generator output, only wall-clock cost.
void busy_spin(std::uint64_t iterations);

} // namespace sdl
