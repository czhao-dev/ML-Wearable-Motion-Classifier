#pragma once

#include <cstdint>
#include <random>

namespace sdl {

// splitmix64-style mix of a base seed and a sample index. Nearby indices must
// not produce correlated RNG streams, and the result must depend only on
// (seed, index) — never on call order or which thread called it — so that
// generation stays position-pure regardless of worker count.
std::uint64_t mix_seed(std::uint64_t seed, std::uint64_t index);

// An RNG engine seeded purely from (seed, index).
std::mt19937_64 make_rng(std::uint64_t seed, std::uint64_t index);

} // namespace sdl
