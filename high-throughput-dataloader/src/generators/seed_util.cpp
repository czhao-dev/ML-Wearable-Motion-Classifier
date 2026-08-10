#include "sdl/generators/seed_util.hpp"

namespace sdl {

std::uint64_t mix_seed(std::uint64_t seed, std::uint64_t index) {
    std::uint64_t z = seed + 0x9E3779B97F4A7C15ULL * (index + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return z;
}

std::mt19937_64 make_rng(std::uint64_t seed, std::uint64_t index) {
    return std::mt19937_64(mix_seed(seed, index));
}

} // namespace sdl
