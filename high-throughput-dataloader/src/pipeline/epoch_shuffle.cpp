#include "sdl/pipeline/epoch_shuffle.hpp"

#include "sdl/generators/seed_util.hpp"

#include <numeric>
#include <random>

namespace sdl {

std::vector<std::size_t> epoch_permutation(std::uint64_t seed, std::size_t epoch, std::size_t num_samples) {
    std::vector<std::size_t> perm(num_samples);
    std::iota(perm.begin(), perm.end(), std::size_t{0});

    auto rng = make_rng(seed, epoch);
    for (std::size_t i = num_samples; i > 1; --i) {
        std::uniform_int_distribution<std::size_t> dist(0, i - 1);
        std::swap(perm[i - 1], perm[dist(rng)]);
    }
    return perm;
}

std::vector<std::size_t> identity_sequence(std::size_t num_samples) {
    std::vector<std::size_t> seq(num_samples);
    std::iota(seq.begin(), seq.end(), std::size_t{0});
    return seq;
}

} // namespace sdl
