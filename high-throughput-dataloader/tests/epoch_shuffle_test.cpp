#include "sdl/pipeline/epoch_shuffle.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

namespace sdl {
namespace {

std::vector<std::size_t> sorted(std::vector<std::size_t> v) {
    std::sort(v.begin(), v.end());
    return v;
}

TEST(EpochPermutation, IsABijectionOverTheFullEpoch) {
    for (std::size_t n : {0u, 1u, 2u, 37u, 1000u}) {
        const std::vector<std::size_t> perm = epoch_permutation(/*seed=*/7, /*epoch=*/3, n);
        ASSERT_EQ(perm.size(), n);
        EXPECT_EQ(sorted(perm), identity_sequence(n)) << "n=" << n;
    }
}

TEST(EpochPermutation, DeterministicForSameSeedAndEpoch) {
    const auto a = epoch_permutation(42, 5, 2000);
    const auto b = epoch_permutation(42, 5, 2000);
    EXPECT_EQ(a, b);
}

TEST(EpochPermutation, DifferentSeedsProduceDifferentOrder) {
    const auto a = epoch_permutation(1, 0, 2000);
    const auto b = epoch_permutation(2, 0, 2000);
    EXPECT_NE(a, b);
}

TEST(EpochPermutation, DifferentEpochsProduceDifferentOrder) {
    const auto a = epoch_permutation(42, 0, 2000);
    const auto b = epoch_permutation(42, 1, 2000);
    EXPECT_NE(a, b);
}

TEST(IdentitySequence, IsUnshuffledAscendingRange) {
    EXPECT_EQ(identity_sequence(5), (std::vector<std::size_t>{0, 1, 2, 3, 4}));
    EXPECT_TRUE(identity_sequence(0).empty());
}

} // namespace
} // namespace sdl
