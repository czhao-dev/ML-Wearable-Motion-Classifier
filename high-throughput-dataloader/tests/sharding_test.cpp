#include "sdl/pipeline/sharding.hpp"

#include "sdl/pipeline/epoch_shuffle.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>

namespace sdl {
namespace {

TEST(ShardIndices, SingleRankGetsTheWholeSequenceUnchanged) {
    const auto epoch = epoch_permutation(1, 0, 100);
    const auto shard = shard_indices(epoch, /*rank=*/0, /*world_size=*/1);
    EXPECT_EQ(shard, epoch);
}

// spec 10.2: shards are pairwise disjoint and their union is the full
// epoch. Covers both evenly- and unevenly-divisible cases.
TEST(ShardIndices, ShardsArePairwiseDisjointAndUnionIsFullEpoch) {
    for (std::size_t num_samples : {100u, 101u, 103u, 997u}) {
        for (std::size_t world_size : {1u, 2u, 3u, 7u}) {
            const auto epoch = epoch_permutation(42, 2, num_samples);

            std::vector<std::size_t> seen(num_samples, 0);
            std::size_t total = 0;
            std::size_t max_shard_size = 0;
            std::size_t min_shard_size = num_samples;

            for (std::size_t rank = 0; rank < world_size; ++rank) {
                const auto shard = shard_indices(epoch, rank, world_size);
                max_shard_size = std::max(max_shard_size, shard.size());
                min_shard_size = std::min(min_shard_size, shard.size());
                for (std::size_t index : shard) {
                    ASSERT_LT(index, num_samples);
                    ASSERT_EQ(seen[index], 0u)
                        << "index " << index << " appeared in more than one rank's shard "
                        << "(num_samples=" << num_samples << ", world_size=" << world_size << ")";
                    seen[index] = 1;
                    ++total;
                }
            }

            EXPECT_EQ(total, num_samples) << "num_samples=" << num_samples << " world_size=" << world_size;
            EXPECT_LE(max_shard_size - min_shard_size, 1u)
                << "shard sizes should differ by at most one element";
        }
    }
}

} // namespace
} // namespace sdl
