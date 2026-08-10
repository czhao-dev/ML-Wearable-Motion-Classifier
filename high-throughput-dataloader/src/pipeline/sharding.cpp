#include "sdl/pipeline/sharding.hpp"

namespace sdl {

std::vector<std::size_t> shard_indices(const std::vector<std::size_t>& epoch_sequence,
                                        std::size_t rank,
                                        std::size_t world_size) {
    std::vector<std::size_t> shard;
    shard.reserve(epoch_sequence.size() / world_size + 1);
    for (std::size_t pos = rank; pos < epoch_sequence.size(); pos += world_size) {
        shard.push_back(epoch_sequence[pos]);
    }
    return shard;
}

} // namespace sdl
