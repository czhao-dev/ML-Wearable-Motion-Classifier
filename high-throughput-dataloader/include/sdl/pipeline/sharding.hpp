#pragma once

#include <cstddef>
#include <vector>

namespace sdl {

// Splits a (typically shuffled) epoch sequence across `world_size` ranks by
// interleaved position — rank r takes positions r, r+world_size,
// r+2*world_size, ... — matching the PyTorch DistributedSampler
// convention. Every element of `epoch_sequence` belongs to exactly one
// rank's shard: the union over all ranks is the full sequence with no
// overlap and nothing dropped (ranks may differ in size by at most one
// element when the sequence length isn't evenly divisible by world_size).
std::vector<std::size_t> shard_indices(const std::vector<std::size_t>& epoch_sequence,
                                        std::size_t rank,
                                        std::size_t world_size);

} // namespace sdl
