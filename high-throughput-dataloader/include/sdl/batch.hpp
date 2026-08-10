#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sdl {

// A collated set of samples: contiguous [B, ...] buffers ready for the
// consumer. Populated by the collate stage (src/collate/), not by generators.
struct Batch {
    std::vector<std::size_t> indices;
    std::vector<float> data;
    std::vector<std::int64_t> data_shape;  // per-sample shape, excludes leading B
    std::vector<float> label;
    std::vector<std::int64_t> label_shape; // per-sample shape, excludes leading B

    std::size_t batch_size() const { return indices.size(); }
};

} // namespace sdl
