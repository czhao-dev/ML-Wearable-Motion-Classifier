#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sdl {

// A single owning training sample: a flattened data buffer with shape, a
// flattened label/target buffer with shape, and the epoch index it was
// generated from.
struct Sample {
    std::size_t index = 0;
    std::vector<float> data;
    std::vector<std::int64_t> data_shape;
    std::vector<float> label;
    std::vector<std::int64_t> label_shape;

    bool operator==(const Sample& other) const = default;
};

} // namespace sdl
