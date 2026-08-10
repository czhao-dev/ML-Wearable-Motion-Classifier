#pragma once

#include "sdl/sample.hpp"

#include <cstdint>

namespace sdl {

// The synthetic data source. Implementations must be deterministic and
// stateless per call: identical (index, seed) always yields a bit-identical
// Sample, independent of call order, thread, or how many times sample() has
// been called before — this is what makes generation position-pure and
// worker-count invariant.
class Generator {
public:
    virtual ~Generator() = default;

    virtual Sample sample(std::size_t index, std::uint64_t seed) const = 0;
};

} // namespace sdl
