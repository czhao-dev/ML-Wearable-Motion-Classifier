#pragma once

#include "sdl/sample.hpp"

namespace sdl {

// A composable augment/decode stage. Implementations must be stateless (no
// mutable state) so a single instance can be safely shared and called
// concurrently across worker threads.
class Transform {
public:
    virtual ~Transform() = default;
    virtual Sample apply(Sample sample) const = 0;
};

} // namespace sdl
