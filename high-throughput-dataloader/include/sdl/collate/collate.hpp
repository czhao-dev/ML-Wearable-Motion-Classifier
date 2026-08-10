#pragma once

#include "sdl/batch.hpp"
#include "sdl/sample.hpp"

#include <vector>

namespace sdl {

// Collates a set of same-shaped Samples into a single contiguous Batch,
// preserving input order. Precondition: every sample shares the same
// data_shape and label_shape (mixed-shape batches are not supported).
// Returns an empty Batch for an empty input.
Batch collate(std::vector<Sample> samples);

} // namespace sdl
