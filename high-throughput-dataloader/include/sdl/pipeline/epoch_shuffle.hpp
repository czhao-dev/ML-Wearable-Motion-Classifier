#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sdl {

// Deterministic seeded Fisher-Yates permutation of [0, num_samples),
// precomputed in full rather than streamed through a windowed buffer. This
// is what makes resumability exact: the entire epoch order is a pure
// function of (seed, epoch, num_samples), so a checkpoint only ever needs
// to record a position — never any shuffle-buffer contents — and the full
// order can always be regenerated on restore. Worker-count invariant: the
// result does not depend on how many workers will later consume it.
std::vector<std::size_t> epoch_permutation(std::uint64_t seed, std::size_t epoch, std::size_t num_samples);

// The unshuffled identity sequence [0, num_samples) — for callers that
// don't need shuffling (e.g. P1 usage, or shuffling disabled).
std::vector<std::size_t> identity_sequence(std::size_t num_samples);

} // namespace sdl
