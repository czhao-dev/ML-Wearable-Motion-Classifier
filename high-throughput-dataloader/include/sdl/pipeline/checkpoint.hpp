#pragma once

#include <cstddef>

namespace sdl {

// Resumable iteration state. `consumer_position` is the count of samples
// the CONSUMER has fully processed within this epoch's shard for this
// rank — not the producer's index-assignment cursor. On restore, any
// samples already generated/queued-but-unconsumed beyond this position are
// discarded rather than replayed: production simply restarts at
// `consumer_position`, and because generation is a pure function of
// index, nothing needs to be drained from a live queue at checkpoint time
// to guarantee pre ++ post == the uninterrupted sequence.
//
// There is no separate "shuffle RNG state" field here: epoch_permutation
// is a pure function of (seed, epoch, num_samples), so the full epoch
// order can always be regenerated exactly from the Loader's static config
// plus this checkpoint's `epoch`.
struct Checkpoint {
    std::size_t epoch = 0;
    std::size_t consumer_position = 0;
};

} // namespace sdl
