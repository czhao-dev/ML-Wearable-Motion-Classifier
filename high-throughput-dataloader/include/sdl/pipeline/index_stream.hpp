#pragma once

#include <atomic>
#include <cstddef>
#include <optional>
#include <vector>

namespace sdl {

// A single dispensed unit. `position` is this call's rank in the shared
// dispensing order (dense, ascending: start_position, start_position+1,
// ...) and is what ordered mode must key its reordering on. `index` is the
// actual sample index to generate, which may be shuffled and is therefore
// NOT generally equal to `position`. The two are returned together from a
// single atomic step because splitting them into separate calls would race
// under concurrent dispensing.
struct DispensedIndex {
    std::size_t position;
    std::size_t index;

    bool operator==(const DispensedIndex& other) const = default;
};

// A shared, thread-safe dispenser over a precomputed ordered index
// sequence, pulled concurrently by worker threads. `start_position`
// supports resuming mid-epoch: positions before it are never redispensed.
class IndexStream {
public:
    explicit IndexStream(std::vector<std::size_t> indices, std::size_t start_position = 0);

    IndexStream(const IndexStream&) = delete;
    IndexStream& operator=(const IndexStream&) = delete;

    // Returns the next (position, index) pair, or std::nullopt once the
    // sequence is exhausted. Safe to call concurrently from multiple
    // worker threads; each position/index is returned to exactly one
    // caller.
    std::optional<DispensedIndex> next();

    std::size_t size() const { return indices_.size(); }

private:
    std::vector<std::size_t> indices_;
    std::atomic<std::size_t> cursor_;
};

} // namespace sdl
