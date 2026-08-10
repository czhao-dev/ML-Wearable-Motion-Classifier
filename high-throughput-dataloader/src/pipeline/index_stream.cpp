#include "sdl/pipeline/index_stream.hpp"

#include <utility>

namespace sdl {

IndexStream::IndexStream(std::vector<std::size_t> indices, std::size_t start_position)
    : indices_(std::move(indices)), cursor_(start_position) {}

std::optional<DispensedIndex> IndexStream::next() {
    const std::size_t pos = cursor_.fetch_add(1, std::memory_order_relaxed);
    if (pos >= indices_.size()) {
        return std::nullopt;
    }
    return DispensedIndex{pos, indices_[pos]};
}

} // namespace sdl
