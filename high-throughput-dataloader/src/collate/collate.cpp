#include "sdl/collate/collate.hpp"

#include <cassert>
#include <iterator>

namespace sdl {

Batch collate(std::vector<Sample> samples) {
    Batch batch;
    if (samples.empty()) {
        return batch;
    }

    const auto data_shape = samples.front().data_shape;
    const auto label_shape = samples.front().label_shape;

    batch.data_shape = data_shape;
    batch.label_shape = label_shape;
    batch.indices.reserve(samples.size());
    batch.data.reserve(samples.size() * samples.front().data.size());
    batch.label.reserve(samples.size() * samples.front().label.size());

    for (auto& sample : samples) {
        assert(sample.data_shape == data_shape);
        assert(sample.label_shape == label_shape);
        batch.indices.push_back(sample.index);
        batch.data.insert(batch.data.end(),
                           std::make_move_iterator(sample.data.begin()),
                           std::make_move_iterator(sample.data.end()));
        batch.label.insert(batch.label.end(),
                            std::make_move_iterator(sample.label.begin()),
                            std::make_move_iterator(sample.label.end()));
    }

    return batch;
}

} // namespace sdl
