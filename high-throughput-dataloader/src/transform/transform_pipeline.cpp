#include "sdl/transform/transform_pipeline.hpp"

#include <utility>

namespace sdl {

TransformPipeline::TransformPipeline(std::vector<std::shared_ptr<const Transform>> stages)
    : stages_(std::move(stages)) {}

Sample TransformPipeline::apply(Sample sample) const {
    for (const auto& stage : stages_) {
        sample = stage->apply(std::move(sample));
    }
    return sample;
}

} // namespace sdl
