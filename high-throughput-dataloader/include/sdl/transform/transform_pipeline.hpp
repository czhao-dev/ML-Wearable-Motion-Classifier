#pragma once

#include "sdl/sample.hpp"
#include "sdl/transform/transform.hpp"

#include <memory>
#include <vector>

namespace sdl {

// Applies a sequence of Transform stages to a Sample, in order. An empty
// pipeline is a no-op passthrough. Stages are shared (not owned uniquely)
// so the same pipeline can be handed to multiple worker threads.
class TransformPipeline {
public:
    TransformPipeline() = default;
    explicit TransformPipeline(std::vector<std::shared_ptr<const Transform>> stages);

    Sample apply(Sample sample) const;

private:
    std::vector<std::shared_ptr<const Transform>> stages_;
};

} // namespace sdl
