#pragma once

#include "sdl/generators/generator.hpp"

#include <cstdint>

namespace sdl {

// Procedural-shapes image generator: renders a random number of filled
// rectangles/circles onto a [C,H,W] canvas and assigns a class label.
// Deterministic in (index, seed).
class ImageGenerator : public Generator {
public:
    struct Config {
        int channels = 1;
        int height = 32;
        int width = 32;
        int num_classes = 10;
        int max_shapes = 3;
        std::uint64_t cost_iterations = 0; // busy-loop cost knob
    };

    explicit ImageGenerator(Config config);

    Sample sample(std::size_t index, std::uint64_t seed) const override;

private:
    Config config_;
};

} // namespace sdl
