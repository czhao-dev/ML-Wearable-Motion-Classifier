#include "sdl/generators/image_generator.hpp"
#include "sdl/generators/cost_knob.hpp"
#include "sdl/generators/seed_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>

namespace sdl {

ImageGenerator::ImageGenerator(Config config) : config_(config) {}

Sample ImageGenerator::sample(std::size_t index, std::uint64_t seed) const {
    auto rng = make_rng(seed, index);

    std::uniform_int_distribution<int> class_dist(0, config_.num_classes - 1);
    const int label = class_dist(rng);

    std::uniform_int_distribution<int> shape_count_dist(1, std::max(1, config_.max_shapes));
    const int num_shapes = shape_count_dist(rng);

    const int c = config_.channels;
    const int h = config_.height;
    const int w = config_.width;

    Sample s;
    s.index = index;
    s.data.assign(static_cast<std::size_t>(c) * static_cast<std::size_t>(h) * static_cast<std::size_t>(w), 0.0f);
    s.data_shape = {c, h, w};
    s.label = {static_cast<float>(label)};
    s.label_shape = {1};

    std::uniform_int_distribution<int> x_dist(0, w - 1);
    std::uniform_int_distribution<int> y_dist(0, h - 1);
    std::uniform_int_distribution<int> size_dist(1, std::max(1, std::min(h, w) / 2));
    std::uniform_real_distribution<float> intensity_dist(0.25f, 1.0f);
    std::uniform_int_distribution<int> shape_type_dist(0, 1); // 0 = rect, 1 = circle

    for (int shape_i = 0; shape_i < num_shapes; ++shape_i) {
        const int cx = x_dist(rng);
        const int cy = y_dist(rng);
        const int r = size_dist(rng);
        const float intensity = intensity_dist(rng);
        const int shape_type = shape_type_dist(rng);

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                bool inside;
                if (shape_type == 0) {
                    inside = std::abs(x - cx) <= r && std::abs(y - cy) <= r;
                } else {
                    const int dx = x - cx;
                    const int dy = y - cy;
                    inside = (dx * dx + dy * dy) <= r * r;
                }
                if (inside) {
                    for (int ch = 0; ch < c; ++ch) {
                        const std::size_t offset =
                            (static_cast<std::size_t>(ch) * static_cast<std::size_t>(h) + static_cast<std::size_t>(y)) *
                                static_cast<std::size_t>(w) +
                            static_cast<std::size_t>(x);
                        s.data[offset] = intensity;
                    }
                }
            }
        }
    }

    busy_spin(config_.cost_iterations);

    return s;
}

} // namespace sdl
