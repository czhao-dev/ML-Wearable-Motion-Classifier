#pragma once

#include "sdl/generators/generator.hpp"

#include <cstdint>
#include <vector>

namespace sdl {

// Regression generator: a latent feature vector is sampled per index, the
// observed features are the latent vector plus noise, and the label is the
// *exact* target computed from the clean latent via a fixed linear function
// (weights/bias) chosen at construction from `function_seed` — independent
// of per-sample index. This lets a consumer verify predictions against a
// known ground-truth function even though the observed features are noisy.
class TabularGenerator : public Generator {
public:
    struct Config {
        int num_features = 8;
        std::uint64_t function_seed = 0;
        float noise_std = 0.1f;
        std::uint64_t cost_iterations = 0;
    };

    explicit TabularGenerator(Config config);

    Sample sample(std::size_t index, std::uint64_t seed) const override;

    const std::vector<float>& weights() const { return weights_; }
    float bias() const { return bias_; }

private:
    Config config_;
    std::vector<float> weights_;
    float bias_ = 0.0f;
};

} // namespace sdl
