#include "sdl/generators/tabular_generator.hpp"
#include "sdl/generators/cost_knob.hpp"
#include "sdl/generators/seed_util.hpp"

#include <random>

namespace sdl {

TabularGenerator::TabularGenerator(Config config) : config_(config) {
    auto rng = make_rng(config_.function_seed, 0);
    std::normal_distribution<float> weight_dist(0.0f, 1.0f);
    weights_.resize(static_cast<std::size_t>(config_.num_features));
    for (auto& w : weights_) {
        w = weight_dist(rng);
    }
    bias_ = weight_dist(rng);
}

Sample TabularGenerator::sample(std::size_t index, std::uint64_t seed) const {
    auto rng = make_rng(seed, index);
    std::normal_distribution<float> latent_dist(0.0f, 1.0f);
    std::normal_distribution<float> noise_dist(0.0f, config_.noise_std);

    std::vector<float> observed(static_cast<std::size_t>(config_.num_features));
    float target = bias_;
    for (std::size_t i = 0; i < observed.size(); ++i) {
        const float latent = latent_dist(rng);
        observed[i] = latent + noise_dist(rng);
        target += weights_[i] * latent;
    }

    Sample s;
    s.index = index;
    s.data = std::move(observed);
    s.data_shape = {config_.num_features};
    s.label = {target};
    s.label_shape = {1};

    busy_spin(config_.cost_iterations);

    return s;
}

} // namespace sdl
