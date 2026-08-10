#include "sdl/generators/token_generator.hpp"
#include "sdl/generators/cost_knob.hpp"
#include "sdl/generators/seed_util.hpp"

#include <cmath>
#include <random>

namespace sdl {

TokenGenerator::TokenGenerator(Config config) : config_(config) {
    auto rng = make_rng(config_.transition_seed, 0);
    std::uniform_real_distribution<float> bias_dist(-1.0f, 1.0f);
    transition_bias_.resize(static_cast<std::size_t>(config_.vocab_size));
    for (auto& b : transition_bias_) {
        b = bias_dist(rng);
    }
}

Sample TokenGenerator::sample(std::size_t index, std::uint64_t seed) const {
    auto rng = make_rng(seed, index);
    std::uniform_int_distribution<int> token_dist(0, config_.vocab_size - 1);
    std::normal_distribution<float> step_noise(0.0f, 1.0f);

    std::vector<int> tokens(static_cast<std::size_t>(config_.seq_len) + 1);
    tokens[0] = token_dist(rng);
    for (int t = 1; t <= config_.seq_len; ++t) {
        const float drift = transition_bias_[static_cast<std::size_t>(tokens[static_cast<std::size_t>(t - 1)])] +
                             step_noise(rng);
        int next = (tokens[static_cast<std::size_t>(t - 1)] + 1 + static_cast<int>(std::lround(drift))) %
                   config_.vocab_size;
        if (next < 0) {
            next += config_.vocab_size;
        }
        tokens[static_cast<std::size_t>(t)] = next;
    }

    Sample s;
    s.index = index;
    s.data.resize(static_cast<std::size_t>(config_.seq_len));
    s.label.resize(static_cast<std::size_t>(config_.seq_len));
    for (int t = 0; t < config_.seq_len; ++t) {
        s.data[static_cast<std::size_t>(t)] = static_cast<float>(tokens[static_cast<std::size_t>(t)]);
        s.label[static_cast<std::size_t>(t)] = static_cast<float>(tokens[static_cast<std::size_t>(t) + 1]);
    }
    s.data_shape = {config_.seq_len};
    s.label_shape = {config_.seq_len};

    busy_spin(config_.cost_iterations);

    return s;
}

} // namespace sdl
