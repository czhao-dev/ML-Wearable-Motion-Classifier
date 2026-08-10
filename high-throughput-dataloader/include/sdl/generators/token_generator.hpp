#pragma once

#include "sdl/generators/generator.hpp"

#include <cstdint>
#include <vector>

namespace sdl {

// Markov-chain token-sequence generator for a next-token prediction task.
// Deterministic in (index, seed); the chain's drift bias is fixed at
// construction from `transition_seed`, independent of per-sample index, so
// all samples are drawn from the same fixed grammar.
class TokenGenerator : public Generator {
public:
    struct Config {
        int vocab_size = 64;
        int seq_len = 16;
        std::uint64_t transition_seed = 0;
        std::uint64_t cost_iterations = 0;
    };

    explicit TokenGenerator(Config config);

    Sample sample(std::size_t index, std::uint64_t seed) const override;

private:
    Config config_;
    std::vector<float> transition_bias_; // per-token additive drift, fixes the chain
};

} // namespace sdl
