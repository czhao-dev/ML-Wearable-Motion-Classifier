#include "sdl/generators/image_generator.hpp"
#include "sdl/generators/tabular_generator.hpp"
#include "sdl/generators/token_generator.hpp"

#include <gtest/gtest.h>

namespace sdl {
namespace {

// ---- ImageGenerator ----

TEST(ImageGenerator, DeterministicForSameIndexAndSeed) {
    ImageGenerator gen(ImageGenerator::Config{});
    const Sample a = gen.sample(/*index=*/7, /*seed=*/42);
    const Sample b = gen.sample(/*index=*/7, /*seed=*/42);
    EXPECT_EQ(a, b);
}

TEST(ImageGenerator, DistinctIndicesProduceDistinctSamples) {
    ImageGenerator gen(ImageGenerator::Config{});
    const Sample a = gen.sample(0, 42);
    for (std::size_t index = 1; index < 20; ++index) {
        EXPECT_NE(gen.sample(index, 42), a) << "index=" << index;
    }
}

TEST(ImageGenerator, CostKnobScalesWorkNotOutput) {
    ImageGenerator::Config cheap;
    cheap.cost_iterations = 0;
    ImageGenerator::Config expensive;
    expensive.cost_iterations = 200000;

    ImageGenerator cheap_gen(cheap);
    ImageGenerator expensive_gen(expensive);

    EXPECT_EQ(cheap_gen.sample(3, 42), expensive_gen.sample(3, 42));
}

// ---- TokenGenerator ----

TEST(TokenGenerator, DeterministicForSameIndexAndSeed) {
    TokenGenerator gen(TokenGenerator::Config{});
    const Sample a = gen.sample(7, 42);
    const Sample b = gen.sample(7, 42);
    EXPECT_EQ(a, b);
}

TEST(TokenGenerator, DistinctIndicesProduceDistinctSamples) {
    TokenGenerator gen(TokenGenerator::Config{});
    const Sample a = gen.sample(0, 42);
    for (std::size_t index = 1; index < 20; ++index) {
        EXPECT_NE(gen.sample(index, 42), a) << "index=" << index;
    }
}

TEST(TokenGenerator, LabelIsNextTokenShiftedInput) {
    TokenGenerator::Config config;
    config.seq_len = 8;
    TokenGenerator gen(config);
    const Sample s = gen.sample(11, 99);
    ASSERT_EQ(s.data.size(), s.label.size());
    for (std::size_t t = 0; t + 1 < s.data.size(); ++t) {
        EXPECT_FLOAT_EQ(s.label[t], s.data[t + 1]);
    }
}

TEST(TokenGenerator, CostKnobScalesWorkNotOutput) {
    TokenGenerator::Config cheap;
    cheap.cost_iterations = 0;
    TokenGenerator::Config expensive;
    expensive.cost_iterations = 200000;

    TokenGenerator cheap_gen(cheap);
    TokenGenerator expensive_gen(expensive);

    EXPECT_EQ(cheap_gen.sample(3, 42), expensive_gen.sample(3, 42));
}

// ---- TabularGenerator ----

TEST(TabularGenerator, DeterministicForSameIndexAndSeed) {
    TabularGenerator gen(TabularGenerator::Config{});
    const Sample a = gen.sample(7, 42);
    const Sample b = gen.sample(7, 42);
    EXPECT_EQ(a, b);
}

TEST(TabularGenerator, DistinctIndicesProduceDistinctSamples) {
    TabularGenerator gen(TabularGenerator::Config{});
    const Sample a = gen.sample(0, 42);
    for (std::size_t index = 1; index < 20; ++index) {
        EXPECT_NE(gen.sample(index, 42), a) << "index=" << index;
    }
}

TEST(TabularGenerator, LabelIsExactKnownFunctionOfCleanLatent) {
    TabularGenerator::Config config;
    config.noise_std = 0.0f; // observed features == clean latent
    TabularGenerator gen(config);

    const Sample s = gen.sample(5, 123);
    ASSERT_EQ(s.data.size(), gen.weights().size());

    float expected = gen.bias();
    for (std::size_t i = 0; i < s.data.size(); ++i) {
        expected += gen.weights()[i] * s.data[i];
    }
    ASSERT_EQ(s.label.size(), 1u);
    EXPECT_NEAR(s.label[0], expected, 1e-4f);
}

TEST(TabularGenerator, CostKnobScalesWorkNotOutput) {
    TabularGenerator::Config cheap;
    cheap.cost_iterations = 0;
    TabularGenerator::Config expensive;
    expensive.cost_iterations = 200000;

    TabularGenerator cheap_gen(cheap);
    TabularGenerator expensive_gen(expensive);

    EXPECT_EQ(cheap_gen.sample(3, 42), expensive_gen.sample(3, 42));
}

} // namespace
} // namespace sdl
