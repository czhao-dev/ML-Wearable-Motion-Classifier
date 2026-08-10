#include "sdl/transform/scale_transform.hpp"
#include "sdl/transform/transform_pipeline.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

namespace sdl {
namespace {

Sample make_sample(std::vector<float> data) {
    Sample s;
    s.index = 0;
    s.data = std::move(data);
    s.data_shape = {static_cast<std::int64_t>(s.data.size())};
    s.label = {0.0f};
    s.label_shape = {1};
    return s;
}

TEST(TransformPipeline, EmptyPipelineIsPassthrough) {
    TransformPipeline pipeline;
    const Sample in = make_sample({1.0f, 2.0f, 3.0f});
    const Sample out = pipeline.apply(in);
    EXPECT_EQ(out, in);
}

TEST(TransformPipeline, SingleStageApplies) {
    TransformPipeline pipeline({std::make_shared<ScaleTransform>(2.0f)});
    const Sample out = pipeline.apply(make_sample({1.0f, 2.0f, 3.0f}));
    EXPECT_FLOAT_EQ(out.data[0], 2.0f);
    EXPECT_FLOAT_EQ(out.data[1], 4.0f);
    EXPECT_FLOAT_EQ(out.data[2], 6.0f);
}

TEST(TransformPipeline, MultipleStagesComposeInOrder) {
    TransformPipeline pipeline({
        std::make_shared<ScaleTransform>(2.0f),
        std::make_shared<ScaleTransform>(3.0f),
    });
    const Sample out = pipeline.apply(make_sample({1.0f}));
    EXPECT_FLOAT_EQ(out.data[0], 6.0f); // 1 * 2 * 3
}

TEST(TransformPipeline, StatelessTransformIsSafeAcrossConcurrentCalls) {
    TransformPipeline pipeline({std::make_shared<ScaleTransform>(2.0f)});

    constexpr int kNumThreads = 8;
    std::vector<std::thread> threads;
    std::vector<Sample> results(kNumThreads);
    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&, t] {
            results[static_cast<std::size_t>(t)] =
                pipeline.apply(make_sample({static_cast<float>(t)}));
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    for (int t = 0; t < kNumThreads; ++t) {
        EXPECT_FLOAT_EQ(results[static_cast<std::size_t>(t)].data[0], static_cast<float>(t) * 2.0f);
    }
}

} // namespace
} // namespace sdl
