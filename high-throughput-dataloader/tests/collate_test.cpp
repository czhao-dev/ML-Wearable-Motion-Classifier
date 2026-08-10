#include "sdl/collate/collate.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace sdl {
namespace {

Sample make_sample(std::size_t index, std::vector<float> data, float label) {
    Sample s;
    s.index = index;
    s.data = std::move(data);
    s.data_shape = {static_cast<std::int64_t>(s.data.size())};
    s.label = {label};
    s.label_shape = {1};
    return s;
}

TEST(Collate, EmptyInputYieldsEmptyBatch) {
    const Batch batch = collate({});
    EXPECT_EQ(batch.batch_size(), 0u);
    EXPECT_TRUE(batch.data.empty());
}

TEST(Collate, ProducesContiguousBufferPreservingOrder) {
    std::vector<Sample> samples;
    samples.push_back(make_sample(10, {1.0f, 2.0f}, 100.0f));
    samples.push_back(make_sample(11, {3.0f, 4.0f}, 101.0f));
    samples.push_back(make_sample(12, {5.0f, 6.0f}, 102.0f));

    const Batch batch = collate(std::move(samples));

    EXPECT_EQ(batch.batch_size(), 3u);
    EXPECT_EQ(batch.indices, (std::vector<std::size_t>{10, 11, 12}));
    EXPECT_EQ(batch.data, (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
    EXPECT_EQ(batch.label, (std::vector<float>{100.0f, 101.0f, 102.0f}));
    EXPECT_EQ(batch.data_shape, (std::vector<std::int64_t>{2}));
    EXPECT_EQ(batch.label_shape, (std::vector<std::int64_t>{1}));
}

} // namespace
} // namespace sdl
