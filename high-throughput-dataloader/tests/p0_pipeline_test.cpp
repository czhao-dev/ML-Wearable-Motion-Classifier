#include "sdl/generators/image_generator.hpp"
#include "sdl/pipeline/mock_consumer.hpp"
#include "sdl/pipeline/producer.hpp"
#include "sdl/queue/bounded_queue.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <vector>

namespace sdl {
namespace {

TEST(P0Pipeline, AllSamplesDeliveredExactlyOnceInOrder) {
    ImageGenerator gen(ImageGenerator::Config{});
    BoundedQueue<Sample> queue(8);
    constexpr std::size_t num_samples = 500;

    std::thread producer([&] { run_single_producer(gen, num_samples, /*seed=*/7, queue); });

    std::vector<std::size_t> received_indices;
    while (auto sample = queue.pop()) {
        received_indices.push_back(sample->index);
    }
    producer.join();

    ASSERT_EQ(received_indices.size(), num_samples);
    for (std::size_t i = 0; i < num_samples; ++i) {
        EXPECT_EQ(received_indices[i], i);
    }
}

TEST(P0Pipeline, MockConsumerReportsAllSamplesAndPositiveWallTime) {
    ImageGenerator gen(ImageGenerator::Config{});
    BoundedQueue<Sample> queue(8);
    constexpr std::size_t num_samples = 50;

    std::thread producer([&] { run_single_producer(gen, num_samples, /*seed=*/7, queue); });
    const ConsumeResult result = run_mock_consumer(queue, std::chrono::microseconds(200));
    producer.join();

    EXPECT_EQ(result.samples_consumed, num_samples);
    EXPECT_GT(result.wall_time.count(), 0);
}

} // namespace
} // namespace sdl
