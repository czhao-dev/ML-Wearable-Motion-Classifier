#include "sdl/pipeline/mock_batch_consumer.hpp"

#include "sdl/generators/image_generator.hpp"
#include "sdl/pipeline/producer.hpp"
#include "sdl/queue/bounded_queue.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

namespace sdl {
namespace {

TEST(MockBatchConsumer, CollatesFullBatchesAndFinalRaggedBatch) {
    ImageGenerator gen(ImageGenerator::Config{});
    BoundedQueue<Sample> queue(8);
    constexpr std::size_t num_samples = 25; // not a multiple of batch_size
    constexpr std::size_t batch_size = 10;

    std::thread producer([&] { run_single_producer(gen, num_samples, /*seed=*/3, queue); });
    const BatchConsumeResult result = run_mock_batch_consumer(queue, batch_size, std::chrono::microseconds(0));
    producer.join();

    EXPECT_EQ(result.samples_consumed, num_samples);
    EXPECT_EQ(result.batches_consumed, 3u); // 10 + 10 + 5 (ragged)
}

} // namespace
} // namespace sdl
