#include "sdl/queue/bounded_queue.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <optional>
#include <thread>
#include <vector>

namespace sdl {
namespace {

using namespace std::chrono_literals;

TEST(BoundedQueue, PushPopPreservesFifoOrder) {
    BoundedQueue<int> q(4);
    ASSERT_TRUE(q.push(1));
    ASSERT_TRUE(q.push(2));
    ASSERT_TRUE(q.push(3));

    EXPECT_EQ(q.pop(), std::optional<int>(1));
    EXPECT_EQ(q.pop(), std::optional<int>(2));
    EXPECT_EQ(q.pop(), std::optional<int>(3));
}

TEST(BoundedQueue, RespectsCapacity) {
    BoundedQueue<int> q(2);
    ASSERT_TRUE(q.push(1));
    ASSERT_TRUE(q.push(2));
    EXPECT_EQ(q.size(), 2u);
    EXPECT_EQ(q.capacity(), 2u);
}

TEST(BoundedQueue, ProducerBlocksWhenFull) {
    BoundedQueue<int> q(2);
    ASSERT_TRUE(q.push(1));
    ASSERT_TRUE(q.push(2));

    std::atomic<bool> pushed{false};
    std::thread producer([&] {
        q.push(3); // should block: queue is at capacity
        pushed.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(50ms);
    EXPECT_FALSE(pushed.load(std::memory_order_acquire)) << "push() returned while queue was still full";

    ASSERT_EQ(q.pop(), std::optional<int>(1)); // frees one slot

    producer.join();
    EXPECT_TRUE(pushed.load(std::memory_order_acquire));
    EXPECT_EQ(q.pop(), std::optional<int>(2));
    EXPECT_EQ(q.pop(), std::optional<int>(3));
}

TEST(BoundedQueue, ConsumerBlocksWhenEmpty) {
    BoundedQueue<int> q(2);

    std::atomic<bool> popped{false};
    std::optional<int> result;
    std::thread consumer([&] {
        result = q.pop(); // should block: queue is empty
        popped.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(50ms);
    EXPECT_FALSE(popped.load(std::memory_order_acquire)) << "pop() returned while queue was still empty";

    ASSERT_TRUE(q.push(42));

    consumer.join();
    EXPECT_TRUE(popped.load(std::memory_order_acquire));
    EXPECT_EQ(result, std::optional<int>(42));
}

TEST(BoundedQueue, CloseDrainsRemainingItemsThenReturnsNullopt) {
    BoundedQueue<int> q(4);
    ASSERT_TRUE(q.push(1));
    ASSERT_TRUE(q.push(2));

    q.close();

    // Remaining items are still delivered after close().
    EXPECT_EQ(q.pop(), std::optional<int>(1));
    EXPECT_EQ(q.pop(), std::optional<int>(2));
    // Drained and closed: pop() no longer blocks, returns nullopt.
    EXPECT_EQ(q.pop(), std::nullopt);

    // push() after close() fails without blocking, even though there's room.
    EXPECT_FALSE(q.push(3));
}

TEST(BoundedQueue, CloseWakesBlockedProducerWithoutEnqueuing) {
    BoundedQueue<int> q(1);
    ASSERT_TRUE(q.push(1)); // fill to capacity

    std::atomic<bool> push_returned{false};
    bool push_result = true;
    std::thread producer([&] {
        push_result = q.push(2); // blocks: queue full
        push_returned.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(50ms);
    EXPECT_FALSE(push_returned.load(std::memory_order_acquire));

    q.close();
    producer.join();

    EXPECT_TRUE(push_returned.load(std::memory_order_acquire));
    EXPECT_FALSE(push_result) << "blocked push() should fail once queue is closed, not enqueue";
    EXPECT_EQ(q.pop(), std::optional<int>(1));
    EXPECT_EQ(q.pop(), std::nullopt);
}

TEST(BoundedQueue, CloseWakesBlockedConsumerOnceDrained) {
    BoundedQueue<int> q(4);

    std::atomic<bool> pop_returned{false};
    std::optional<int> result = 999;
    std::thread consumer([&] {
        result = q.pop(); // blocks: queue empty
        pop_returned.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(50ms);
    EXPECT_FALSE(pop_returned.load(std::memory_order_acquire));

    q.close();
    consumer.join();

    EXPECT_TRUE(pop_returned.load(std::memory_order_acquire));
    EXPECT_EQ(result, std::nullopt);
}

// Many producers, many consumers, no lost or duplicated items, and no
// deadlock/lost-wakeup under contention with a small capacity.
TEST(BoundedQueue, StressManyProducersManyConsumersExactlyOnce) {
    constexpr int kNumProducers = 4;
    constexpr int kItemsPerProducer = 5000;
    constexpr int kNumConsumers = 3;
    constexpr std::size_t kCapacity = 16;

    BoundedQueue<int> q(kCapacity);

    std::vector<std::thread> producers;
    for (int p = 0; p < kNumProducers; ++p) {
        producers.emplace_back([&, p] {
            for (int i = 0; i < kItemsPerProducer; ++i) {
                ASSERT_TRUE(q.push(p * kItemsPerProducer + i));
            }
        });
    }

    std::vector<std::thread> consumers;
    std::vector<std::vector<int>> received(static_cast<std::size_t>(kNumConsumers));
    for (int c = 0; c < kNumConsumers; ++c) {
        consumers.emplace_back([&, c] {
            while (auto item = q.pop()) {
                received[static_cast<std::size_t>(c)].push_back(*item);
            }
        });
    }

    for (auto& t : producers) {
        t.join();
    }
    q.close();
    for (auto& t : consumers) {
        t.join();
    }

    std::size_t total_received = 0;
    for (const auto& v : received) {
        total_received += v.size();
    }
    EXPECT_EQ(total_received, static_cast<std::size_t>(kNumProducers * kItemsPerProducer));

    std::vector<int> all;
    all.reserve(total_received);
    for (const auto& v : received) {
        all.insert(all.end(), v.begin(), v.end());
    }
    std::sort(all.begin(), all.end());
    for (int expected = 0; expected < kNumProducers * kItemsPerProducer; ++expected) {
        ASSERT_EQ(all[static_cast<std::size_t>(expected)], expected) << "item " << expected << " lost or duplicated";
    }
}

} // namespace
} // namespace sdl
