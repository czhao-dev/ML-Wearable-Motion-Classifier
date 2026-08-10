#include "sdl/pipeline/reorder_buffer.hpp"

#include "sdl/generators/cost_knob.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

namespace sdl {
namespace {

using namespace std::chrono_literals;

TEST(ReorderBuffer, ReleasesItemsInAscendingOrderRegardlessOfInsertOrder) {
    ReorderBuffer<int> buf(4);
    ASSERT_TRUE(buf.insert(2, 200));
    ASSERT_TRUE(buf.insert(0, 0));
    ASSERT_TRUE(buf.insert(1, 100));

    EXPECT_EQ(buf.pop(), std::optional<int>(0));
    EXPECT_EQ(buf.pop(), std::optional<int>(100));
    EXPECT_EQ(buf.pop(), std::optional<int>(200));
}

TEST(ReorderBuffer, InsertBlocksWhenIndexIsOutsideWindow) {
    ReorderBuffer<int> buf(2); // window covers indices [0, 2)

    std::atomic<bool> inserted{false};
    std::thread inserter([&] {
        buf.insert(2, 222); // outside window until index 0 is popped
        inserted.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(50ms);
    EXPECT_FALSE(inserted.load(std::memory_order_acquire)) << "insert() returned before window advanced";

    ASSERT_TRUE(buf.insert(0, 0));
    EXPECT_EQ(buf.pop(), std::optional<int>(0)); // advances window to [1, 3)

    inserter.join();
    EXPECT_TRUE(inserted.load(std::memory_order_acquire));

    ASSERT_TRUE(buf.insert(1, 111));
    EXPECT_EQ(buf.pop(), std::optional<int>(111));
    EXPECT_EQ(buf.pop(), std::optional<int>(222));
}

TEST(ReorderBuffer, CloseDrainsBufferedItemsThenReturnsNullopt) {
    ReorderBuffer<int> buf(4);
    ASSERT_TRUE(buf.insert(1, 100));
    ASSERT_TRUE(buf.insert(0, 0));

    buf.close();

    EXPECT_EQ(buf.pop(), std::optional<int>(0));
    EXPECT_EQ(buf.pop(), std::optional<int>(100));
    EXPECT_EQ(buf.pop(), std::nullopt);

    EXPECT_FALSE(buf.insert(2, 200));
}

// NOTE on this test's shape: ReorderBuffer's bounded sliding window only
// guarantees liveness if indices are dispensed from a shared *monotonic*
// counter (as IndexStream does in production) — that keeps the "frontier"
// of in-flight (dispensed-but-not-yet-inserted) indices within a bounded
// distance of next_index_, since capacity >= num_threads. Indices handed
// out in arbitrary/random order (not from a shared monotonic counter) can
// deadlock the buffer: a thread can get stuck inserting an out-of-window
// index while the specific index the buffer is waiting for sits unattempted
// behind another blocked thread. So this test dispenses via a shared atomic
// counter, with random jitter before each insert to create genuine
// out-of-order completion, mirroring how WorkerPool actually drives it.
TEST(ReorderBuffer, StressConcurrentOutOfOrderCompletionYieldsStrictSequence) {
    constexpr int kNumItems = 10000;
    constexpr int kNumThreads = 4;
    constexpr std::size_t kCapacity = 2 * kNumThreads; // >= num_threads for liveness

    ReorderBuffer<int> buf(kCapacity);
    std::atomic<int> next_to_dispense{0};

    std::vector<std::thread> inserters;
    for (int t = 0; t < kNumThreads; ++t) {
        inserters.emplace_back([&, t] {
            std::mt19937_64 rng(42 + static_cast<std::uint64_t>(t));
            std::uniform_int_distribution<int> jitter(0, 20);
            while (true) {
                const int index = next_to_dispense.fetch_add(1, std::memory_order_relaxed);
                if (index >= kNumItems) {
                    break;
                }
                busy_spin(static_cast<std::uint64_t>(jitter(rng)) * 1000); // out-of-order jitter
                ASSERT_TRUE(buf.insert(static_cast<std::size_t>(index), index * 10));
            }
        });
    }

    std::vector<int> received;
    received.reserve(kNumItems);
    std::thread popper([&] {
        while (auto item = buf.pop()) {
            received.push_back(*item);
        }
    });

    for (auto& t : inserters) {
        t.join();
    }
    buf.close();
    popper.join();

    ASSERT_EQ(received.size(), static_cast<std::size_t>(kNumItems));
    for (int i = 0; i < kNumItems; ++i) {
        ASSERT_EQ(received[static_cast<std::size_t>(i)], i * 10) << "out of order at position " << i;
    }
}

} // namespace
} // namespace sdl
