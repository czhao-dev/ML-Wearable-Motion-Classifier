#include "sdl/pipeline/index_stream.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

namespace sdl {
namespace {

std::vector<std::size_t> identity(std::size_t n) {
    std::vector<std::size_t> v(n);
    std::iota(v.begin(), v.end(), std::size_t{0});
    return v;
}

TEST(IndexStream, SingleThreadYieldsSequentialIndicesThenNullopt) {
    IndexStream stream(identity(3));
    EXPECT_EQ(stream.next(), (std::optional<DispensedIndex>({0, 0})));
    EXPECT_EQ(stream.next(), (std::optional<DispensedIndex>({1, 1})));
    EXPECT_EQ(stream.next(), (std::optional<DispensedIndex>({2, 2})));
    EXPECT_EQ(stream.next(), std::nullopt);
    EXPECT_EQ(stream.next(), std::nullopt);
}

TEST(IndexStream, DispensesTheGivenIndicesNotJustPosition) {
    IndexStream stream({10, 20, 30});
    ASSERT_EQ(stream.next(), (std::optional<DispensedIndex>({0, 10})));
    ASSERT_EQ(stream.next(), (std::optional<DispensedIndex>({1, 20})));
    ASSERT_EQ(stream.next(), (std::optional<DispensedIndex>({2, 30})));
}

TEST(IndexStream, StartPositionResumesWithoutRedispensingEarlierPositions) {
    IndexStream stream({10, 20, 30, 40, 50}, /*start_position=*/2);
    ASSERT_EQ(stream.next(), (std::optional<DispensedIndex>({2, 30})));
    ASSERT_EQ(stream.next(), (std::optional<DispensedIndex>({3, 40})));
    ASSERT_EQ(stream.next(), (std::optional<DispensedIndex>({4, 50})));
    EXPECT_EQ(stream.next(), std::nullopt);
}

TEST(IndexStream, ConcurrentWorkersGetEachIndexExactlyOnce) {
    constexpr std::size_t kNumSamples = 20000;
    constexpr int kNumWorkers = 8;

    IndexStream stream(identity(kNumSamples));
    std::mutex mutex;
    std::vector<std::size_t> received;
    received.reserve(kNumSamples);

    std::vector<std::thread> workers;
    for (int w = 0; w < kNumWorkers; ++w) {
        workers.emplace_back([&] {
            std::vector<std::size_t> local;
            while (auto dispensed = stream.next()) {
                local.push_back(dispensed->index);
            }
            std::lock_guard<std::mutex> lock(mutex);
            received.insert(received.end(), local.begin(), local.end());
        });
    }
    for (auto& t : workers) {
        t.join();
    }

    ASSERT_EQ(received.size(), kNumSamples);
    std::sort(received.begin(), received.end());
    for (std::size_t i = 0; i < kNumSamples; ++i) {
        ASSERT_EQ(received[i], i) << "index " << i << " lost or duplicated";
    }
}

} // namespace
} // namespace sdl
