#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace sdl {

// Buffers out-of-order items tagged by an ascending index and releases them
// in strict order starting from index 0. Implemented as a bounded sliding
// window [next_index_, next_index_ + capacity): insert() blocks while its
// index falls outside that window (i.e. there isn't room yet because the
// window hasn't advanced far enough), and pop() blocks until the item at
// next_index_ has arrived.
//
// This bounded capacity is what makes ordered-mode throughput bounded by the
// slowest in-flight producer of the next needed index: once `capacity`
// items are buffered ahead of a stalled slot, every other producer blocks
// in insert() too, rather than letting memory grow unbounded.
//
// close() means "no more inserts will ever happen" — pop() keeps draining
// whatever was already buffered in order, and only returns std::nullopt
// once it reaches a slot that was never (and will never be) filled.
template <typename T>
class ReorderBuffer {
public:
    // `start_index` aligns the window with wherever dispensing resumes from
    // (e.g. IndexStream's start_position on restore) — the first item this
    // buffer will release is `start_index`, not always 0.
    explicit ReorderBuffer(std::size_t capacity, std::size_t start_index = 0)
        : capacity_(capacity), next_index_(start_index), slots_(capacity), filled_(capacity, 0) {}

    ReorderBuffer(const ReorderBuffer&) = delete;
    ReorderBuffer& operator=(const ReorderBuffer&) = delete;

    // Blocks while `index` is outside the current window and the buffer is
    // open. Returns false (item not stored) if closed before/while waiting.
    bool insert(std::size_t index, T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        room_.wait(lock, [&] { return index < next_index_ + capacity_ || closed_; });
        if (closed_) {
            return false;
        }
        const std::size_t slot = index % capacity_;
        slots_[slot] = std::move(item);
        filled_[slot] = 1;
        lock.unlock();
        ready_.notify_all();
        return true;
    }

    // Blocks until the next item in order is available. Returns
    // std::nullopt once closed and there is no next item and never will be.
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] { return filled_[next_index_ % capacity_] != 0 || closed_; });
        const std::size_t slot = next_index_ % capacity_;
        if (filled_[slot] == 0) {
            return std::nullopt;
        }
        T item = std::move(slots_[slot]);
        filled_[slot] = 0;
        ++next_index_;
        lock.unlock();
        room_.notify_all();
        return item;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        room_.notify_all();
        ready_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable room_;
    std::condition_variable ready_;
    std::size_t capacity_;
    std::size_t next_index_;
    std::vector<T> slots_;
    std::vector<std::uint8_t> filled_;
    bool closed_ = false;
};

} // namespace sdl
