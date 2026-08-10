#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace sdl {

// A bounded, thread-safe MPMC queue providing backpressure between producer
// worker threads and a consumer. Producers block in push() when the queue is
// full; the consumer blocks in pop() when the queue is empty.
//
// close() signals that no more items will ever be pushed: any push() already
// blocked or newly called returns false without enqueuing, while pop()
// continues to drain whatever items remain and only returns std::nullopt
// once the queue is both closed and empty. This lets the consumer finish
// processing in-flight items instead of losing them on shutdown.
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {}

    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;

    // Blocks while the queue is full and open. Returns false (item not
    // enqueued) if the queue is or becomes closed; true otherwise.
    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] { return items_.size() < capacity_ || closed_; });
        if (closed_) {
            return false;
        }
        items_.push_back(std::move(item));
        lock.unlock();
        not_empty_.notify_one();
        return true;
    }

    // Blocks while the queue is empty and open. Returns std::nullopt once
    // the queue is closed and fully drained.
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return !items_.empty() || closed_; });
        if (items_.empty()) {
            return std::nullopt;
        }
        T item = std::move(items_.front());
        items_.pop_front();
        lock.unlock();
        not_full_.notify_one();
        return item;
    }

    // Signals that no more items will be pushed. Safe to call once, from
    // any thread; wakes every blocked push() and pop().
    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        not_full_.notify_all();
        not_empty_.notify_all();
    }

    std::size_t capacity() const { return capacity_; }

    // Approximate — for diagnostics/tests only; may be stale the instant
    // after it's read under concurrent use.
    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return items_.size();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
    std::deque<T> items_;
    std::size_t capacity_;
    bool closed_ = false;
};

} // namespace sdl
