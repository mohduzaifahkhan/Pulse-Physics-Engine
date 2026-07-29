/**
 * @file ring_buffer.h
 * @brief Lock-free single-producer single-consumer (SPSC) ring buffer.
 *
 * Power-of-2 capacity enforced at compile time. Uses std::atomic with
 * acquire/release semantics for lock-free synchronization between exactly
 * one producer thread and one consumer thread.
 *
 * Cache-line padding between head and tail indices prevents false sharing.
 * Storage is in-place (no heap allocation).
 *
 * Use cases: job system communication, event queues, producer-consumer
 * pipelines between physics threads.
 *
 * Thread safety: Lock-free for exactly 1 producer and 1 consumer.
 * NOT safe for multiple producers or multiple consumers.
 */

#pragma once

#include <pulse/math/math_common.h>
#include <pulse/utilities/assert.h>
#include <pulse/utilities/type_traits.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

namespace pulse {
namespace util {

/**
 * @class RingBuffer
 * @brief Lock-free SPSC ring buffer with in-place storage.
 *
 * @tparam T Element type (must be trivially copyable for lock-free guarantee).
 * @tparam N Capacity (must be a power of 2).
 */
template <typename T, std::size_t N>
class RingBuffer {
    static_assert(N > 0, "RingBuffer capacity must be > 0");
    static_assert(is_power_of_2_v<N>, "RingBuffer capacity must be a power of 2");

public:
    static constexpr std::size_t Capacity = N;
    static constexpr std::size_t Mask = N - 1;

    // ── Construction ─────────────────────────────────────────────────────

    RingBuffer() noexcept : head_(0), tail_(0) {}

    // Non-copyable, non-movable (due to atomics).
    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

    // ── Producer API (single thread only) ────────────────────────────────

    /// Try to push an element. Returns false if the buffer is full.
    /// Only call from the producer thread.
    PULSE_FORCE_INLINE bool tryPush(const T& item) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t tail = tail_.load(std::memory_order_acquire);

        if (nextIndex(head) == tail) {
            return false; // Full
        }

        constructAt(head & Mask, item);
        head_.store(nextIndex(head), std::memory_order_release);
        return true;
    }

    /// Try to push an element by move. Returns false if the buffer is full.
    PULSE_FORCE_INLINE bool tryPush(T&& item) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t tail = tail_.load(std::memory_order_acquire);

        if (nextIndex(head) == tail) {
            return false; // Full
        }

        constructAt(head & Mask, static_cast<T&&>(item));
        head_.store(nextIndex(head), std::memory_order_release);
        return true;
    }

    // ── Consumer API (single thread only) ────────────────────────────────

    /// Try to pop an element. Returns false if the buffer is empty.
    /// Only call from the consumer thread.
    PULSE_FORCE_INLINE bool tryPop(T& item) noexcept {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        const std::size_t head = head_.load(std::memory_order_acquire);

        if (tail == head) {
            return false; // Empty
        }

        item = static_cast<T&&>(*ptrAt(tail & Mask));
        destroyAt(tail & Mask);
        tail_.store(nextIndex(tail), std::memory_order_release);
        return true;
    }

    // ── Queries (approximate, for diagnostics) ───────────────────────────

    /// Approximate number of elements (may be slightly stale in concurrent use).
    [[nodiscard]] std::size_t size() const noexcept {
        const std::size_t head = head_.load(std::memory_order_acquire);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        return (head >= tail) ? (head - tail) : (2 * N - tail + head);
    }

    /// Check if approximately empty.
    [[nodiscard]] bool empty() const noexcept {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

    /// Check if approximately full.
    [[nodiscard]] bool full() const noexcept {
        const std::size_t head = head_.load(std::memory_order_acquire);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        return nextIndex(head) == tail;
    }

private:
    /// Increment index with wraparound. We use virtual indices (0..2N-1)
    /// to distinguish empty from full, and mask when accessing storage.
    [[nodiscard]] static constexpr std::size_t nextIndex(std::size_t idx) noexcept {
        return (idx + 1) & (2 * N - 1);
    }

    // ── Storage ──────────────────────────────────────────────────────────

    /// Element storage (uninitialized).
    alignas(alignof(T)) unsigned char storage_[sizeof(T) * N];

    [[nodiscard]] T* ptrAt(std::size_t i) noexcept {
        return reinterpret_cast<T*>(storage_ + sizeof(T) * i);
    }

    void constructAt(std::size_t i, const T& val) noexcept {
        new (ptrAt(i)) T(val);
    }

    void constructAt(std::size_t i, T&& val) noexcept {
        new (ptrAt(i)) T(static_cast<T&&>(val));
    }

    void destroyAt(std::size_t i) noexcept {
        if (!std::is_trivially_destructible<T>::value) {
            ptrAt(i)->~T();
        }
    }

    // ── Cache-line separated atomics ─────────────────────────────────────
    //
    // Head is written by producer, read by consumer.
    // Tail is written by consumer, read by producer.
    // Separate cache lines prevent false sharing.

    PULSE_CACHE_ALIGN std::atomic<std::size_t> head_;
    PULSE_CACHE_ALIGN std::atomic<std::size_t> tail_;
};

} // namespace util
} // namespace pulse
