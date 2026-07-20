/**
 * @file work_stealing_queue.h
 * @brief Lock-free work-stealing deque (Chase-Lev algorithm).
 *
 * Implements a fixed-capacity circular deque where the owner thread pushes
 * and pops from the bottom (LIFO — cache-warm execution) and thief threads
 * steal from the top (FIFO — coarse-grained work distribution).
 *
 * The Chase-Lev algorithm provides:
 * - Wait-free push (owner only, no contention)
 * - Wait-free pop (owner only, CAS on contention with steal)
 * - Lock-free steal (thieves, single CAS)
 *
 * Capacity must be a power of 2 for efficient modular indexing. The queue
 * does NOT grow — if full, push fails. Size the queue for worst-case job count.
 *
 * Memory ordering:
 * - bottom_: relaxed loads by owner, release stores (visible to thieves)
 * - top_: acquire loads, CAS with acq_rel (contended between thieves)
 * - Fence between bottom store and buffer write ensures visibility
 *
 * Reference: "Dynamic Circular Work-Stealing Deque" — Chase & Lev, SPAA 2005.
 *
 * Thread safety: One owner (push/pop), multiple thieves (steal). Lock-free.
 */

#pragma once

#include <pulse/math/math_common.h>
#include <pulse/utilities/assert.h>
#include <pulse/utilities/type_traits.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace pulse {
namespace jobs {

/**
 * @class WorkStealingQueue
 * @brief Lock-free work-stealing deque for job scheduling.
 *
 * @tparam T Element type (typically Job*).
 * @tparam Capacity Maximum number of elements. Must be a power of 2.
 */
template <typename T, std::size_t Capacity = 4096>
class WorkStealingQueue {
    static_assert(Capacity > 0, "WorkStealingQueue capacity must be > 0");
    static_assert(util::is_power_of_2_v<Capacity>,
                  "WorkStealingQueue capacity must be a power of 2");

public:
    static constexpr std::size_t Mask = Capacity - 1;

    // ── Construction ─────────────────────────────────────────────────────

    WorkStealingQueue() noexcept : bottom_(0), top_(0) {}

    // Non-copyable, non-movable (contains atomics).
    WorkStealingQueue(const WorkStealingQueue&) = delete;
    WorkStealingQueue& operator=(const WorkStealingQueue&) = delete;

    // ── Owner API (single thread — the worker that owns this queue) ──────

    /// Push an item onto the bottom of the deque.
    /// Only the owner thread may call this.
    /// Returns false if the queue is full.
    PULSE_FORCE_INLINE bool push(const T& item) noexcept {
        int64_t b = bottom_.load(std::memory_order_relaxed);
        int64_t t = top_.load(std::memory_order_acquire);

        if (b - t >= static_cast<int64_t>(Capacity)) {
            return false; // Full
        }

        buffer_[static_cast<std::size_t>(b) & Mask] = item;
        // Ensure the item is written before bottom is updated.
        std::atomic_thread_fence(std::memory_order_release);
        bottom_.store(b + 1, std::memory_order_relaxed);
        return true;
    }

    /// Pop an item from the bottom of the deque (LIFO).
    /// Only the owner thread may call this.
    /// Returns false if the queue is empty.
    PULSE_FORCE_INLINE bool pop(T& item) noexcept {
        int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
        bottom_.store(b, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        int64_t t = top_.load(std::memory_order_relaxed);

        if (t <= b) {
            // Non-empty
            item = buffer_[static_cast<std::size_t>(b) & Mask];
            if (t == b) {
                // Last element — race with steal. Try to claim it.
                if (!top_.compare_exchange_strong(t, t + 1,
                        std::memory_order_seq_cst,
                        std::memory_order_relaxed)) {
                    // Thief got it first.
                    bottom_.store(b + 1, std::memory_order_relaxed);
                    return false;
                }
                bottom_.store(b + 1, std::memory_order_relaxed);
            }
            return true;
        } else {
            // Empty
            bottom_.store(b + 1, std::memory_order_relaxed);
            return false;
        }
    }

    // ── Thief API (called by other worker threads) ───────────────────────

    /// Steal an item from the top of the deque (FIFO).
    /// Any thread may call this concurrently.
    /// Returns false if the queue is empty or contention lost.
    PULSE_FORCE_INLINE bool steal(T& item) noexcept {
        int64_t t = top_.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        int64_t b = bottom_.load(std::memory_order_acquire);

        if (t >= b) {
            return false; // Empty
        }

        item = buffer_[static_cast<std::size_t>(t) & Mask];

        // Try to advance top. If another thief beat us, we fail.
        if (!top_.compare_exchange_strong(t, t + 1,
                std::memory_order_seq_cst,
                std::memory_order_relaxed)) {
            return false; // Contention — another thief won.
        }

        return true;
    }

    // ── Queries (approximate — for diagnostics only) ─────────────────────

    /// Approximate number of items in the queue.
    [[nodiscard]] std::size_t size() const noexcept {
        int64_t b = bottom_.load(std::memory_order_relaxed);
        int64_t t = top_.load(std::memory_order_relaxed);
        return static_cast<std::size_t>(b >= t ? b - t : 0);
    }

    /// Check if approximately empty.
    [[nodiscard]] bool empty() const noexcept {
        return size() == 0;
    }

    // ── Reset (owner only, when no thieves active) ───────────────────────

    /// Reset the queue. Only call when the system is idle.
    void reset() noexcept {
        bottom_.store(0, std::memory_order_relaxed);
        top_.store(0, std::memory_order_relaxed);
    }

private:
    T buffer_[Capacity];

    // Cache-line separated indices to prevent false sharing.
    // Bottom: modified by owner only.
    // Top: contended by thieves (CAS).
    PULSE_CACHE_ALIGN std::atomic<int64_t> bottom_;
    PULSE_CACHE_ALIGN std::atomic<int64_t> top_;
};

} // namespace jobs
} // namespace pulse
