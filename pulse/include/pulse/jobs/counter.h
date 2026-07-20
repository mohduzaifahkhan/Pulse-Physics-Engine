/**
 * @file counter.h
 * @brief Atomic counter for job dependency tracking and fork-join synchronization.
 *
 * A wait-free atomic counter that jobs decrement upon completion. The dispatching
 * thread can wait on a counter to implement fork-join parallelism: submit N jobs
 * sharing a counter initialized to N, then call waitForCounter() which spins
 * (and steals work) until the counter reaches zero.
 *
 * Memory layout: single cache-line-padded atomic int32_t to avoid false sharing
 * when multiple threads decrement concurrently.
 *
 * Thread safety: fully thread-safe. All operations use atomic instructions with
 * appropriate memory ordering.
 */

#pragma once

#include <pulse/math/math_common.h>
#include <pulse/utilities/assert.h>

#include <atomic>
#include <cstdint>

namespace pulse {
namespace jobs {

/**
 * @struct AtomicCounter
 * @brief Wait-free atomic counter for job completion tracking.
 *
 * Usage:
 *   AtomicCounter counter;
 *   counter.reset(4);                  // 4 jobs to complete
 *   for (int i = 0; i < 4; ++i)
 *       submit(job, &counter);         // Each job decrements on finish
 *   JobSystem::waitForCounter(&counter); // Blocks (stealing work) until 0
 */
struct PULSE_CACHE_ALIGN AtomicCounter {
    std::atomic<int32_t> value{0};

    // ── Construction ─────────────────────────────────────────────────────

    /// Default: counter at 0 (complete).
    AtomicCounter() noexcept = default;

    /// Initialize with a specific count.
    explicit AtomicCounter(int32_t initialCount) noexcept
        : value(initialCount) {}

    // Non-copyable, non-movable (contains atomic).
    AtomicCounter(const AtomicCounter&) = delete;
    AtomicCounter& operator=(const AtomicCounter&) = delete;

    // ── Operations ───────────────────────────────────────────────────────

    /// Reset the counter to n. Only call when no jobs are in flight.
    PULSE_FORCE_INLINE void reset(int32_t n) noexcept {
        PULSE_ASSERT(n >= 0);
        value.store(n, std::memory_order_release);
    }

    /// Decrement the counter by 1 (called by a completing job).
    /// Returns the value BEFORE decrement (so caller can check if it was the last).
    [[nodiscard]] PULSE_FORCE_INLINE int32_t decrement() noexcept {
        int32_t prev = value.fetch_sub(1, std::memory_order_acq_rel);
        PULSE_ASSERT(prev > 0); // Should never go below 0.
        return prev;
    }

    /// Increment the counter by 1 (used when dynamically adding jobs).
    PULSE_FORCE_INLINE void increment() noexcept {
        value.fetch_add(1, std::memory_order_acq_rel);
    }

    /// Check if all jobs have completed (counter == 0).
    [[nodiscard]] PULSE_FORCE_INLINE bool isComplete() const noexcept {
        return value.load(std::memory_order_acquire) <= 0;
    }

    /// Get current count (approximate in concurrent context).
    [[nodiscard]] PULSE_FORCE_INLINE int32_t load() const noexcept {
        return value.load(std::memory_order_acquire);
    }
};

} // namespace jobs
} // namespace pulse
