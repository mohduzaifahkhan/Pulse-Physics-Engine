/**
 * @file job_allocator.h
 * @brief Per-thread pool allocator for Job objects — zero-overhead allocation.
 *
 * Pre-allocates a fixed number of Job slots per thread. Allocation is O(1)
 * free-list pop, deallocation is O(1) free-list push. The pool is reset
 * each frame (or when the system is idle) so no per-job free is needed
 * in the common case.
 *
 * Capacity is fixed at compile time. If exhausted, the allocator asserts
 * in debug and returns nullptr in release. The caller should never exceed
 * the job budget — physics workloads have bounded, predictable job counts.
 *
 * Memory layout:
 *   ┌──────┬──────┬──────┬──────┬─────┬──────┐
 *   │ Job0 │ Job1 │ Job2 │ Job3 │ ... │ JobN │  (contiguous, cache-aligned)
 *   └──────┴──────┴──────┴──────┴─────┴──────┘
 *   Free list: intrusive linked list through first 8 bytes of each free slot.
 *
 * Thread safety: NOT thread-safe. Each worker thread owns one JobAllocator.
 */

#pragma once

#include <pulse/math/math_common.h>
#include <pulse/jobs/job.h>
#include <pulse/utilities/assert.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

namespace pulse {
namespace jobs {

/**
 * @class JobAllocator
 * @brief Fixed-capacity pool allocator for Job objects.
 *
 * @tparam MaxJobs Maximum number of jobs that can be alive simultaneously.
 *                 Must be large enough for worst-case frame job count.
 */
template <std::size_t MaxJobs = 4096>
class JobAllocator {
    static_assert(MaxJobs > 0, "JobAllocator must have capacity > 0");

public:
    // ── Construction / Destruction ───────────────────────────────────────

    JobAllocator() noexcept {
        // Allocate aligned storage for all jobs.
        storage_ = static_cast<Job*>(
#if defined(PULSE_COMPILER_MSVC)
            _aligned_malloc(MaxJobs * sizeof(Job), alignof(Job))
#else
            std::aligned_alloc(alignof(Job), MaxJobs * sizeof(Job))
#endif
        );
        PULSE_ASSERT_MSG(storage_ != nullptr, "JobAllocator: allocation failed");
        reset();
    }

    ~JobAllocator() noexcept {
#if defined(PULSE_COMPILER_MSVC)
        _aligned_free(storage_);
#else
        std::free(storage_);
#endif
    }

    // Non-copyable, non-movable.
    JobAllocator(const JobAllocator&) = delete;
    JobAllocator& operator=(const JobAllocator&) = delete;

    // ── Allocation ───────────────────────────────────────────────────────

    /// Allocate a single Job. Returns nullptr if the pool is exhausted.
    [[nodiscard]] PULSE_FORCE_INLINE Job* allocate() noexcept {
        if (PULSE_UNLIKELY(freeList_ == nullptr)) {
            PULSE_ASSERT_MSG(false, "JobAllocator: pool exhausted");
            return nullptr;
        }

        // Pop from free list.
        Job* job = freeList_;
        freeList_ = *reinterpret_cast<Job**>(job);
        allocatedCount_++;

        // Zero-initialize the job.
        std::memset(job, 0, sizeof(Job));
        return job;
    }

    /// Free a single Job back to the pool.
    /// Not typically needed — use reset() at frame boundaries instead.
    PULSE_FORCE_INLINE void free(Job* job) noexcept {
        PULSE_ASSERT(job != nullptr);
        PULSE_ASSERT(owns(job));

        // Push onto free list.
        *reinterpret_cast<Job**>(job) = freeList_;
        freeList_ = job;
        allocatedCount_--;
    }

    // ── Reset ────────────────────────────────────────────────────────────

    /// Reset the pool — all jobs become available. Call at frame boundaries.
    /// Outstanding job pointers become invalid after this call.
    void reset() noexcept {
        // Rebuild free list by threading through all slots.
        freeList_ = &storage_[0];
        for (std::size_t i = 0; i < MaxJobs - 1; ++i) {
            *reinterpret_cast<Job**>(&storage_[i]) = &storage_[i + 1];
        }
        *reinterpret_cast<Job**>(&storage_[MaxJobs - 1]) = nullptr;
        allocatedCount_ = 0;
    }

    // ── Queries ──────────────────────────────────────────────────────────

    /// Number of currently allocated jobs.
    [[nodiscard]] std::size_t allocatedCount() const noexcept {
        return allocatedCount_;
    }

    /// Total capacity.
    [[nodiscard]] static constexpr std::size_t capacity() noexcept {
        return MaxJobs;
    }

    /// Check if this allocator owns the given job pointer.
    [[nodiscard]] bool owns(const Job* job) const noexcept {
        return job >= storage_ && job < storage_ + MaxJobs;
    }

private:
    Job*        storage_        = nullptr;   ///< Contiguous aligned storage.
    Job*        freeList_       = nullptr;   ///< Intrusive free list head.
    std::size_t allocatedCount_ = 0;         ///< Number of live jobs.
};

} // namespace jobs
} // namespace pulse
