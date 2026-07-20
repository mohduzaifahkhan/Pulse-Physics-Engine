/**
 * @file job.h
 * @brief Job descriptor — the fundamental unit of work in the job system.
 *
 * A Job packages a function pointer and a small inline data buffer. The inline
 * storage (48 bytes) is sized to hold typical lambda captures (a few pointers,
 * loop bounds, array pointers) without any heap allocation. For larger payloads,
 * the user passes a pointer to externally-owned data.
 *
 * Jobs are allocated from per-thread pools (JobAllocator) and recycled each
 * frame. They should never be heap-allocated individually.
 *
 * Memory layout (128 bytes = 2 cache lines):
 *   [function (8)] [data (8)] [counter (8)] [priority (1)] [pad (7)]
 *   [inlineData (48 bytes, 16-byte aligned)]
 *   [padding to 128 bytes]
 *
 * Thread safety: A Job is written by its creator (single thread) and read by
 * its executor (single thread). No concurrent access by design.
 */

#pragma once

#include <pulse/math/math_common.h>
#include <pulse/jobs/counter.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>

namespace pulse {
namespace jobs {

// ── Job priority levels ──────────────────────────────────────────────────────

enum class JobPriority : uint8_t {
    High   = 0,   ///< Critical path: broadphase, constraint solving.
    Normal = 1,   ///< Default: most physics work.
    Low    = 2    ///< Background: island sleep detection, statistics.
};

// ── Job descriptor ───────────────────────────────────────────────────────────

/**
 * @struct Job
 * @brief A single unit of work: function pointer + inline data + counter link.
 *
 * Jobs are lightweight descriptors allocated from per-thread pools. The
 * function signature is `void(Job*)` — the job itself is passed so the
 * executor can access inline data.
 */
struct PULSE_CACHE_ALIGN Job {
    /// Job entry function signature: receives the job itself for data access.
    using EntryFn = void(*)(Job*);

    // ── Fields ───────────────────────────────────────────────────────────

    EntryFn         function = nullptr;  ///< Function to execute.
    AtomicCounter*  counter  = nullptr;  ///< Optional counter to decrement on completion.
    JobPriority     priority = JobPriority::Normal;

    /// Inline data storage for lambda captures / small payloads.
    /// 48 bytes: fits 6 pointers, or a Vec3 + pointer + loop bounds, etc.
    static constexpr std::size_t InlineCapacity = 48;
    alignas(16) uint8_t inlineData[InlineCapacity] = {};

    // ── Inline data helpers ──────────────────────────────────────────────

    /// Store a trivially-copyable value into inline storage.
    template <typename T>
    PULSE_FORCE_INLINE void setData(const T& value) noexcept {
        static_assert(sizeof(T) <= InlineCapacity,
                      "Data too large for Job inline storage");
        static_assert(std::is_trivially_copyable_v<T>,
                      "Job inline data must be trivially copyable");
        std::memcpy(inlineData, &value, sizeof(T));
    }

    /// Retrieve the inline data as type T.
    template <typename T>
    [[nodiscard]] PULSE_FORCE_INLINE T& getData() noexcept {
        static_assert(sizeof(T) <= InlineCapacity,
                      "Data type too large for Job inline storage");
        return *reinterpret_cast<T*>(inlineData);
    }

    template <typename T>
    [[nodiscard]] PULSE_FORCE_INLINE const T& getData() const noexcept {
        static_assert(sizeof(T) <= InlineCapacity,
                      "Data type too large for Job inline storage");
        return *reinterpret_cast<const T*>(inlineData);
    }
};

// Verify Job fits in expected cache-line layout.
static_assert(sizeof(Job) <= 128, "Job should fit in 2 cache lines");
static_assert(alignof(Job) >= PULSE_CACHE_LINE, "Job must be cache-line aligned");

// ── Job creation helpers ─────────────────────────────────────────────────────

/// Create a job from a function pointer and optional data.
PULSE_FORCE_INLINE void initJob(Job* job, Job::EntryFn fn,
                                AtomicCounter* counter = nullptr,
                                JobPriority priority = JobPriority::Normal) noexcept {
    job->function = fn;
    job->counter  = counter;
    job->priority = priority;
}

/// Create a job wrapping a small callable (lambda with captures ≤ 48 bytes).
/// The callable is stored inline — no heap allocation.
///
/// Usage:
///   Job* j = jobAlloc.allocate();
///   initJobLambda(j, [&positions, dt](Job*) {
///       for (auto& p : positions) p += velocity * dt;
///   });
template <typename Func>
PULSE_FORCE_INLINE void initJobLambda(Job* job, Func&& fn,
                                      AtomicCounter* counter = nullptr,
                                      JobPriority priority = JobPriority::Normal) noexcept {
    using FuncType = std::decay_t<Func>;
    static_assert(sizeof(FuncType) <= Job::InlineCapacity,
                  "Lambda too large for Job inline storage. Reduce captures or pass data via pointer.");
    static_assert(std::is_trivially_destructible_v<FuncType>,
                  "Job lambdas must be trivially destructible (no RAII captures).");

    // Store the lambda in inline data.
    new (job->inlineData) FuncType(static_cast<Func&&>(fn));

    // Set the entry function to a trampoline that invokes the stored lambda.
    job->function = [](Job* self) {
        auto& storedFn = *reinterpret_cast<FuncType*>(self->inlineData);
        storedFn(self);
    };

    job->counter  = counter;
    job->priority = priority;
}

} // namespace jobs
} // namespace pulse
