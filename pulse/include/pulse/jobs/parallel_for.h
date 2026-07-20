/**
 * @file parallel_for.h
 * @brief High-level parallel iteration API built on the job system.
 *
 * Provides parallelFor() which splits a range [0, count) into batches and
 * dispatches them across worker threads. The calling thread participates
 * in execution (via work-stealing wait) so no CPU core is idle.
 *
 * Batch sizing:
 * - Manual: parallelFor(count, batchSize, func)
 * - Automatic: parallelFor(count, func) — splits by thread count
 *
 * The callable receives (uint32_t start, uint32_t end) defining the
 * half-open range [start, end) to process.
 *
 * Usage:
 *   // Update 100K body positions in parallel
 *   parallelFor(bodyCount, 256, [&](uint32_t start, uint32_t end) {
 *       for (uint32_t i = start; i < end; ++i) {
 *           positions[i] = positions[i] + velocities[i] * dt;
 *       }
 *   });
 *
 * Thread safety: The func callable must be safe for concurrent execution
 * on disjoint index ranges. No synchronization is provided within ranges.
 */

#pragma once

#include <pulse/math/math_common.h>
#include <pulse/jobs/job_system.h>
#include <pulse/jobs/counter.h>
#include <pulse/utilities/assert.h>

#include <cstdint>
#include <type_traits>

namespace pulse {
namespace jobs {

// ── Batch descriptor (fits in Job inline data) ───────────────────────────────

/// Data passed to each batch job via inline storage.
struct ParallelForBatch {
    void*    callable;    ///< Pointer to the user's lambda (on stack of caller).
    uint32_t start;       ///< Start index (inclusive).
    uint32_t end;         ///< End index (exclusive).
};

static_assert(sizeof(ParallelForBatch) <= Job::InlineCapacity,
              "ParallelForBatch must fit in Job inline storage");

// ── parallelFor with explicit batch size ─────────────────────────────────────

/**
 * @brief Split [0, count) into batches of batchSize and execute in parallel.
 *
 * Blocks until all batches complete. The calling thread participates in
 * execution via work-stealing while waiting.
 *
 * @tparam Func Callable with signature void(uint32_t start, uint32_t end).
 * @param count     Total number of iterations.
 * @param batchSize Number of iterations per batch.
 * @param func      Function to call for each batch.
 */
template <typename Func>
void parallelFor(uint32_t count, uint32_t batchSize, Func&& func) {
    PULSE_ASSERT(batchSize > 0);

    if (count == 0) return;

    // If only one batch or system not initialized, run sequentially.
    if (count <= batchSize || !JobSystem::isInitialized() ||
        JobSystem::threadCount() <= 1) {
        func(0, count);
        return;
    }

    // Calculate number of batches.
    uint32_t numBatches = (count + batchSize - 1) / batchSize;

    // Set up completion counter.
    AtomicCounter counter(static_cast<int32_t>(numBatches));

    // Submit batch jobs.
    for (uint32_t batch = 0; batch < numBatches; ++batch) {
        uint32_t start = batch * batchSize;
        uint32_t end   = start + batchSize;
        if (end > count) end = count;

        // Each batch job captures a pointer to the user's callable.
        // The callable lives on the caller's stack and is valid until
        // waitForCounter returns.
        Job* job = JobSystem::createJob();
        PULSE_ASSERT(job != nullptr);

        auto& batchData = job->getData<ParallelForBatch>();
        batchData.callable = reinterpret_cast<void*>(&func);
        batchData.start    = start;
        batchData.end      = end;

        // Trampoline: extract the callable and invoke with range.
        using FuncType = std::decay_t<Func>;
        job->function = [](Job* self) {
            auto& data = self->getData<ParallelForBatch>();
            auto* fn = reinterpret_cast<FuncType*>(data.callable);
            (*fn)(data.start, data.end);
        };
        job->counter  = &counter;
        job->priority = JobPriority::High;

        JobSystem::submit(job);
    }

    // Wait for all batches to complete (steals work while waiting).
    JobSystem::waitForCounter(&counter);
}

// ── parallelFor with automatic batch sizing ──────────────────────────────────

/**
 * @brief Split [0, count) across available threads with automatic batch sizing.
 *
 * Uses threadCount * 4 batches for good load balancing (enough granularity
 * to handle uneven workloads, but not so many that overhead dominates).
 */
template <typename Func>
void parallelFor(uint32_t count, Func&& func) {
    if (count == 0) return;

    uint32_t threads = JobSystem::isInitialized() ? JobSystem::threadCount() : 1;
    // 4x oversubscription for load balancing.
    uint32_t batchSize = (count + threads * 4 - 1) / (threads * 4);
    if (batchSize < 1) batchSize = 1;

    parallelFor(count, batchSize, static_cast<Func&&>(func));
}

} // namespace jobs
} // namespace pulse
