/**
 * @file worker_thread.h
 * @brief Worker thread descriptor with thread-local state.
 *
 * Each WorkerContext bundles the per-thread resources needed by a job system
 * worker: a work-stealing queue, a job allocator, and thread metadata.
 * The actual std::thread is owned and managed by the JobSystem.
 *
 * The main thread (index 0) also has a WorkerContext but does NOT have an
 * associated std::thread — it uses the calling thread.
 *
 * Thread safety: Each WorkerContext is owned by exactly one thread.
 * The work-stealing queue is the only cross-thread access point (via steal).
 */

#pragma once

#include <pulse/math/math_common.h>
#include <pulse/jobs/job.h>
#include <pulse/jobs/work_stealing_queue.h>
#include <pulse/jobs/job_allocator.h>

#include <atomic>
#include <cstdint>

namespace pulse {
namespace jobs {

/// Maximum number of worker threads supported.
constexpr uint32_t MaxWorkerThreads = 64;

/// Number of jobs each worker can buffer in its local queue.
constexpr std::size_t WorkerQueueCapacity = 4096;

/// Number of jobs each worker can allocate per frame.
constexpr std::size_t WorkerJobCapacity = 4096;

/**
 * @struct WorkerContext
 * @brief Per-thread state for a job system worker.
 *
 * Contains the thread's local work-stealing queue and job allocator.
 * Allocated once at job system init and destroyed at shutdown.
 */
struct WorkerContext {
    /// The worker's local work-stealing deque.
    /// Owner pushes/pops, other workers steal.
    WorkStealingQueue<Job*, WorkerQueueCapacity> queue;

    /// Per-thread job allocator. Reset each frame.
    JobAllocator<WorkerJobCapacity> allocator;

    /// This worker's index (0 = main thread, 1..N = workers).
    uint32_t threadIndex = 0;

    /// Flag to signal the worker to exit its loop.
    std::atomic<bool> running{false};

    /// Flag indicating the worker is actively executing a job.
    std::atomic<bool> busy{false};
};

} // namespace jobs
} // namespace pulse
