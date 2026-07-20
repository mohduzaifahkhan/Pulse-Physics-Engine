/**
 * @file job_system.h
 * @brief Main job system orchestrator — init, shutdown, dispatch, and wait.
 *
 * The JobSystem manages a pool of worker threads, each with their own
 * work-stealing queue and job allocator. Jobs are submitted to the
 * submitting thread's local queue and executed LIFO (cache-warm). When
 * a worker's queue is empty, it steals work from other workers' queues.
 *
 * Fork-join parallelism:
 *   1. Create an AtomicCounter initialized to N.
 *   2. Submit N jobs, each linked to the counter.
 *   3. Call waitForCounter() — the waiting thread steals work while waiting.
 *   4. When counter reaches 0, wait returns and all jobs are complete.
 *
 * Lifecycle:
 *   JobSystem::init();       // Start workers
 *   // ... game loop ...
 *   //   JobSystem::beginFrame();      // Reset per-thread allocators
 *   //   submit jobs, wait on counters
 *   //   JobSystem::endFrame();
 *   // ... end game loop ...
 *   JobSystem::shutdown();   // Join all workers
 *
 * Thread safety: init/shutdown are NOT thread-safe (call from main thread).
 * submit/wait/createJob are thread-safe (callable from any worker).
 */

#pragma once

#include <pulse/math/math_common.h>
#include <pulse/jobs/job.h>
#include <pulse/jobs/counter.h>
#include <pulse/jobs/worker_thread.h>
#include <pulse/utilities/assert.h>
#include <pulse/utilities/profiler.h>

#include <atomic>
#include <cstdint>
#include <thread>

namespace pulse {
namespace jobs {

/**
 * @class JobSystem
 * @brief Global job system manager — singleton with static interface.
 *
 * All methods are static. The system maintains global state internally.
 * Thread-to-worker mapping uses thread_local storage for O(1) lookups.
 */
class JobSystem {
public:
    // ── Lifecycle ────────────────────────────────────────────────────────

    /// Initialize the job system with the given number of worker threads.
    /// @param numWorkers Number of worker threads. 0 = auto (hardware_concurrency - 1).
    static void init(uint32_t numWorkers = 0) noexcept;

    /// Shut down the job system. Waits for all workers to finish.
    /// Must be called from the main thread.
    static void shutdown() noexcept;

    /// Check if the job system has been initialized.
    [[nodiscard]] static bool isInitialized() noexcept;

    // ── Frame management ─────────────────────────────────────────────────

    /// Call at the start of each frame. Resets per-thread job allocators.
    static void beginFrame() noexcept;

    /// Call at the end of each frame. Currently a no-op placeholder for
    /// future profiler frame merging.
    static void endFrame() noexcept;

    // ── Job creation ─────────────────────────────────────────────────────

    /// Allocate a Job from the current thread's allocator.
    /// Returns a zeroed job ready to be initialized.
    [[nodiscard]] static Job* createJob() noexcept;

    /// Allocate and initialize a job with a function pointer.
    [[nodiscard]] static Job* createJob(Job::EntryFn fn,
                                        AtomicCounter* counter = nullptr,
                                        JobPriority priority = JobPriority::Normal) noexcept;

    /// Allocate and initialize a job with a lambda.
    template <typename Func>
    [[nodiscard]] static Job* createJobLambda(Func&& fn,
                                              AtomicCounter* counter = nullptr,
                                              JobPriority priority = JobPriority::Normal) noexcept {
        Job* job = createJob();
        if (PULSE_LIKELY(job != nullptr)) {
            initJobLambda(job, static_cast<Func&&>(fn), counter, priority);
        }
        return job;
    }

    // ── Job submission ───────────────────────────────────────────────────

    /// Submit a job for execution. The job is pushed onto the calling
    /// thread's local queue. Other workers may steal it.
    static void submit(Job* job) noexcept;

    /// Submit multiple jobs at once.
    static void submitBatch(Job** jobs, uint32_t count) noexcept;

    // ── Synchronization ──────────────────────────────────────────────────

    /// Wait until the counter reaches 0. The calling thread steals and
    /// executes jobs while waiting — no CPU time is wasted.
    static void waitForCounter(AtomicCounter* counter) noexcept;

    // ── Thread info ──────────────────────────────────────────────────────

    /// Get the current thread's worker index (0 = main thread).
    [[nodiscard]] static uint32_t currentThreadIndex() noexcept;

    /// Get the total number of threads (main + workers).
    [[nodiscard]] static uint32_t threadCount() noexcept;

    /// Get the number of worker threads (excluding main).
    [[nodiscard]] static uint32_t workerCount() noexcept;

    /// Get a worker's context by index.
    [[nodiscard]] static WorkerContext* getWorkerContext(uint32_t index) noexcept;

private:
    // ── Internal ─────────────────────────────────────────────────────────

    /// Worker thread entry point.
    static void workerEntry(uint32_t threadIndex) noexcept;

    /// Try to execute one job from local queue, or steal from others.
    /// Returns true if a job was executed.
    static bool tryExecuteOne(uint32_t threadIndex) noexcept;

    /// Execute a single job and handle counter decrement.
    static void executeJob(Job* job) noexcept;

    /// Try to steal a job from another worker's queue.
    static Job* tryStealJob(uint32_t thiefIndex) noexcept;
};

} // namespace jobs
} // namespace pulse
