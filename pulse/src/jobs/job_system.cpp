/**
 * @file job_system.cpp
 * @brief Job system implementation — worker lifecycle, dispatch, and synchronization.
 *
 * Contains the global state, worker thread entry loop, work-stealing logic,
 * and the init/shutdown lifecycle. This is the only non-header TU in the
 * jobs module — everything else is header-only/inline for performance.
 */

#include <pulse/jobs/job_system.h>
#include <pulse/jobs/counter.h>
#include <pulse/jobs/worker_thread.h>
#include <pulse/utilities/assert.h>
#include <pulse/utilities/profiler.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>

namespace pulse {
namespace jobs {

// ── Global state ─────────────────────────────────────────────────────────────

namespace {

    /// All worker contexts (index 0 = main thread).
    WorkerContext* g_workers[MaxWorkerThreads] = {};

    /// Worker std::thread objects (index 0 unused — main thread).
    std::thread g_threads[MaxWorkerThreads];

    /// Total number of threads (main + workers).
    uint32_t g_threadCount = 0;

    /// Number of worker threads (excluding main).
    uint32_t g_workerCount = 0;

    /// System initialized flag.
    std::atomic<bool> g_initialized{false};

    /// Global signal for workers to shut down.
    std::atomic<bool> g_shutdownRequested{false};

} // anonymous namespace

// ── Thread-local worker index ────────────────────────────────────────────────

/// Each thread stores its own worker index for O(1) lookup.
static thread_local uint32_t tl_threadIndex = 0;

// ── Lifecycle ────────────────────────────────────────────────────────────────

void JobSystem::init(uint32_t numWorkers) noexcept {
    PULSE_ASSERT_MSG(!g_initialized.load(std::memory_order_relaxed),
                     "JobSystem::init: already initialized");

    // Auto-detect worker count.
    if (numWorkers == 0) {
        uint32_t hw = std::thread::hardware_concurrency();
        numWorkers = (hw > 1) ? (hw - 1) : 1;
    }

    if (numWorkers > MaxWorkerThreads - 1) {
        numWorkers = MaxWorkerThreads - 1;
    }

    g_workerCount = numWorkers;
    g_threadCount = numWorkers + 1; // +1 for main thread
    g_shutdownRequested.store(false, std::memory_order_relaxed);

    // Allocate worker contexts.
    for (uint32_t i = 0; i < g_threadCount; ++i) {
        g_workers[i] = new WorkerContext();
        g_workers[i]->threadIndex = i;
        g_workers[i]->running.store(true, std::memory_order_relaxed);
    }

    // Main thread is worker 0.
    tl_threadIndex = 0;

    // Launch worker threads (indices 1..N).
    for (uint32_t i = 1; i < g_threadCount; ++i) {
        g_threads[i] = std::thread(workerEntry, i);
    }

    g_initialized.store(true, std::memory_order_release);
}

void JobSystem::shutdown() noexcept {
    if (!g_initialized.load(std::memory_order_acquire)) return;

    // Signal all workers to stop.
    g_shutdownRequested.store(true, std::memory_order_release);

    for (uint32_t i = 1; i < g_threadCount; ++i) {
        g_workers[i]->running.store(false, std::memory_order_release);
    }

    // Join all worker threads.
    for (uint32_t i = 1; i < g_threadCount; ++i) {
        if (g_threads[i].joinable()) {
            g_threads[i].join();
        }
    }

    // Free worker contexts.
    for (uint32_t i = 0; i < g_threadCount; ++i) {
        delete g_workers[i];
        g_workers[i] = nullptr;
    }

    g_threadCount = 0;
    g_workerCount = 0;
    g_initialized.store(false, std::memory_order_release);
}

bool JobSystem::isInitialized() noexcept {
    return g_initialized.load(std::memory_order_acquire);
}

// ── Frame management ─────────────────────────────────────────────────────────

void JobSystem::beginFrame() noexcept {
    PULSE_ASSERT(isInitialized());

    // Reset all per-thread job allocators.
    for (uint32_t i = 0; i < g_threadCount; ++i) {
        g_workers[i]->allocator.reset();
    }
}

void JobSystem::endFrame() noexcept {
    // Currently a no-op. Future: merge per-thread profiler data.
}

// ── Job creation ─────────────────────────────────────────────────────────────

Job* JobSystem::createJob() noexcept {
    PULSE_ASSERT(isInitialized());
    WorkerContext* ctx = g_workers[tl_threadIndex];
    return ctx->allocator.allocate();
}

Job* JobSystem::createJob(Job::EntryFn fn, AtomicCounter* counter,
                          JobPriority priority) noexcept {
    Job* job = createJob();
    if (PULSE_LIKELY(job != nullptr)) {
        initJob(job, fn, counter, priority);
    }
    return job;
}

// ── Job submission ───────────────────────────────────────────────────────────

void JobSystem::submit(Job* job) noexcept {
    PULSE_ASSERT(job != nullptr);
    PULSE_ASSERT(job->function != nullptr);
    PULSE_ASSERT(isInitialized());

    WorkerContext* ctx = g_workers[tl_threadIndex];
    bool pushed = ctx->queue.push(job);
    PULSE_ASSERT_MSG(pushed, "JobSystem::submit: worker queue is full");
    (void)pushed;
}

void JobSystem::submitBatch(Job** jobs, uint32_t count) noexcept {
    for (uint32_t i = 0; i < count; ++i) {
        submit(jobs[i]);
    }
}

// ── Synchronization ──────────────────────────────────────────────────────────

void JobSystem::waitForCounter(AtomicCounter* counter) noexcept {
    PULSE_ASSERT(counter != nullptr);

    // Spin and steal work while waiting for the counter to reach 0.
    while (!counter->isComplete()) {
        if (!tryExecuteOne(tl_threadIndex)) {
            // No work available anywhere — yield to OS scheduler.
            std::this_thread::yield();
        }
    }
}

// ── Thread info ──────────────────────────────────────────────────────────────

uint32_t JobSystem::currentThreadIndex() noexcept {
    return tl_threadIndex;
}

uint32_t JobSystem::threadCount() noexcept {
    return g_threadCount;
}

uint32_t JobSystem::workerCount() noexcept {
    return g_workerCount;
}

WorkerContext* JobSystem::getWorkerContext(uint32_t index) noexcept {
    PULSE_ASSERT(index < g_threadCount);
    return g_workers[index];
}

// ── Worker thread entry ──────────────────────────────────────────────────────

void JobSystem::workerEntry(uint32_t threadIndex) noexcept {
    // Set thread-local index for this worker.
    tl_threadIndex = threadIndex;

    WorkerContext* ctx = g_workers[threadIndex];

    while (ctx->running.load(std::memory_order_acquire)) {
        if (!tryExecuteOne(threadIndex)) {
            // No work available — yield briefly to reduce power consumption.
            std::this_thread::yield();
        }
    }
}

// ── Internal: try to execute one job ─────────────────────────────────────────

bool JobSystem::tryExecuteOne(uint32_t threadIndex) noexcept {
    WorkerContext* ctx = g_workers[threadIndex];

    // First: try to pop from our own queue (LIFO, cache-warm).
    Job* job = nullptr;
    if (ctx->queue.pop(job)) {
        executeJob(job);
        return true;
    }

    // Second: try to steal from other workers.
    job = tryStealJob(threadIndex);
    if (job != nullptr) {
        executeJob(job);
        return true;
    }

    return false;
}

void JobSystem::executeJob(Job* job) noexcept {
    PULSE_ASSERT(job != nullptr);
    PULSE_ASSERT(job->function != nullptr);

    // Execute the job function.
    job->function(job);

    // Decrement the counter if one is attached.
    if (job->counter != nullptr) {
        job->counter->decrement();
    }
}

Job* JobSystem::tryStealJob(uint32_t thiefIndex) noexcept {
    // Round-robin steal attempt starting from a pseudo-random offset
    // to distribute stealing evenly and reduce contention.
    uint32_t total = g_threadCount;
    if (total <= 1) return nullptr;

    // Simple offset: start from (thiefIndex + 1) and wrap around.
    for (uint32_t i = 1; i < total; ++i) {
        uint32_t victimIndex = (thiefIndex + i) % total;
        WorkerContext* victim = g_workers[victimIndex];

        Job* job = nullptr;
        if (victim->queue.steal(job)) {
            return job;
        }
    }

    return nullptr;
}

} // namespace jobs
} // namespace pulse
