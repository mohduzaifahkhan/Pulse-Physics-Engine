/**
 * @file jobs_bench.cpp
 * @brief Performance benchmarks for the Pulse job system module.
 *
 * Benchmarks:
 * 1. Job allocation throughput (per-thread pool allocator)
 * 2. WorkStealingQueue push/pop throughput (single-threaded)
 * 3. Single job submit + wait round-trip latency
 * 4. Batch job throughput (N jobs submit + wait)
 * 5. parallelFor scaling (1 to N threads)
 * 6. Work stealing efficiency (imbalanced load)
 * 7. Fork-join overhead (serial vs. parallel)
 */

#define PULSE_DISABLE_ASSERTS 1
#define PULSE_DISABLE_PROFILING 1

#include <pulse/jobs/counter.h>
#include <pulse/jobs/work_stealing_queue.h>
#include <pulse/jobs/job_allocator.h>
#include <pulse/jobs/job.h>
#include <pulse/jobs/worker_thread.h>
#include <pulse/jobs/job_system.h>
#include <pulse/jobs/parallel_for.h>

#include <pulse/math/math_common.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <atomic>
#include <cmath>

// ── Benchmark timer ──────────────────────────────────────────────────────────

static double benchmarkMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(
        high_resolution_clock::now().time_since_epoch()
    ).count();
}

#define BENCH_START() double bench_start_ = benchmarkMs()
#define BENCH_END(label, ops) do { \
    double bench_elapsed_ = benchmarkMs() - bench_start_; \
    double bench_opsPerSec_ = (ops) / (bench_elapsed_ / 1000.0); \
    std::printf("  %-45s %8.3f ms  (%12.0f ops/sec)\n", \
                label, bench_elapsed_, bench_opsPerSec_); \
} while(0)

#define BENCH_END_SIMPLE(label) do { \
    double bench_elapsed_ = benchmarkMs() - bench_start_; \
    std::printf("  %-45s %8.3f ms\n", label, bench_elapsed_); \
} while(0)

// Prevent the compiler from optimizing away results.
template <typename T>
static void doNotOptimize(T&& val) {
    volatile auto* p = &val;
    (void)p;
}

// ── 1. Job Allocation Throughput ─────────────────────────────────────────────

void bench_job_allocation() {
    constexpr int Iters = 10;
    constexpr std::size_t Cap = 4096;

    pulse::jobs::JobAllocator<Cap> allocator;

    // Allocate all, then reset — measure alloc throughput.
    {
        BENCH_START();
        for (int iter = 0; iter < Iters; ++iter) {
            for (std::size_t i = 0; i < Cap; ++i) {
                auto* job = allocator.allocate();
                doNotOptimize(job);
            }
            allocator.reset();
        }
        BENCH_END("Allocate 4096 jobs (x10 iters)", static_cast<double>(Cap) * Iters);
    }

    // Allocate/free cycle — measure alloc+free pair throughput.
    {
        BENCH_START();
        for (int iter = 0; iter < Iters; ++iter) {
            for (std::size_t i = 0; i < Cap; ++i) {
                auto* job = allocator.allocate();
                allocator.free(job);
            }
        }
        BENCH_END("Alloc+free cycle 4096 (x10 iters)", static_cast<double>(Cap) * Iters);
    }
}

// ── 2. WorkStealingQueue Push/Pop Throughput ─────────────────────────────────

void bench_wsq_throughput() {
    constexpr int Iters = 100;
    constexpr std::size_t Cap = 4096;

    pulse::jobs::WorkStealingQueue<int, Cap> queue;

    // Push N, then pop N — measure round-trip throughput.
    {
        BENCH_START();
        for (int iter = 0; iter < Iters; ++iter) {
            for (int i = 0; i < static_cast<int>(Cap); ++i) {
                queue.push(i);
            }
            int item;
            for (std::size_t i = 0; i < Cap; ++i) {
                queue.pop(item);
                doNotOptimize(item);
            }
        }
        BENCH_END("WSQ push+pop 4096 (x100 iters)", static_cast<double>(Cap * 2) * Iters);
    }

    // Push N, then steal N — measure steal throughput.
    {
        BENCH_START();
        for (int iter = 0; iter < Iters; ++iter) {
            for (int i = 0; i < static_cast<int>(Cap); ++i) {
                queue.push(i);
            }
            int item;
            for (std::size_t i = 0; i < Cap; ++i) {
                queue.steal(item);
                doNotOptimize(item);
            }
        }
        BENCH_END("WSQ push+steal 4096 (x100 iters)", static_cast<double>(Cap * 2) * Iters);
    }
}

// ── 3. Single Job Submit + Wait Latency ──────────────────────────────────────

void bench_single_job_latency() {
    pulse::jobs::JobSystem::init(2);

    constexpr int N = 10000;

    {
        pulse::jobs::JobSystem::beginFrame();
        BENCH_START();
        for (int i = 0; i < N; ++i) {
            pulse::jobs::AtomicCounter counter(1);
            auto* job = pulse::jobs::JobSystem::createJobLambda(
                [](pulse::jobs::Job*) {
                    // Minimal work — measuring dispatch overhead.
                    volatile int x = 0;
                    (void)x;
                }, &counter);
            pulse::jobs::JobSystem::submit(job);
            pulse::jobs::JobSystem::waitForCounter(&counter);
        }
        BENCH_END("Single job submit+wait (x10K)", static_cast<double>(N));
        pulse::jobs::JobSystem::endFrame();
    }

    pulse::jobs::JobSystem::shutdown();
}

// ── 4. Batch Job Throughput ──────────────────────────────────────────────────

void bench_batch_throughput() {
    pulse::jobs::JobSystem::init(4);

    constexpr int BatchSizes[] = { 16, 64, 256, 1024 };

    for (int batchSize : BatchSizes) {
        constexpr int Iters = 50;

        pulse::jobs::JobSystem::beginFrame();
        BENCH_START();
        for (int iter = 0; iter < Iters; ++iter) {
            std::atomic<int> sum{0};
            std::atomic<int>* sumPtr = &sum;
            pulse::jobs::AtomicCounter counter(batchSize);

            for (int i = 0; i < batchSize; ++i) {
                auto* job = pulse::jobs::JobSystem::createJobLambda(
                    [sumPtr](pulse::jobs::Job*) {
                        sumPtr->fetch_add(1, std::memory_order_relaxed);
                    }, &counter);
                pulse::jobs::JobSystem::submit(job);
            }
            pulse::jobs::JobSystem::waitForCounter(&counter);
        }

        char label[64];
        std::snprintf(label, sizeof(label), "Batch %4d jobs (x%d iters)", batchSize, Iters);
        BENCH_END(label, static_cast<double>(batchSize) * Iters);
        pulse::jobs::JobSystem::endFrame();

        // Reset allocators for next batch size.
        pulse::jobs::JobSystem::beginFrame();
        pulse::jobs::JobSystem::endFrame();
    }

    pulse::jobs::JobSystem::shutdown();
}

// ── 5. parallelFor Scaling ───────────────────────────────────────────────────

void bench_parallel_for_scaling() {
    constexpr uint32_t N = 100'000;
    constexpr int Iters = 20;

    // Allocate an array for work.
    float* data = static_cast<float*>(
#if defined(PULSE_COMPILER_MSVC)
        _aligned_malloc(N * sizeof(float), 64)
#else
        std::aligned_alloc(64, N * sizeof(float))
#endif
    );

    // ── Serial baseline ────────────────────────────────────────────────
    {
        BENCH_START();
        for (int iter = 0; iter < Iters; ++iter) {
            for (uint32_t i = 0; i < N; ++i) {
                data[i] = std::sinf(static_cast<float>(i) * 0.001f) +
                           std::cosf(static_cast<float>(i) * 0.002f);
            }
        }
        BENCH_END("Serial sin+cos 100K (x20 iters)", static_cast<double>(N) * Iters);
    }

    // ── Parallel with different thread counts ──────────────────────────
    uint32_t threadCounts[] = { 2, 4 };
    for (uint32_t threads : threadCounts) {
        pulse::jobs::JobSystem::init(threads - 1);  // -1 because main counts as a thread

        pulse::jobs::JobSystem::beginFrame();
        BENCH_START();
        for (int iter = 0; iter < Iters; ++iter) {
            float* dp = data;
            pulse::jobs::parallelFor(N, 1024u, [dp](uint32_t start, uint32_t end) {
                for (uint32_t i = start; i < end; ++i) {
                    dp[i] = std::sinf(static_cast<float>(i) * 0.001f) +
                             std::cosf(static_cast<float>(i) * 0.002f);
                }
            });
        }
        char label[64];
        std::snprintf(label, sizeof(label), "Parallel sin+cos 100K (%u threads, x20)", threads);
        BENCH_END(label, static_cast<double>(N) * Iters);
        pulse::jobs::JobSystem::endFrame();

        pulse::jobs::JobSystem::shutdown();
    }

    doNotOptimize(data[0]);

#if defined(PULSE_COMPILER_MSVC)
    _aligned_free(data);
#else
    std::free(data);
#endif
}

// ── 6. Work Stealing Efficiency ──────────────────────────────────────────────

void bench_work_stealing() {
    pulse::jobs::JobSystem::init(4);

    constexpr int N = 1024;
    constexpr int Iters = 20;

    // All jobs submitted from main thread — workers must steal.
    {
        pulse::jobs::JobSystem::beginFrame();
        BENCH_START();
        for (int iter = 0; iter < Iters; ++iter) {
            std::atomic<int> sum{0};
            std::atomic<int>* sumPtr = &sum;
            pulse::jobs::AtomicCounter counter(N);

            for (int i = 0; i < N; ++i) {
                auto* job = pulse::jobs::JobSystem::createJobLambda(
                    [sumPtr](pulse::jobs::Job*) {
                        // Simulate some work.
                        volatile int v = 0;
                        for (int k = 0; k < 100; ++k) v += k;
                        (void)v;
                        sumPtr->fetch_add(1, std::memory_order_relaxed);
                    }, &counter);
                pulse::jobs::JobSystem::submit(job);
            }
            pulse::jobs::JobSystem::waitForCounter(&counter);
        }
        BENCH_END("Steal-heavy 1024 jobs (x20 iters)", static_cast<double>(N) * Iters);
        pulse::jobs::JobSystem::endFrame();
    }

    pulse::jobs::JobSystem::shutdown();
}

// ── 7. Fork-Join Overhead ────────────────────────────────────────────────────

void bench_fork_join_overhead() {
    constexpr uint32_t N = 10'000;
    constexpr int Iters = 50;

    float data[N];

    // ── Serial: direct loop ────────────────────────────────────────────
    {
        BENCH_START();
        for (int iter = 0; iter < Iters; ++iter) {
            for (uint32_t i = 0; i < N; ++i) {
                data[i] = static_cast<float>(i) * 2.0f + 1.0f;
            }
        }
        BENCH_END_SIMPLE("Serial array fill 10K (x50 iters)");
    }
    doNotOptimize(data[0]);

    // ── Parallel: fork-join via parallelFor ─────────────────────────────
    pulse::jobs::JobSystem::init(4);
    {
        pulse::jobs::JobSystem::beginFrame();
        BENCH_START();
        for (int iter = 0; iter < Iters; ++iter) {
            float* dp = data;
            pulse::jobs::parallelFor(N, 512u, [dp](uint32_t start, uint32_t end) {
                for (uint32_t i = start; i < end; ++i) {
                    dp[i] = static_cast<float>(i) * 2.0f + 1.0f;
                }
            });
        }
        BENCH_END_SIMPLE("Parallel array fill 10K (x50 iters, 4 threads)");
        pulse::jobs::JobSystem::endFrame();
    }
    doNotOptimize(data[0]);

    pulse::jobs::JobSystem::shutdown();
}

// ── AtomicCounter Throughput ─────────────────────────────────────────────────

void bench_atomic_counter() {
    constexpr int N = 1'000'000;

    pulse::jobs::AtomicCounter counter(N);

    {
        BENCH_START();
        for (int i = 0; i < N; ++i) {
            counter.decrement();
        }
        BENCH_END("Counter decrement 1M (single-threaded)", static_cast<double>(N));
    }

    // Concurrent decrement.
    counter.reset(N);
    {
        constexpr int NumThreads = 4;
        int perThread = N / NumThreads;

        BENCH_START();
        std::thread threads[NumThreads];
        for (int t = 0; t < NumThreads; ++t) {
            threads[t] = std::thread([&counter, perThread]() {
                for (int i = 0; i < perThread; ++i) {
                    counter.decrement();
                }
            });
        }
        for (auto& t : threads) {
            t.join();
        }
        BENCH_END("Counter decrement 1M (4 threads)", static_cast<double>(N));
    }
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    std::printf("╔═══════════════════════════════════════════════════════════╗\n");
    std::printf("║          Pulse Job System — Benchmarks                   ║\n");
    std::printf("╚═══════════════════════════════════════════════════════════╝\n\n");

    std::printf("── Job Allocation ─────────────────────────────────────────\n");
    bench_job_allocation();

    std::printf("\n── WorkStealingQueue ───────────────────────────────────────\n");
    bench_wsq_throughput();

    std::printf("\n── AtomicCounter ───────────────────────────────────────────\n");
    bench_atomic_counter();

    std::printf("\n── Single Job Latency ──────────────────────────────────────\n");
    bench_single_job_latency();

    std::printf("\n── Batch Job Throughput ─────────────────────────────────────\n");
    bench_batch_throughput();

    std::printf("\n── parallelFor Scaling ─────────────────────────────────────\n");
    bench_parallel_for_scaling();

    std::printf("\n── Work Stealing Efficiency ────────────────────────────────\n");
    bench_work_stealing();

    std::printf("\n── Fork-Join Overhead ──────────────────────────────────────\n");
    bench_fork_join_overhead();

    std::printf("\n══════════════════════════════════════════════════════════════\n");
    std::printf("  Benchmarks complete.\n");
    std::printf("══════════════════════════════════════════════════════════════\n");

    return 0;
}
