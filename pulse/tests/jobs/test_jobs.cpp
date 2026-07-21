/**
 * @file test_jobs.cpp
 * @brief Comprehensive unit tests for the Pulse job system module.
 *
 * Covers: AtomicCounter, WorkStealingQueue, JobAllocator, Job descriptor,
 * JobSystem lifecycle, job submission & execution, fork-join synchronization,
 * and parallelFor.
 *
 * Uses the same lightweight test framework as test_math.cpp, test_memory.cpp,
 * and test_utilities.cpp.
 */

// Force asserts ON for testing regardless of build mode.
#define PULSE_ENABLE_ASSERTS 1
#define PULSE_ENABLE_PROFILING 1

#include <pulse/jobs/counter.h>
#include <pulse/jobs/work_stealing_queue.h>
#include <pulse/jobs/job_allocator.h>
#include <pulse/jobs/job.h>
#include <pulse/jobs/worker_thread.h>
#include <pulse/jobs/job_system.h>
#include <pulse/jobs/parallel_for.h>

#include <pulse/math/math_common.h>
#include <pulse/utilities/assert.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <thread>
#include <atomic>
#include <vector>

// ── Minimal test framework ────────────────────────────────────────────────────

static int g_totalTests = 0;
static int g_passedTests = 0;
static int g_failedTests = 0;

#define TEST_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            std::printf("  FAIL: %s (line %d): %s\n", __func__, __LINE__, #expr); \
            return false; \
        } \
    } while(0)

#define RUN_TEST(func) \
    do { \
        g_totalTests++; \
        if (func()) { \
            g_passedTests++; \
        } else { \
            g_failedTests++; \
            std::printf("FAILED: %s\n", #func); \
        } \
    } while(0)

// =============================================================================
// 1. ATOMIC COUNTER TESTS
// =============================================================================

static bool test_counter_default_construction() {
    pulse::jobs::AtomicCounter counter;
    TEST_ASSERT(counter.load() == 0);
    TEST_ASSERT(counter.isComplete());
    return true;
}

static bool test_counter_explicit_construction() {
    pulse::jobs::AtomicCounter counter(5);
    TEST_ASSERT(counter.load() == 5);
    TEST_ASSERT(!counter.isComplete());
    return true;
}

static bool test_counter_reset() {
    pulse::jobs::AtomicCounter counter;
    counter.reset(10);
    TEST_ASSERT(counter.load() == 10);
    TEST_ASSERT(!counter.isComplete());

    counter.reset(0);
    TEST_ASSERT(counter.load() == 0);
    TEST_ASSERT(counter.isComplete());
    return true;
}

static bool test_counter_decrement() {
    pulse::jobs::AtomicCounter counter(3);

    int32_t prev = counter.decrement();
    TEST_ASSERT(prev == 3);  // Returns value BEFORE decrement.
    TEST_ASSERT(counter.load() == 2);

    prev = counter.decrement();
    TEST_ASSERT(prev == 2);
    TEST_ASSERT(counter.load() == 1);

    prev = counter.decrement();
    TEST_ASSERT(prev == 1);
    TEST_ASSERT(counter.load() == 0);
    TEST_ASSERT(counter.isComplete());
    return true;
}

static bool test_counter_increment() {
    pulse::jobs::AtomicCounter counter(0);
    TEST_ASSERT(counter.isComplete());

    counter.increment();
    TEST_ASSERT(counter.load() == 1);
    TEST_ASSERT(!counter.isComplete());

    counter.increment();
    TEST_ASSERT(counter.load() == 2);
    return true;
}

static bool test_counter_concurrent_decrement() {
    constexpr int N = 8;
    pulse::jobs::AtomicCounter counter(N);

    std::vector<std::thread> threads;
    threads.reserve(N);
    for (int i = 0; i < N; ++i) {
        threads.emplace_back([&counter]() {
            counter.decrement();
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    TEST_ASSERT(counter.load() == 0);
    TEST_ASSERT(counter.isComplete());
    return true;
}

static bool test_counter_cache_line_alignment() {
    TEST_ASSERT(alignof(pulse::jobs::AtomicCounter) >= PULSE_CACHE_LINE);
    return true;
}

// =============================================================================
// 2. WORK STEALING QUEUE TESTS
// =============================================================================

static bool test_wsq_empty_pop() {
    pulse::jobs::WorkStealingQueue<int*, 64> queue;
    int* item = nullptr;
    TEST_ASSERT(!queue.pop(item));
    TEST_ASSERT(queue.empty());
    TEST_ASSERT(queue.size() == 0);
    return true;
}

static bool test_wsq_empty_steal() {
    pulse::jobs::WorkStealingQueue<int*, 64> queue;
    int* item = nullptr;
    TEST_ASSERT(!queue.steal(item));
    return true;
}

static bool test_wsq_push_pop_single() {
    pulse::jobs::WorkStealingQueue<int*, 64> queue;
    int value = 42;
    TEST_ASSERT(queue.push(&value));
    TEST_ASSERT(queue.size() == 1);

    int* popped = nullptr;
    TEST_ASSERT(queue.pop(popped));
    TEST_ASSERT(popped == &value);
    TEST_ASSERT(queue.empty());
    return true;
}

static bool test_wsq_push_steal_single() {
    pulse::jobs::WorkStealingQueue<int*, 64> queue;
    int value = 42;
    TEST_ASSERT(queue.push(&value));

    int* stolen = nullptr;
    TEST_ASSERT(queue.steal(stolen));
    TEST_ASSERT(stolen == &value);
    TEST_ASSERT(queue.empty());
    return true;
}

static bool test_wsq_lifo_order() {
    // Pop returns in LIFO (Last-In-First-Out) order.
    pulse::jobs::WorkStealingQueue<int, 64> queue;
    for (int i = 0; i < 5; ++i) {
        TEST_ASSERT(queue.push(i));
    }

    for (int i = 4; i >= 0; --i) {
        int item = -1;
        TEST_ASSERT(queue.pop(item));
        TEST_ASSERT(item == i);
    }
    TEST_ASSERT(queue.empty());
    return true;
}

static bool test_wsq_fifo_order() {
    // Steal returns in FIFO (First-In-First-Out) order.
    pulse::jobs::WorkStealingQueue<int, 64> queue;
    for (int i = 0; i < 5; ++i) {
        TEST_ASSERT(queue.push(i));
    }

    for (int i = 0; i < 5; ++i) {
        int item = -1;
        TEST_ASSERT(queue.steal(item));
        TEST_ASSERT(item == i);
    }
    TEST_ASSERT(queue.empty());
    return true;
}

static bool test_wsq_full_capacity() {
    pulse::jobs::WorkStealingQueue<int, 16> queue;  // Capacity = 16
    for (int i = 0; i < 16; ++i) {
        TEST_ASSERT(queue.push(i));
    }
    TEST_ASSERT(queue.size() == 16);

    // Should fail when full.
    TEST_ASSERT(!queue.push(999));
    return true;
}

static bool test_wsq_size_and_empty() {
    pulse::jobs::WorkStealingQueue<int, 64> queue;
    TEST_ASSERT(queue.empty());
    TEST_ASSERT(queue.size() == 0);

    queue.push(1);
    TEST_ASSERT(!queue.empty());
    TEST_ASSERT(queue.size() == 1);

    queue.push(2);
    TEST_ASSERT(queue.size() == 2);

    int item;
    queue.pop(item);
    TEST_ASSERT(queue.size() == 1);

    queue.pop(item);
    TEST_ASSERT(queue.empty());
    return true;
}

static bool test_wsq_reset() {
    pulse::jobs::WorkStealingQueue<int, 64> queue;
    for (int i = 0; i < 10; ++i) {
        queue.push(i);
    }
    TEST_ASSERT(queue.size() == 10);

    queue.reset();
    TEST_ASSERT(queue.empty());
    TEST_ASSERT(queue.size() == 0);

    // Should be usable again after reset.
    TEST_ASSERT(queue.push(42));
    int item;
    TEST_ASSERT(queue.pop(item));
    TEST_ASSERT(item == 42);
    return true;
}

static bool test_wsq_concurrent_steal() {
    // One producer pushes, multiple stealers steal concurrently.
    pulse::jobs::WorkStealingQueue<int, 4096> queue;
    constexpr int N = 1000;

    for (int i = 0; i < N; ++i) {
        queue.push(i);
    }

    std::atomic<int> totalStolen{0};
    constexpr int NumStealers = 4;
    std::vector<std::thread> stealers;
    stealers.reserve(NumStealers);

    for (int t = 0; t < NumStealers; ++t) {
        stealers.emplace_back([&queue, &totalStolen]() {
            int item;
            while (queue.steal(item)) {
                totalStolen.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& t : stealers) {
        t.join();
    }

    TEST_ASSERT(totalStolen.load() == N);
    TEST_ASSERT(queue.empty());
    return true;
}

// =============================================================================
// 3. JOB ALLOCATOR TESTS
// =============================================================================

static bool test_job_allocator_single_alloc() {
    pulse::jobs::JobAllocator<64> allocator;
    pulse::jobs::Job* job = allocator.allocate();
    TEST_ASSERT(job != nullptr);
    TEST_ASSERT(allocator.allocatedCount() == 1);
    TEST_ASSERT(allocator.owns(job));

    // Verify zeroed.
    TEST_ASSERT(job->function == nullptr);
    TEST_ASSERT(job->counter == nullptr);
    TEST_ASSERT(job->priority == pulse::jobs::JobPriority::Normal);
    return true;
}

static bool test_job_allocator_all_unique() {
    constexpr std::size_t Cap = 128;
    pulse::jobs::JobAllocator<Cap> allocator;
    pulse::jobs::Job* jobs[Cap];

    for (std::size_t i = 0; i < Cap; ++i) {
        jobs[i] = allocator.allocate();
        TEST_ASSERT(jobs[i] != nullptr);
    }

    // Verify all pointers are unique.
    for (std::size_t i = 0; i < Cap; ++i) {
        for (std::size_t j = i + 1; j < Cap; ++j) {
            TEST_ASSERT(jobs[i] != jobs[j]);
        }
    }

    TEST_ASSERT(allocator.allocatedCount() == Cap);
    return true;
}

static bool test_job_allocator_free() {
    pulse::jobs::JobAllocator<64> allocator;
    pulse::jobs::Job* job1 = allocator.allocate();
    pulse::jobs::Job* job2 = allocator.allocate();
    TEST_ASSERT(allocator.allocatedCount() == 2);

    allocator.free(job1);
    TEST_ASSERT(allocator.allocatedCount() == 1);

    // Should be able to re-allocate.
    pulse::jobs::Job* job3 = allocator.allocate();
    TEST_ASSERT(job3 != nullptr);
    TEST_ASSERT(allocator.allocatedCount() == 2);

    allocator.free(job2);
    allocator.free(job3);
    TEST_ASSERT(allocator.allocatedCount() == 0);
    return true;
}

static bool test_job_allocator_reset() {
    constexpr std::size_t Cap = 32;
    pulse::jobs::JobAllocator<Cap> allocator;

    // Exhaust the pool.
    for (std::size_t i = 0; i < Cap; ++i) {
        allocator.allocate();
    }
    TEST_ASSERT(allocator.allocatedCount() == Cap);

    // Reset makes all slots available again.
    allocator.reset();
    TEST_ASSERT(allocator.allocatedCount() == 0);

    // Can allocate again.
    pulse::jobs::Job* job = allocator.allocate();
    TEST_ASSERT(job != nullptr);
    TEST_ASSERT(allocator.allocatedCount() == 1);
    return true;
}

static bool test_job_allocator_owns() {
    pulse::jobs::JobAllocator<32> allocator;
    pulse::jobs::Job* job = allocator.allocate();
    TEST_ASSERT(allocator.owns(job));

    // A random stack pointer should not be owned.
    pulse::jobs::Job stackJob{};
    TEST_ASSERT(!allocator.owns(&stackJob));
    return true;
}

static bool test_job_allocator_capacity() {
    pulse::jobs::JobAllocator<256> allocator;
    TEST_ASSERT(allocator.capacity() == 256);
    return true;
}

// =============================================================================
// 4. JOB DESCRIPTOR TESTS
// =============================================================================

static bool test_job_set_get_data() {
    pulse::jobs::Job job{};

    struct TestData {
        int a;
        float b;
        uint64_t c;
    };

    TestData input{42, 3.14f, 0xDEADBEEFull};
    job.setData(input);

    const auto& output = job.getData<TestData>();
    TEST_ASSERT(output.a == 42);
    TEST_ASSERT(output.b == 3.14f);
    TEST_ASSERT(output.c == 0xDEADBEEFull);
    return true;
}

static bool test_job_init_function() {
    static int callCount = 0;
    callCount = 0;

    pulse::jobs::Job job{};
    pulse::jobs::AtomicCounter counter(1);

    pulse::jobs::initJob(&job, [](pulse::jobs::Job*) {
        callCount++;
    }, &counter, pulse::jobs::JobPriority::High);

    TEST_ASSERT(job.function != nullptr);
    TEST_ASSERT(job.counter == &counter);
    TEST_ASSERT(job.priority == pulse::jobs::JobPriority::High);

    // Execute manually.
    job.function(&job);
    TEST_ASSERT(callCount == 1);
    return true;
}

static bool test_job_init_lambda() {
    pulse::jobs::Job job{};
    int result = 0;
    int* resultPtr = &result;

    pulse::jobs::initJobLambda(&job, [resultPtr](pulse::jobs::Job*) {
        *resultPtr = 42;
    });

    TEST_ASSERT(job.function != nullptr);

    job.function(&job);
    TEST_ASSERT(result == 42);
    return true;
}

static bool test_job_lambda_with_data() {
    pulse::jobs::Job job{};

    struct CaptureData {
        int* target;
        int value;
    };
    static_assert(sizeof(CaptureData) <= pulse::jobs::Job::InlineCapacity,
                  "CaptureData should fit in inline storage");

    int result = 0;
    CaptureData capture{&result, 123};

    job.setData(capture);
    job.function = [](pulse::jobs::Job* self) {
        auto& data = self->getData<CaptureData>();
        *data.target = data.value;
    };

    job.function(&job);
    TEST_ASSERT(result == 123);
    return true;
}

static bool test_job_size_and_alignment() {
    TEST_ASSERT(sizeof(pulse::jobs::Job) <= 128);
    TEST_ASSERT(alignof(pulse::jobs::Job) >= PULSE_CACHE_LINE);
    return true;
}

static bool test_job_priority_values() {
    TEST_ASSERT(static_cast<uint8_t>(pulse::jobs::JobPriority::High) == 0);
    TEST_ASSERT(static_cast<uint8_t>(pulse::jobs::JobPriority::Normal) == 1);
    TEST_ASSERT(static_cast<uint8_t>(pulse::jobs::JobPriority::Low) == 2);
    return true;
}

static bool test_job_inline_capacity() {
    TEST_ASSERT(pulse::jobs::Job::InlineCapacity == 48);

    // Verify we can store 6 pointers.
    static_assert(sizeof(void*) * 6 <= pulse::jobs::Job::InlineCapacity,
                  "Inline storage should hold 6 pointers");
    return true;
}

// =============================================================================
// 5. JOB SYSTEM LIFECYCLE TESTS
// =============================================================================

static bool test_jobsystem_init_shutdown() {
    TEST_ASSERT(!pulse::jobs::JobSystem::isInitialized());

    pulse::jobs::JobSystem::init(2);
    TEST_ASSERT(pulse::jobs::JobSystem::isInitialized());
    TEST_ASSERT(pulse::jobs::JobSystem::workerCount() == 2);
    TEST_ASSERT(pulse::jobs::JobSystem::threadCount() == 3);  // main + 2 workers

    pulse::jobs::JobSystem::shutdown();
    TEST_ASSERT(!pulse::jobs::JobSystem::isInitialized());
    return true;
}

static bool test_jobsystem_main_thread_index() {
    pulse::jobs::JobSystem::init(2);

    // Main thread should be index 0.
    TEST_ASSERT(pulse::jobs::JobSystem::currentThreadIndex() == 0);

    pulse::jobs::JobSystem::shutdown();
    return true;
}

static bool test_jobsystem_worker_context() {
    pulse::jobs::JobSystem::init(2);

    auto* ctx = pulse::jobs::JobSystem::getWorkerContext(0);
    TEST_ASSERT(ctx != nullptr);
    TEST_ASSERT(ctx->threadIndex == 0);

    auto* ctx1 = pulse::jobs::JobSystem::getWorkerContext(1);
    TEST_ASSERT(ctx1 != nullptr);
    TEST_ASSERT(ctx1->threadIndex == 1);

    pulse::jobs::JobSystem::shutdown();
    return true;
}

static bool test_jobsystem_begin_end_frame() {
    pulse::jobs::JobSystem::init(2);

    // Should not crash.
    pulse::jobs::JobSystem::beginFrame();
    pulse::jobs::JobSystem::endFrame();

    pulse::jobs::JobSystem::shutdown();
    return true;
}

static bool test_jobsystem_create_job() {
    pulse::jobs::JobSystem::init(2);
    pulse::jobs::JobSystem::beginFrame();

    auto* job = pulse::jobs::JobSystem::createJob();
    TEST_ASSERT(job != nullptr);
    TEST_ASSERT(job->function == nullptr);  // Zeroed.

    pulse::jobs::JobSystem::endFrame();
    pulse::jobs::JobSystem::shutdown();
    return true;
}

static bool test_jobsystem_create_job_with_fn() {
    pulse::jobs::JobSystem::init(2);
    pulse::jobs::JobSystem::beginFrame();

    static bool executed = false;
    executed = false;

    auto* job = pulse::jobs::JobSystem::createJob(
        [](pulse::jobs::Job*) { executed = true; });
    TEST_ASSERT(job != nullptr);
    TEST_ASSERT(job->function != nullptr);

    // Execute manually to verify.
    job->function(job);
    TEST_ASSERT(executed);

    pulse::jobs::JobSystem::endFrame();
    pulse::jobs::JobSystem::shutdown();
    return true;
}

static bool test_jobsystem_create_job_lambda() {
    pulse::jobs::JobSystem::init(2);
    pulse::jobs::JobSystem::beginFrame();

    int result = 0;
    int* ptr = &result;

    auto* job = pulse::jobs::JobSystem::createJobLambda(
        [ptr](pulse::jobs::Job*) { *ptr = 99; });
    TEST_ASSERT(job != nullptr);

    job->function(job);
    TEST_ASSERT(result == 99);

    pulse::jobs::JobSystem::endFrame();
    pulse::jobs::JobSystem::shutdown();
    return true;
}

// =============================================================================
// 6. JOB SUBMISSION & EXECUTION TESTS
// =============================================================================

static bool test_submit_single_job() {
    pulse::jobs::JobSystem::init(2);
    pulse::jobs::JobSystem::beginFrame();

    std::atomic<int> result{0};
    std::atomic<int>* ptr = &result;
    pulse::jobs::AtomicCounter counter(1);

    auto* job = pulse::jobs::JobSystem::createJobLambda(
        [ptr](pulse::jobs::Job*) {
            ptr->store(42, std::memory_order_relaxed);
        }, &counter);

    pulse::jobs::JobSystem::submit(job);
    pulse::jobs::JobSystem::waitForCounter(&counter);

    TEST_ASSERT(result.load() == 42);
    TEST_ASSERT(counter.isComplete());

    pulse::jobs::JobSystem::endFrame();
    pulse::jobs::JobSystem::shutdown();
    return true;
}

static bool test_submit_multiple_jobs() {
    pulse::jobs::JobSystem::init(4);
    pulse::jobs::JobSystem::beginFrame();

    constexpr int N = 64;
    std::atomic<int> sum{0};
    std::atomic<int>* sumPtr = &sum;
    pulse::jobs::AtomicCounter counter(N);

    for (int i = 0; i < N; ++i) {
        auto* job = pulse::jobs::JobSystem::createJobLambda(
            [sumPtr](pulse::jobs::Job*) {
                sumPtr->fetch_add(1, std::memory_order_relaxed);
            }, &counter);
        pulse::jobs::JobSystem::submit(job);
    }

    pulse::jobs::JobSystem::waitForCounter(&counter);

    TEST_ASSERT(sum.load() == N);
    TEST_ASSERT(counter.isComplete());

    pulse::jobs::JobSystem::endFrame();
    pulse::jobs::JobSystem::shutdown();
    return true;
}

static bool test_submit_job_with_inline_data() {
    pulse::jobs::JobSystem::init(2);
    pulse::jobs::JobSystem::beginFrame();

    struct JobData {
        std::atomic<int>* target;
        int value;
    };

    std::atomic<int> result{0};
    pulse::jobs::AtomicCounter counter(1);

    auto* job = pulse::jobs::JobSystem::createJob();
    auto& data = job->getData<JobData>();
    data.target = &result;
    data.value = 77;

    job->function = [](pulse::jobs::Job* self) {
        auto& d = self->getData<JobData>();
        d.target->store(d.value, std::memory_order_relaxed);
    };
    job->counter = &counter;

    pulse::jobs::JobSystem::submit(job);
    pulse::jobs::JobSystem::waitForCounter(&counter);

    TEST_ASSERT(result.load() == 77);

    pulse::jobs::JobSystem::endFrame();
    pulse::jobs::JobSystem::shutdown();
    return true;
}

static bool test_submit_batch() {
    pulse::jobs::JobSystem::init(4);
    pulse::jobs::JobSystem::beginFrame();

    constexpr int N = 32;
    std::atomic<int> count{0};
    std::atomic<int>* countPtr = &count;
    pulse::jobs::AtomicCounter counter(N);

    pulse::jobs::Job* jobs[N];
    for (int i = 0; i < N; ++i) {
        jobs[i] = pulse::jobs::JobSystem::createJobLambda(
            [countPtr](pulse::jobs::Job*) {
                countPtr->fetch_add(1, std::memory_order_relaxed);
            }, &counter);
    }

    pulse::jobs::JobSystem::submitBatch(jobs, N);
    pulse::jobs::JobSystem::waitForCounter(&counter);

    TEST_ASSERT(count.load() == N);

    pulse::jobs::JobSystem::endFrame();
    pulse::jobs::JobSystem::shutdown();
    return true;
}

static bool test_submit_with_priorities() {
    pulse::jobs::JobSystem::init(2);
    pulse::jobs::JobSystem::beginFrame();

    std::atomic<int> executed{0};
    std::atomic<int>* ptr = &executed;
    pulse::jobs::AtomicCounter counter(3);

    auto* high = pulse::jobs::JobSystem::createJobLambda(
        [ptr](pulse::jobs::Job*) { ptr->fetch_add(1, std::memory_order_relaxed); },
        &counter, pulse::jobs::JobPriority::High);
    auto* normal = pulse::jobs::JobSystem::createJobLambda(
        [ptr](pulse::jobs::Job*) { ptr->fetch_add(1, std::memory_order_relaxed); },
        &counter, pulse::jobs::JobPriority::Normal);
    auto* low = pulse::jobs::JobSystem::createJobLambda(
        [ptr](pulse::jobs::Job*) { ptr->fetch_add(1, std::memory_order_relaxed); },
        &counter, pulse::jobs::JobPriority::Low);

    pulse::jobs::JobSystem::submit(high);
    pulse::jobs::JobSystem::submit(normal);
    pulse::jobs::JobSystem::submit(low);
    pulse::jobs::JobSystem::waitForCounter(&counter);

    // All three should have executed regardless of priority.
    TEST_ASSERT(executed.load() == 3);

    pulse::jobs::JobSystem::endFrame();
    pulse::jobs::JobSystem::shutdown();
    return true;
}

// =============================================================================
// 7. FORK-JOIN & SYNCHRONIZATION TESTS
// =============================================================================

static bool test_fork_join_basic() {
    pulse::jobs::JobSystem::init(4);
    pulse::jobs::JobSystem::beginFrame();

    constexpr int N = 8;
    std::atomic<int> completed{0};
    std::atomic<int>* ptr = &completed;
    pulse::jobs::AtomicCounter counter(N);

    for (int i = 0; i < N; ++i) {
        auto* job = pulse::jobs::JobSystem::createJobLambda(
            [ptr](pulse::jobs::Job*) {
                // Simulate a bit of work.
                volatile int sum = 0;
                for (int k = 0; k < 1000; ++k) sum += k;
                (void)sum;
                ptr->fetch_add(1, std::memory_order_relaxed);
            }, &counter);
        pulse::jobs::JobSystem::submit(job);
    }

    pulse::jobs::JobSystem::waitForCounter(&counter);
    TEST_ASSERT(completed.load() == N);

    pulse::jobs::JobSystem::endFrame();
    pulse::jobs::JobSystem::shutdown();
    return true;
}

static bool test_fork_join_large_batch() {
    pulse::jobs::JobSystem::init(4);
    pulse::jobs::JobSystem::beginFrame();

    constexpr int N = 512;
    std::atomic<int> sum{0};
    std::atomic<int>* sumPtr = &sum;
    pulse::jobs::AtomicCounter counter(N);

    for (int i = 0; i < N; ++i) {
        auto* job = pulse::jobs::JobSystem::createJobLambda(
            [sumPtr](pulse::jobs::Job*) {
                sumPtr->fetch_add(1, std::memory_order_relaxed);
            }, &counter);
        pulse::jobs::JobSystem::submit(job);
    }

    pulse::jobs::JobSystem::waitForCounter(&counter);
    TEST_ASSERT(sum.load() == N);

    pulse::jobs::JobSystem::endFrame();
    pulse::jobs::JobSystem::shutdown();
    return true;
}

static bool test_multiple_counters() {
    pulse::jobs::JobSystem::init(4);
    pulse::jobs::JobSystem::beginFrame();

    // Two independent groups.
    constexpr int GroupA = 16;
    constexpr int GroupB = 16;

    std::atomic<int> sumA{0};
    std::atomic<int> sumB{0};
    std::atomic<int>* ptrA = &sumA;
    std::atomic<int>* ptrB = &sumB;

    pulse::jobs::AtomicCounter counterA(GroupA);
    pulse::jobs::AtomicCounter counterB(GroupB);

    for (int i = 0; i < GroupA; ++i) {
        auto* job = pulse::jobs::JobSystem::createJobLambda(
            [ptrA](pulse::jobs::Job*) {
                ptrA->fetch_add(1, std::memory_order_relaxed);
            }, &counterA);
        pulse::jobs::JobSystem::submit(job);
    }

    for (int i = 0; i < GroupB; ++i) {
        auto* job = pulse::jobs::JobSystem::createJobLambda(
            [ptrB](pulse::jobs::Job*) {
                ptrB->fetch_add(1, std::memory_order_relaxed);
            }, &counterB);
        pulse::jobs::JobSystem::submit(job);
    }

    pulse::jobs::JobSystem::waitForCounter(&counterA);
    TEST_ASSERT(sumA.load() == GroupA);

    pulse::jobs::JobSystem::waitForCounter(&counterB);
    TEST_ASSERT(sumB.load() == GroupB);

    pulse::jobs::JobSystem::endFrame();
    pulse::jobs::JobSystem::shutdown();
    return true;
}

static bool test_sequential_fork_joins() {
    pulse::jobs::JobSystem::init(4);
    pulse::jobs::JobSystem::beginFrame();

    // Multiple sequential fork-join waves.
    for (int wave = 0; wave < 4; ++wave) {
        constexpr int N = 32;
        std::atomic<int> count{0};
        std::atomic<int>* ptr = &count;
        pulse::jobs::AtomicCounter counter(N);

        for (int i = 0; i < N; ++i) {
            auto* job = pulse::jobs::JobSystem::createJobLambda(
                [ptr](pulse::jobs::Job*) {
                    ptr->fetch_add(1, std::memory_order_relaxed);
                }, &counter);
            pulse::jobs::JobSystem::submit(job);
        }

        pulse::jobs::JobSystem::waitForCounter(&counter);
        TEST_ASSERT(count.load() == N);
    }

    pulse::jobs::JobSystem::endFrame();
    pulse::jobs::JobSystem::shutdown();
    return true;
}

static bool test_work_stealing_under_imbalance() {
    // Submit all jobs from main thread — workers should steal them.
    pulse::jobs::JobSystem::init(4);
    pulse::jobs::JobSystem::beginFrame();

    constexpr int N = 256;
    std::atomic<int> total{0};
    std::atomic<int>* ptr = &total;
    pulse::jobs::AtomicCounter counter(N);

    // All jobs go into the main thread's queue.
    for (int i = 0; i < N; ++i) {
        auto* job = pulse::jobs::JobSystem::createJobLambda(
            [ptr](pulse::jobs::Job*) {
                // Simulate work.
                volatile int v = 0;
                for (int k = 0; k < 500; ++k) v += k;
                (void)v;
                ptr->fetch_add(1, std::memory_order_relaxed);
            }, &counter);
        pulse::jobs::JobSystem::submit(job);
    }

    pulse::jobs::JobSystem::waitForCounter(&counter);
    TEST_ASSERT(total.load() == N);

    pulse::jobs::JobSystem::endFrame();
    pulse::jobs::JobSystem::shutdown();
    return true;
}

// =============================================================================
// 8. PARALLEL FOR TESTS
// =============================================================================

static bool test_parallel_for_empty() {
    pulse::jobs::JobSystem::init(2);
    pulse::jobs::JobSystem::beginFrame();

    int callCount = 0;
    // count = 0 should be a no-op.
    pulse::jobs::parallelFor(0u, 16u, [&callCount](uint32_t, uint32_t) {
        callCount++;
    });

    TEST_ASSERT(callCount == 0);

    pulse::jobs::JobSystem::endFrame();
    pulse::jobs::JobSystem::shutdown();
    return true;
}

static bool test_parallel_for_single_element() {
    pulse::jobs::JobSystem::init(2);
    pulse::jobs::JobSystem::beginFrame();

    int result = -1;
    pulse::jobs::parallelFor(1u, 16u, [&result](uint32_t start, uint32_t end) {
        result = static_cast<int>(end - start);
    });

    TEST_ASSERT(result == 1);

    pulse::jobs::JobSystem::endFrame();
    pulse::jobs::JobSystem::shutdown();
    return true;
}

static bool test_parallel_for_all_indices() {
    pulse::jobs::JobSystem::init(4);
    pulse::jobs::JobSystem::beginFrame();

    constexpr uint32_t N = 1000;
    std::atomic<int> flags[N];
    for (uint32_t i = 0; i < N; ++i) {
        flags[i].store(0, std::memory_order_relaxed);
    }

    std::atomic<int>* flagsPtr = flags;
    pulse::jobs::parallelFor(N, 64u, [flagsPtr](uint32_t start, uint32_t end) {
        for (uint32_t i = start; i < end; ++i) {
            flagsPtr[i].fetch_add(1, std::memory_order_relaxed);
        }
    });

    // Every index should have been processed exactly once.
    for (uint32_t i = 0; i < N; ++i) {
        TEST_ASSERT(flags[i].load() == 1);
    }

    pulse::jobs::JobSystem::endFrame();
    pulse::jobs::JobSystem::shutdown();
    return true;
}

static bool test_parallel_for_auto_batch() {
    pulse::jobs::JobSystem::init(4);
    pulse::jobs::JobSystem::beginFrame();

    constexpr uint32_t N = 500;
    std::atomic<int> flags[N];
    for (uint32_t i = 0; i < N; ++i) {
        flags[i].store(0, std::memory_order_relaxed);
    }

    std::atomic<int>* flagsPtr = flags;
    pulse::jobs::parallelFor(N, [flagsPtr](uint32_t start, uint32_t end) {
        for (uint32_t i = start; i < end; ++i) {
            flagsPtr[i].fetch_add(1, std::memory_order_relaxed);
        }
    });

    for (uint32_t i = 0; i < N; ++i) {
        TEST_ASSERT(flags[i].load() == 1);
    }

    pulse::jobs::JobSystem::endFrame();
    pulse::jobs::JobSystem::shutdown();
    return true;
}

static bool test_parallel_for_data_correctness() {
    // Fill an array in parallel and verify every element.
    pulse::jobs::JobSystem::init(4);
    pulse::jobs::JobSystem::beginFrame();

    constexpr uint32_t N = 2048;
    float data[N];
    std::memset(data, 0, sizeof(data));

    float* dataPtr = data;
    pulse::jobs::parallelFor(N, 128u, [dataPtr](uint32_t start, uint32_t end) {
        for (uint32_t i = start; i < end; ++i) {
            dataPtr[i] = static_cast<float>(i) * 2.0f + 1.0f;
        }
    });

    for (uint32_t i = 0; i < N; ++i) {
        float expected = static_cast<float>(i) * 2.0f + 1.0f;
        TEST_ASSERT(std::fabs(data[i] - expected) < 1e-6f);
    }

    pulse::jobs::JobSystem::endFrame();
    pulse::jobs::JobSystem::shutdown();
    return true;
}

static bool test_parallel_for_small_batch_size() {
    pulse::jobs::JobSystem::init(4);
    pulse::jobs::JobSystem::beginFrame();

    constexpr uint32_t N = 100;
    std::atomic<int> sum{0};
    std::atomic<int>* sumPtr = &sum;

    pulse::jobs::parallelFor(N, 1u, [sumPtr](uint32_t start, uint32_t end) {
        for (uint32_t i = start; i < end; ++i) {
            sumPtr->fetch_add(static_cast<int>(i), std::memory_order_relaxed);
        }
    });

    // Sum of 0..99 = 4950
    TEST_ASSERT(sum.load() == 4950);

    pulse::jobs::JobSystem::endFrame();
    pulse::jobs::JobSystem::shutdown();
    return true;
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    std::printf("╔═══════════════════════════════════════════════════════════╗\n");
    std::printf("║          Pulse Job System — Unit Tests                   ║\n");
    std::printf("╚═══════════════════════════════════════════════════════════╝\n\n");

    // ── AtomicCounter tests ─────────────────────────────────────────────
    std::printf("── AtomicCounter ──────────────────────────────────\n");
    RUN_TEST(test_counter_default_construction);
    RUN_TEST(test_counter_explicit_construction);
    RUN_TEST(test_counter_reset);
    RUN_TEST(test_counter_decrement);
    RUN_TEST(test_counter_increment);
    RUN_TEST(test_counter_concurrent_decrement);
    RUN_TEST(test_counter_cache_line_alignment);

    // ── WorkStealingQueue tests ─────────────────────────────────────────
    std::printf("\n── WorkStealingQueue ───────────────────────────────\n");
    RUN_TEST(test_wsq_empty_pop);
    RUN_TEST(test_wsq_empty_steal);
    RUN_TEST(test_wsq_push_pop_single);
    RUN_TEST(test_wsq_push_steal_single);
    RUN_TEST(test_wsq_lifo_order);
    RUN_TEST(test_wsq_fifo_order);
    RUN_TEST(test_wsq_full_capacity);
    RUN_TEST(test_wsq_size_and_empty);
    RUN_TEST(test_wsq_reset);
    RUN_TEST(test_wsq_concurrent_steal);

    // ── JobAllocator tests ──────────────────────────────────────────────
    std::printf("\n── JobAllocator ────────────────────────────────────\n");
    RUN_TEST(test_job_allocator_single_alloc);
    RUN_TEST(test_job_allocator_all_unique);
    RUN_TEST(test_job_allocator_free);
    RUN_TEST(test_job_allocator_reset);
    RUN_TEST(test_job_allocator_owns);
    RUN_TEST(test_job_allocator_capacity);

    // ── Job descriptor tests ────────────────────────────────────────────
    std::printf("\n── Job Descriptor ──────────────────────────────────\n");
    RUN_TEST(test_job_set_get_data);
    RUN_TEST(test_job_init_function);
    RUN_TEST(test_job_init_lambda);
    RUN_TEST(test_job_lambda_with_data);
    RUN_TEST(test_job_size_and_alignment);
    RUN_TEST(test_job_priority_values);
    RUN_TEST(test_job_inline_capacity);

    // ── JobSystem lifecycle tests ───────────────────────────────────────
    std::printf("\n── JobSystem Lifecycle ─────────────────────────────\n");
    RUN_TEST(test_jobsystem_init_shutdown);
    RUN_TEST(test_jobsystem_main_thread_index);
    RUN_TEST(test_jobsystem_worker_context);
    RUN_TEST(test_jobsystem_begin_end_frame);
    RUN_TEST(test_jobsystem_create_job);
    RUN_TEST(test_jobsystem_create_job_with_fn);
    RUN_TEST(test_jobsystem_create_job_lambda);

    // ── Job submission & execution tests ────────────────────────────────
    std::printf("\n── Job Submission & Execution ──────────────────────\n");
    RUN_TEST(test_submit_single_job);
    RUN_TEST(test_submit_multiple_jobs);
    RUN_TEST(test_submit_job_with_inline_data);
    RUN_TEST(test_submit_batch);
    RUN_TEST(test_submit_with_priorities);

    // ── Fork-join & synchronization tests ───────────────────────────────
    std::printf("\n── Fork-Join & Synchronization ─────────────────────\n");
    RUN_TEST(test_fork_join_basic);
    RUN_TEST(test_fork_join_large_batch);
    RUN_TEST(test_multiple_counters);
    RUN_TEST(test_sequential_fork_joins);
    RUN_TEST(test_work_stealing_under_imbalance);

    // ── parallelFor tests ───────────────────────────────────────────────
    std::printf("\n── parallelFor ─────────────────────────────────────\n");
    RUN_TEST(test_parallel_for_empty);
    RUN_TEST(test_parallel_for_single_element);
    RUN_TEST(test_parallel_for_all_indices);
    RUN_TEST(test_parallel_for_auto_batch);
    RUN_TEST(test_parallel_for_data_correctness);
    RUN_TEST(test_parallel_for_small_batch_size);

    // ── Summary ─────────────────────────────────────────────────────────
    std::printf("\n════════════════════════════════════════════════════\n");
    std::printf("  Total: %d  |  Passed: %d  |  Failed: %d\n",
                g_totalTests, g_passedTests, g_failedTests);
    std::printf("════════════════════════════════════════════════════\n");

    return g_failedTests > 0 ? 1 : 0;
}
