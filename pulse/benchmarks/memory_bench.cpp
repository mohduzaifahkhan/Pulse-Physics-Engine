/**
 * @file memory_bench.cpp
 * @brief Performance benchmarks for the Pulse memory allocator library.
 *
 * Measures throughput of critical allocator operations (allocate, deallocate,
 * reset) across all allocator types to verify zero-overhead design goals
 * and establish baseline performance numbers.
 */

#include <pulse/memory/allocator_base.h>
#include <pulse/memory/arena_allocator.h>
#include <pulse/memory/pool_allocator.h>
#include <pulse/memory/stack_allocator.h>
#include <pulse/memory/frame_allocator.h>
#include <pulse/memory/free_list_allocator.h>

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>

using namespace pulse::memory;

// Prevent compiler from optimizing away the result
template <typename T>
static void doNotOptimize(T const& val) {
    volatile auto sink = val;
    (void)sink;
}

struct BenchResult {
    const char* name;
    double opsPerSecond;
    double nsPerOp;
};

template <typename Func>
BenchResult benchmark(const char* name, int iterations, Func&& func) {
    // Warm up
    for (int i = 0; i < 1000; ++i) {
        func(i);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func(i);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();
    double opsPerSec = static_cast<double>(iterations) / elapsed;
    double nsPerOp = (elapsed * 1.0e9) / static_cast<double>(iterations);

    return {name, opsPerSec, nsPerOp};
}

void printResult(const BenchResult& r) {
    std::printf("  %-35s %12.0f ops/s  |  %.2f ns/op\n", r.name, r.opsPerSecond, r.nsPerOp);
}

int main() {
    constexpr int N = 10'000'000;

    std::printf("╔══════════════════════════════════════════════════════════════════╗\n");
    std::printf("║         PULSE Physics Engine - Memory Allocator Benchmarks      ║\n");
    std::printf("╠══════════════════════════════════════════════════════════════════╣\n");
    std::printf("║  Iterations per benchmark: %d                          ║\n", N);
    std::printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    // ── ArenaAllocator ───────────────────────────────────────────────────────
    std::printf("── ArenaAllocator ──────────────────────────────────────────────\n");

    {
        // Single allocation throughput (allocate + reset cycle)
        printResult(benchmark("Arena alloc (64B)", N, [](int) {
            static ArenaAllocator arena(1024 * 1024); // 1 MB
            void* p = arena.allocate(64);
            doNotOptimize(p);
            if (arena.remaining() < 64) arena.reset();
        }));

        printResult(benchmark("Arena alloc (256B)", N, [](int) {
            static ArenaAllocator arena(1024 * 1024);
            void* p = arena.allocate(256);
            doNotOptimize(p);
            if (arena.remaining() < 256) arena.reset();
        }));

        printResult(benchmark("Arena alloc (1KB)", N, [](int) {
            static ArenaAllocator arena(4 * 1024 * 1024);
            void* p = arena.allocate(1024);
            doNotOptimize(p);
            if (arena.remaining() < 1024) arena.reset();
        }));

        // Reset throughput
        printResult(benchmark("Arena reset", N, [](int) {
            static ArenaAllocator arena(65536);
            arena.allocate(64);
            arena.reset();
        }));

        // Batch: 100 allocs then reset
        printResult(benchmark("Arena 100-alloc + reset", N / 100, [](int) {
            static ArenaAllocator arena(1024 * 1024);
            for (int j = 0; j < 100; ++j) {
                void* p = arena.allocate(64);
                doNotOptimize(p);
            }
            arena.reset();
        }));
    }

    // ── malloc baseline ──────────────────────────────────────────────────────
    std::printf("\n── malloc baseline ─────────────────────────────────────────────\n");

    printResult(benchmark("malloc/free (64B)", N, [](int) {
        void* p = std::malloc(64);
        doNotOptimize(p);
        std::free(p);
    }));

    printResult(benchmark("malloc/free (256B)", N, [](int) {
        void* p = std::malloc(256);
        doNotOptimize(p);
        std::free(p);
    }));

    printResult(benchmark("malloc/free (1KB)", N, [](int) {
        void* p = std::malloc(1024);
        doNotOptimize(p);
        std::free(p);
    }));

    // ── PoolAllocator ────────────────────────────────────────────────────────
    std::printf("\n── PoolAllocator ───────────────────────────────────────────────\n");

    {
        printResult(benchmark("Pool alloc (64B blocks)", N, [](int) {
            static PoolAllocator<64> pool(100000);
            void* p = pool.allocate(64);
            doNotOptimize(p);
            if (pool.isFull()) pool.reset();
        }));

        // Alloc + dealloc cycle
        printResult(benchmark("Pool alloc+dealloc cycle", N, [](int) {
            static PoolAllocator<64> pool(1024);
            void* p = pool.allocate(64);
            doNotOptimize(p);
            pool.deallocate(p);
        }));

        printResult(benchmark("Pool reset (1000 blocks)", N / 100, [](int) {
            static PoolAllocator<64> pool(1000);
            for (int j = 0; j < 1000; ++j) {
                pool.allocate(64);
            }
            pool.reset();
        }));
    }

    // ── StackAllocator ───────────────────────────────────────────────────────
    std::printf("\n── StackAllocator ──────────────────────────────────────────────\n");

    {
        printResult(benchmark("Stack alloc (64B)", N, [](int) {
            static StackAllocator stack(4 * 1024 * 1024);
            void* p = stack.allocate(64);
            doNotOptimize(p);
            if (stack.remaining() < 256) stack.reset();
        }));

        // Push/pop (LIFO alloc + dealloc)
        printResult(benchmark("Stack push+pop (64B)", N, [](int) {
            static StackAllocator stack(65536);
            void* p = stack.allocate(64);
            doNotOptimize(p);
            stack.deallocate(p);
        }));

        // Mark/rollback (batch of 10)
        printResult(benchmark("Stack mark+10alloc+rollback", N / 10, [](int) {
            static StackAllocator stack(1024 * 1024);
            auto marker = stack.mark();
            for (int j = 0; j < 10; ++j) {
                void* p = stack.allocate(64);
                doNotOptimize(p);
            }
            stack.rollback(marker);
        }));

        printResult(benchmark("Stack reset", N, [](int) {
            static StackAllocator stack(65536);
            stack.allocate(64);
            stack.reset();
        }));
    }

    // ── FrameAllocator ───────────────────────────────────────────────────────
    std::printf("\n── FrameAllocator ──────────────────────────────────────────────\n");

    {
        // Full frame cycle: beginFrame + N allocs
        printResult(benchmark("Frame beginFrame+50alloc", N / 50, [](int) {
            static FrameAllocator frame(1024 * 1024);
            frame.beginFrame();
            for (int j = 0; j < 50; ++j) {
                void* p = frame.allocate(64);
                doNotOptimize(p);
            }
        }));

        printResult(benchmark("Frame alloc (64B)", N, [](int) {
            static FrameAllocator frame(4 * 1024 * 1024);
            static int count = 0;
            if (count % 10000 == 0) frame.beginFrame();
            count++;
            void* p = frame.allocate(64);
            doNotOptimize(p);
        }));

        printResult(benchmark("Frame beginFrame", N, [](int) {
            static FrameAllocator frame(65536);
            frame.beginFrame();
        }));
    }

    // ── FreeListAllocator ────────────────────────────────────────────────────
    std::printf("\n── FreeListAllocator ───────────────────────────────────────────\n");

    {
        // Alloc+dealloc cycle (first-fit)
        printResult(benchmark("FreeList alloc+dealloc (FF)", N, [](int) {
            static FreeListAllocator fl(1024 * 1024, FitStrategy::FirstFit);
            void* p = fl.allocate(64);
            doNotOptimize(p);
            fl.deallocate(p);
        }));

        // Alloc+dealloc cycle (best-fit)
        printResult(benchmark("FreeList alloc+dealloc (BF)", N, [](int) {
            static FreeListAllocator fl(1024 * 1024, FitStrategy::BestFit);
            void* p = fl.allocate(64);
            doNotOptimize(p);
            fl.deallocate(p);
        }));

        // Variable-size alloc+dealloc
        printResult(benchmark("FreeList var-size alloc+dealloc", N, [](int i) {
            static FreeListAllocator fl(4 * 1024 * 1024, FitStrategy::FirstFit);
            std::size_t sizes[] = {32, 64, 128, 256, 512};
            void* p = fl.allocate(sizes[i % 5]);
            doNotOptimize(p);
            fl.deallocate(p);
        }));

        // Mixed pattern: alloc 10, free 5, repeat
        printResult(benchmark("FreeList mixed alloc/free", N / 10, [](int) {
            static FreeListAllocator fl(4 * 1024 * 1024, FitStrategy::FirstFit);
            static void* ptrs[10];
            static int cycle = 0;

            for (int j = 0; j < 10; ++j) {
                ptrs[j] = fl.allocate(64);
                doNotOptimize(ptrs[j]);
            }
            for (int j = 0; j < 5; ++j) {
                fl.deallocate(ptrs[j]);
            }
            for (int j = 5; j < 10; ++j) {
                fl.deallocate(ptrs[j]);
            }

            cycle++;
            if (cycle % 1000 == 0) fl.reset();
        }));

        printResult(benchmark("FreeList reset", N, [](int) {
            static FreeListAllocator fl(65536, FitStrategy::FirstFit);
            fl.allocate(64);
            fl.reset();
        }));
    }

    std::printf("\n══════════════════════════════════════════════════════════════════\n");
    std::printf("  Done.\n");

    return 0;
}
