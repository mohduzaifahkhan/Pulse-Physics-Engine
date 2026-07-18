/**
 * @file utilities_bench.cpp
 * @brief Performance benchmarks for the Pulse utilities module.
 *
 * Benchmarks:
 * 1. SoA vs AoS iteration (Vec3 position + velocity update)
 * 2. Handle pool allocate/free throughput
 * 3. FixedBitset operations (set/test/popcount)
 * 4. RingBuffer push/pop throughput
 * 5. FixedArray vs raw array push_back
 */

#define PULSE_DISABLE_ASSERTS 1
#define PULSE_DISABLE_PROFILING 1

#include <pulse/utilities/bitset.h>
#include <pulse/utilities/handle.h>
#include <pulse/utilities/handle_pool.h>
#include <pulse/utilities/fixed_array.h>
#include <pulse/utilities/soa_array.h>
#include <pulse/utilities/ring_buffer.h>
#include <pulse/utilities/profiler.h>
#include <pulse/math/vec3.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>

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
    std::printf("  %-40s %8.3f ms  (%12.0f ops/sec)\n", \
                label, bench_elapsed_, bench_opsPerSec_); \
} while(0)

// ── 1. SoA vs AoS Iteration ─────────────────────────────────────────────────

struct AoSBody {
    pulse::Vec3 position;
    pulse::Vec3 velocity;
    float mass;
    float padding;
};

void bench_soa_vs_aos() {
    constexpr int N = 100'000;
    constexpr int Iters = 100;
    float dt = 1.0f / 60.0f;

    // ── AoS ──────────────────────────────────────────────────────────────
    auto* aos = static_cast<AoSBody*>(
        _aligned_malloc(N * sizeof(AoSBody), 64));

    for (int i = 0; i < N; ++i) {
        float fi = static_cast<float>(i);
        aos[i].position = pulse::Vec3(fi * 0.1f, fi * 0.2f, fi * 0.3f);
        aos[i].velocity = pulse::Vec3(0.0f, -9.81f, 0.0f);
        aos[i].mass = 1.0f;
    }

    {
        BENCH_START();
        for (int iter = 0; iter < Iters; ++iter) {
            for (int i = 0; i < N; ++i) {
                aos[i].position = aos[i].position + aos[i].velocity * dt;
            }
        }
        BENCH_END("AoS position update (100K x 100)", static_cast<double>(N) * Iters);
    }
    _aligned_free(aos);

    // ── SoA ──────────────────────────────────────────────────────────────
    pulse::util::SoAArray<pulse::Vec3, pulse::Vec3, float> soa(N);
    for (int i = 0; i < N; ++i) {
        float fi = static_cast<float>(i);
        soa.add(
            pulse::Vec3(fi * 0.1f, fi * 0.2f, fi * 0.3f),
            pulse::Vec3(0.0f, -9.81f, 0.0f),
            1.0f
        );
    }

    {
        BENCH_START();
        pulse::Vec3* positions  = soa.getArray<0>();
        pulse::Vec3* velocities = soa.getArray<1>();
        for (int iter = 0; iter < Iters; ++iter) {
            for (int i = 0; i < N; ++i) {
                positions[i] = positions[i] + velocities[i] * dt;
            }
        }
        BENCH_END("SoA position update (100K x 100)", static_cast<double>(N) * Iters);
    }
}

// ── 2. Handle Pool Throughput ────────────────────────────────────────────────

void bench_handle_pool() {
    constexpr int N = 100'000;

    pulse::util::HandlePool<> pool(N);

    // Allocate all
    {
        BENCH_START();
        for (int i = 0; i < N; ++i) {
            pool.allocate();
        }
        BENCH_END("HandlePool allocate (100K)", N);
    }

    pool.reset();

    // Allocate + free cycle
    {
        BENCH_START();
        pulse::util::Handle<> handles[1000];
        for (int round = 0; round < N / 1000; ++round) {
            for (int i = 0; i < 1000; ++i) {
                handles[i] = pool.allocate();
            }
            for (int i = 0; i < 1000; ++i) {
                pool.free(handles[i]);
            }
        }
        BENCH_END("HandlePool alloc+free cycle (100K)", N);
    }
}

// ── 3. FixedBitset Operations ────────────────────────────────────────────────

void bench_bitset() {
    constexpr int Iters = 1'000'000;

    // Set/test throughput
    {
        pulse::util::FixedBitset<1024> bs;
        BENCH_START();
        for (int i = 0; i < Iters; ++i) {
            bs.set(i & 1023);
            volatile bool v = bs.test(i & 1023);
            (void)v;
        }
        BENCH_END("Bitset set+test (1M ops, 1024 bits)", Iters);
    }

    // Popcount throughput
    {
        pulse::util::FixedBitset<1024> bs;
        bs.setAll();
        BENCH_START();
        volatile int total = 0;
        for (int i = 0; i < Iters; ++i) {
            total += bs.countSet();
        }
        (void)total;
        BENCH_END("Bitset popcount (1M calls, 1024 bits)", Iters);
    }

    // forEachSet throughput
    {
        pulse::util::FixedBitset<256> bs;
        // Set every 4th bit.
        for (int i = 0; i < 256; i += 4) bs.set(i);

        BENCH_START();
        volatile int sum = 0;
        for (int iter = 0; iter < Iters / 10; ++iter) {
            bs.forEachSet([&](std::size_t idx) {
                sum += static_cast<int>(idx);
            });
        }
        (void)sum;
        BENCH_END("Bitset forEachSet (100K iters, 64 bits set)", Iters / 10);
    }
}

// ── 4. RingBuffer Throughput ─────────────────────────────────────────────────

void bench_ring_buffer() {
    constexpr int N = 1'000'000;

    pulse::util::RingBuffer<int, 1024> rb;

    // Sequential push/pop (same thread)
    {
        BENCH_START();
        for (int i = 0; i < N; ++i) {
            rb.tryPush(i);
            int val;
            rb.tryPop(val);
        }
        BENCH_END("RingBuffer push+pop cycle (1M)", N);
    }

    // Burst push then burst pop
    {
        BENCH_START();
        for (int round = 0; round < N / 512; ++round) {
            for (int i = 0; i < 512; ++i) {
                rb.tryPush(i);
            }
            int val;
            for (int i = 0; i < 512; ++i) {
                rb.tryPop(val);
            }
        }
        BENCH_END("RingBuffer burst 512 push+pop (1M)", N);
    }
}

// ── 5. FixedArray Throughput ─────────────────────────────────────────────────

void bench_fixed_array() {
    constexpr int N = 1'000'000;

    // push_back + pop_back cycle
    {
        pulse::util::FixedArray<int, 1024> arr;
        BENCH_START();
        for (int i = 0; i < N; ++i) {
            if (arr.full()) arr.clear();
            arr.push_back(i);
        }
        BENCH_END("FixedArray push_back (1M, cap 1024)", N);
    }

    // Iteration throughput
    {
        pulse::util::FixedArray<float, 1024> arr;
        for (int i = 0; i < 1024; ++i) {
            arr.push_back(static_cast<float>(i));
        }

        BENCH_START();
        volatile float sum = 0.0f;
        for (int iter = 0; iter < N / 1024; ++iter) {
            for (float val : arr) {
                sum += val;
            }
        }
        (void)sum;
        BENCH_END("FixedArray iteration (1M elements)", N);
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main() {
    std::printf("╔═══════════════════════════════════════════════════════════╗\n");
    std::printf("║        Pulse Utilities Module — Benchmarks               ║\n");
    std::printf("╚═══════════════════════════════════════════════════════════╝\n\n");

    std::printf("── SoA vs AoS Iteration ───────────────────────────────────\n");
    bench_soa_vs_aos();

    std::printf("\n── Handle Pool ─────────────────────────────────────────────\n");
    bench_handle_pool();

    std::printf("\n── FixedBitset ─────────────────────────────────────────────\n");
    bench_bitset();

    std::printf("\n── RingBuffer ──────────────────────────────────────────────\n");
    bench_ring_buffer();

    std::printf("\n── FixedArray ──────────────────────────────────────────────\n");
    bench_fixed_array();

    std::printf("\n══════════════════════════════════════════════════════════════\n");
    std::printf("  Benchmarks complete.\n");
    std::printf("══════════════════════════════════════════════════════════════\n");

    return 0;
}
