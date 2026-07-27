/**
 * @file rigidbody_bench.cpp
 * @brief Performance benchmarks for the Pulse rigid body module (Module 11).
 *
 * Measures throughput for:
 *  1. Body creation — 10K bodies
 *  2. Body destruction — 10K swap-and-pop removals
 *  3. World inertia update — batch recompute for 10K bodies
 *  4. SolverBody conversion — to/from for 10K bodies
 *  5. Island building — 10K bodies with varying connectivity
 *  6. Sleep update pass — 10K bodies
 */

#include <pulse/rigidbody/rigid_body_common.h>
#include <pulse/rigidbody/rigid_body.h>
#include <pulse/rigidbody/body_manager.h>
#include <pulse/rigidbody/island_manager.h>
#include <pulse/rigidbody/sleep_manager.h>

#include <pulse/solver/solver_common.h>
#include <pulse/math/math_common.h>
#include <pulse/math/vec3.h>

#include <cstdio>
#include <cstdint>
#include <chrono>
#include <vector>

using namespace pulse;

// ── Timing utilities ─────────────────────────────────────────────────────────

using Clock = std::chrono::high_resolution_clock;

struct BenchResult {
    double totalMs;
    double opsPerSec;
    uint64_t ops;
};

static BenchResult benchLoop(const char* name, uint64_t ops,
                              std::chrono::duration<double, std::milli> elapsed) {
    double ms = elapsed.count();
    double perSec = (ms > 0.001) ? (static_cast<double>(ops) / (ms / 1000.0)) : 0;
    std::printf("  %-45s %8.2f ms  %10.0f ops/sec  (%llu ops)\n",
                name, ms, perSec, static_cast<unsigned long long>(ops));
    return {ms, perSec, ops};
}

// ── Test data ────────────────────────────────────────────────────────────────

static BodyDef makeBenchDef(float x, float y) {
    BodyDef def;
    def.type = BodyType::Dynamic;
    def.initialTransform = Transform(Vec3(x, y, 0));
    def.mass = 1.0f;
    float I = 0.4f * 1.0f * 0.5f * 0.5f;
    def.localInertia = Mat3(I, 0, 0,  0, I, 0,  0, 0, I);
    def.linearVelocity = Vec3(0, -1.0f, 0);
    def.restitution = 0.3f;
    def.friction = 0.5f;
    return def;
}

// ═════════════════════════════════════════════════════════════════════════════
// Bench 1: Body creation throughput (10K bodies)
// ═════════════════════════════════════════════════════════════════════════════

static void bench_body_creation() {
    std::printf("\n=== Bench 1: Body creation throughput (10K bodies) ===\n");

    constexpr uint32_t N = 10000;
    constexpr uint32_t Reps = 100;
    uint64_t totalOps = 0;

    auto start = Clock::now();
    for (uint32_t rep = 0; rep < Reps; ++rep) {
        BodyManager mgr(N);
        for (uint32_t i = 0; i < N; ++i) {
            float x = static_cast<float>(i % 100) * 2.0f;
            float y = static_cast<float>(i / 100) * 2.0f;
            mgr.createBody(makeBenchDef(x, y));
        }
        totalOps += N;
    }
    auto end = Clock::now();
    benchLoop("Body creation (10K × 100 reps)", totalOps, end - start);
}

// ═════════════════════════════════════════════════════════════════════════════
// Bench 2: Body destruction throughput (swap-and-pop)
// ═════════════════════════════════════════════════════════════════════════════

static void bench_body_destruction() {
    std::printf("\n=== Bench 2: Body destruction throughput (10K bodies) ===\n");

    constexpr uint32_t N = 10000;
    constexpr uint32_t Reps = 50;
    uint64_t totalOps = 0;

    auto start = Clock::now();
    for (uint32_t rep = 0; rep < Reps; ++rep) {
        BodyManager mgr(N);
        std::vector<BodyHandle> handles(N);
        for (uint32_t i = 0; i < N; ++i) {
            handles[i] = mgr.createBody(makeBenchDef(
                static_cast<float>(i), 0));
        }

        // Destroy all bodies (various orders to test swap-and-pop).
        for (uint32_t i = 0; i < N; ++i) {
            if (mgr.isValid(handles[i])) {
                mgr.destroyBody(handles[i]);
            }
        }
        totalOps += N;
    }
    auto end = Clock::now();
    benchLoop("Body destruction (10K × 50 reps)", totalOps, end - start);
}

// ═════════════════════════════════════════════════════════════════════════════
// Bench 3: World inertia update (batch, 10K bodies)
// ═════════════════════════════════════════════════════════════════════════════

static void bench_world_inertia() {
    std::printf("\n=== Bench 3: World inertia update (10K bodies) ===\n");

    constexpr uint32_t N = 10000;
    constexpr uint32_t Iters = 1000;

    RigidBodyStore store(N);
    for (uint32_t i = 0; i < N; ++i) {
        BodyDef def = makeBenchDef(static_cast<float>(i), 0);
        // Add some rotation variety.
        float angle = static_cast<float>(i) * 0.01f;
        def.initialTransform = Transform(Vec3(static_cast<float>(i), 0, 0),
            Quat::fromAxisAngle(Vec3::unitY(), angle));
        store.add(def);
    }

    auto start = Clock::now();
    for (uint32_t iter = 0; iter < Iters; ++iter) {
        store.updateAllWorldInertias();
    }
    auto end = Clock::now();
    benchLoop("World inertia update (10K × 1000 iters)",
              static_cast<uint64_t>(N) * Iters, end - start);
}

// ═════════════════════════════════════════════════════════════════════════════
// Bench 4: SolverBody conversion throughput (10K bodies)
// ═════════════════════════════════════════════════════════════════════════════

static void bench_solver_body_conversion() {
    std::printf("\n=== Bench 4: SolverBody conversion (10K bodies) ===\n");

    constexpr uint32_t N = 10000;
    constexpr uint32_t Iters = 1000;

    RigidBodyStore store(N);
    for (uint32_t i = 0; i < N; ++i) {
        store.add(makeBenchDef(static_cast<float>(i), 0));
    }
    store.updateAllWorldInertias();

    // Benchmark toSolverBody
    std::vector<SolverBody> solverBodies(N);
    auto start = Clock::now();
    for (uint32_t iter = 0; iter < Iters; ++iter) {
        for (uint32_t i = 0; i < N; ++i) {
            solverBodies[i] = store.toSolverBody(i);
        }
    }
    auto mid = Clock::now();
    benchLoop("toSolverBody (10K × 1000 iters)",
              static_cast<uint64_t>(N) * Iters, mid - start);

    // Benchmark fromSolverBody
    auto start2 = Clock::now();
    for (uint32_t iter = 0; iter < Iters; ++iter) {
        for (uint32_t i = 0; i < N; ++i) {
            store.fromSolverBody(i, solverBodies[i]);
        }
    }
    auto end = Clock::now();
    benchLoop("fromSolverBody (10K × 1000 iters)",
              static_cast<uint64_t>(N) * Iters, end - start2);
}

// ═════════════════════════════════════════════════════════════════════════════
// Bench 5: Island building (10K bodies)
// ═════════════════════════════════════════════════════════════════════════════

static void bench_island_building() {
    std::printf("\n=== Bench 5: Island building (10K bodies) ===\n");

    constexpr uint32_t N = 10000;

    // Bench 5a: 100 islands of 100 bodies each.
    {
        constexpr uint32_t Iters = 500;
        IslandManager islands(N);

        auto start = Clock::now();
        for (uint32_t iter = 0; iter < Iters; ++iter) {
            islands.reset(N);
            // Create 100 chains of 100.
            for (uint32_t chain = 0; chain < 100; ++chain) {
                for (uint32_t i = 1; i < 100; ++i) {
                    islands.unite(chain * 100 + 0, chain * 100 + i);
                }
            }
            islands.buildIslands();
        }
        auto end = Clock::now();
        benchLoop("100 islands × 100 bodies (500 iters)",
                  static_cast<uint64_t>(N) * Iters, end - start);
    }

    // Bench 5b: 1 giant island.
    {
        constexpr uint32_t Iters = 500;
        IslandManager islands(N);

        auto start = Clock::now();
        for (uint32_t iter = 0; iter < Iters; ++iter) {
            islands.reset(N);
            for (uint32_t i = 1; i < N; ++i) {
                islands.unite(0, i);
            }
            islands.buildIslands();
        }
        auto end = Clock::now();
        benchLoop("1 island × 10K bodies (500 iters)",
                  static_cast<uint64_t>(N) * Iters, end - start);
    }

    // Bench 5c: 10K separate islands.
    {
        constexpr uint32_t Iters = 500;
        IslandManager islands(N);

        auto start = Clock::now();
        for (uint32_t iter = 0; iter < Iters; ++iter) {
            islands.reset(N);
            // No unites — every body is its own island.
            islands.buildIslands();
        }
        auto end = Clock::now();
        benchLoop("10K islands × 1 body each (500 iters)",
                  static_cast<uint64_t>(N) * Iters, end - start);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Bench 6: Sleep update pass (10K bodies)
// ═════════════════════════════════════════════════════════════════════════════

static void bench_sleep_update() {
    std::printf("\n=== Bench 6: Sleep update pass (10K bodies) ===\n");

    constexpr uint32_t N = 10000;
    constexpr uint32_t Iters = 1000;

    RigidBodyStore store(N);
    for (uint32_t i = 0; i < N; ++i) {
        BodyDef def = makeBenchDef(static_cast<float>(i), 0);
        // Half with velocity (awake), half stationary (will sleep).
        if (i % 2 == 0) {
            def.linearVelocity = Vec3(0, -2.0f, 0);
        } else {
            def.linearVelocity = Vec3::zero();
        }
        store.add(def);
    }

    SleepManager sleepMgr;

    auto start = Clock::now();
    for (uint32_t iter = 0; iter < Iters; ++iter) {
        sleepMgr.updateSleep(store, 1.0f / 60.0f);
    }
    auto end = Clock::now();
    benchLoop("Sleep update (10K × 1000 iters)",
              static_cast<uint64_t>(N) * Iters, end - start);
}

// ═════════════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("=== Pulse RigidBody Module Benchmarks ===\n");

    bench_body_creation();
    bench_body_destruction();
    bench_world_inertia();
    bench_solver_body_conversion();
    bench_island_building();
    bench_sleep_update();

    std::printf("\n=== Benchmarks complete ===\n");
    return 0;
}
