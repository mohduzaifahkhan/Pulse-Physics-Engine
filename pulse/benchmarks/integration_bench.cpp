/**
 * @file integration_bench.cpp
 * @brief Performance benchmarks for the Pulse integration module (Module 12).
 *
 * Measures throughput for:
 *  1. Semi-Implicit Euler velocity integration (1K / 10K bodies)
 *  2. Semi-Implicit Euler position integration (1K / 10K bodies)
 *  3. Semi-Implicit Euler full step (1K / 10K bodies)
 *  4. Velocity Verlet full step (1K / 10K bodies)
 *  5. RK4 full step (1K / 10K bodies)
 *  6. Unified dispatch overhead comparison
 */

#include <pulse/integration/integration_common.h>
#include <pulse/integration/semi_implicit_euler.h>
#include <pulse/integration/velocity_verlet.h>
#include <pulse/integration/rk4.h>
#include <pulse/integration/integrator.h>

#include <pulse/rigidbody/rigid_body_common.h>
#include <pulse/rigidbody/rigid_body.h>
#include <pulse/math/math_common.h>
#include <pulse/math/vec3.h>
#include <pulse/math/mat3.h>

#include <cstdio>
#include <cstdint>
#include <chrono>

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
    std::printf("  %-50s %8.2f ms  %10.0f ops/sec  (%llu ops)\n",
                name, ms, perSec, static_cast<unsigned long long>(ops));
    return {ms, perSec, ops};
}

// ── Test data ────────────────────────────────────────────────────────────────

static void populateStore(RigidBodyStore& store, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        BodyDef def;
        def.type = BodyType::Dynamic;
        def.initialTransform = Transform(Vec3(
            static_cast<float>(i % 100) * 2.0f,
            static_cast<float>(i / 100) * 2.0f,
            0.0f));
        def.mass = 1.0f;
        float I = 0.4f * 1.0f * 0.5f * 0.5f;
        def.localInertia = Mat3(I, 0, 0, 0, I, 0, 0, 0, I);
        def.linearVelocity = Vec3(0, -1.0f, 0);
        def.angularVelocity = Vec3(0.1f, 0.2f, 0.3f);
        def.linearDamping = 0.01f;
        def.angularDamping = 0.01f;
        def.restitution = 0.3f;
        def.friction = 0.5f;
        store.add(def);
    }
    store.updateAllWorldInertias();
}

// ═════════════════════════════════════════════════════════════════════════════
// Bench 1: Euler Velocity Integration
// ═════════════════════════════════════════════════════════════════════════════

static void bench_euler_velocity() {
    std::printf("\n=== Bench 1: Euler Velocity Integration ===\n");

    const uint32_t sizes[] = {1000, 10000};
    const uint32_t Reps = 1000;

    for (uint32_t N : sizes) {
        RigidBodyStore store(N);
        populateStore(store, N);
        Vec3 gravity(0, -9.81f, 0);
        float dt = 1.0f / 60.0f;

        auto start = Clock::now();
        for (uint32_t r = 0; r < Reps; ++r) {
            // Apply some force to prevent compiler from optimizing away
            store.force(r % N) += Vec3(0.01f, 0, 0);
            semiImplicitEulerIntegrateVelocities(store, gravity, dt);
        }
        auto end = Clock::now();

        char name[128];
        std::snprintf(name, sizeof(name), "Euler velocity (%u bodies x %u reps)", N, Reps);
        benchLoop(name, static_cast<uint64_t>(N) * Reps, end - start);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Bench 2: Euler Position Integration
// ═════════════════════════════════════════════════════════════════════════════

static void bench_euler_position() {
    std::printf("\n=== Bench 2: Euler Position Integration ===\n");

    const uint32_t sizes[] = {1000, 10000};
    const uint32_t Reps = 1000;

    for (uint32_t N : sizes) {
        RigidBodyStore store(N);
        populateStore(store, N);
        float dt = 1.0f / 60.0f;

        auto start = Clock::now();
        for (uint32_t r = 0; r < Reps; ++r) {
            semiImplicitEulerIntegratePositions(store, dt);
        }
        auto end = Clock::now();

        char name[128];
        std::snprintf(name, sizeof(name), "Euler position (%u bodies x %u reps)", N, Reps);
        benchLoop(name, static_cast<uint64_t>(N) * Reps, end - start);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Bench 3: Euler Full Step
// ═════════════════════════════════════════════════════════════════════════════

static void bench_euler_full() {
    std::printf("\n=== Bench 3: Euler Full Step ===\n");

    const uint32_t sizes[] = {1000, 10000};
    const uint32_t Reps = 1000;

    for (uint32_t N : sizes) {
        RigidBodyStore store(N);
        populateStore(store, N);
        Vec3 gravity(0, -9.81f, 0);
        float dt = 1.0f / 60.0f;

        auto start = Clock::now();
        for (uint32_t r = 0; r < Reps; ++r) {
            semiImplicitEulerIntegrate(store, gravity, dt);
        }
        auto end = Clock::now();

        char name[128];
        std::snprintf(name, sizeof(name), "Euler full step (%u bodies x %u reps)", N, Reps);
        benchLoop(name, static_cast<uint64_t>(N) * Reps, end - start);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Bench 4: Velocity Verlet Full Step
// ═════════════════════════════════════════════════════════════════════════════

static void bench_verlet_full() {
    std::printf("\n=== Bench 4: Velocity Verlet Full Step ===\n");

    const uint32_t sizes[] = {1000, 10000};
    const uint32_t Reps = 1000;

    for (uint32_t N : sizes) {
        RigidBodyStore store(N);
        populateStore(store, N);
        Vec3 gravity(0, -9.81f, 0);
        float dt = 1.0f / 60.0f;

        auto start = Clock::now();
        for (uint32_t r = 0; r < Reps; ++r) {
            verletIntegrate(store, gravity, dt);
        }
        auto end = Clock::now();

        char name[128];
        std::snprintf(name, sizeof(name), "Verlet full step (%u bodies x %u reps)", N, Reps);
        benchLoop(name, static_cast<uint64_t>(N) * Reps, end - start);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Bench 5: RK4 Full Step
// ═════════════════════════════════════════════════════════════════════════════

static void bench_rk4_full() {
    std::printf("\n=== Bench 5: RK4 Full Step ===\n");

    const uint32_t sizes[] = {1000, 10000};
    const uint32_t Reps = 500;

    for (uint32_t N : sizes) {
        RigidBodyStore store(N);
        populateStore(store, N);
        Vec3 gravity(0, -9.81f, 0);
        float dt = 1.0f / 60.0f;

        auto start = Clock::now();
        for (uint32_t r = 0; r < Reps; ++r) {
            rk4Integrate(store, gravity, dt);
        }
        auto end = Clock::now();

        char name[128];
        std::snprintf(name, sizeof(name), "RK4 full step (%u bodies x %u reps)", N, Reps);
        benchLoop(name, static_cast<uint64_t>(N) * Reps, end - start);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Bench 6: Integrator Comparison (Euler vs Verlet vs RK4)
// ═════════════════════════════════════════════════════════════════════════════

static void bench_comparison() {
    std::printf("\n=== Bench 6: Integrator Comparison (10K bodies x 500 reps) ===\n");

    constexpr uint32_t N = 10000;
    constexpr uint32_t Reps = 500;
    Vec3 gravity(0, -9.81f, 0);
    float dt = 1.0f / 60.0f;

    // Euler
    {
        RigidBodyStore store(N);
        populateStore(store, N);
        IntegrationConfig cfg;
        cfg.type = IntegratorType::SemiImplicitEuler;
        cfg.gravity = gravity;

        auto start = Clock::now();
        for (uint32_t r = 0; r < Reps; ++r) {
            integrate(store, cfg, dt);
        }
        auto end = Clock::now();
        benchLoop("Dispatch: Euler", static_cast<uint64_t>(N) * Reps, end - start);
    }

    // Verlet
    {
        RigidBodyStore store(N);
        populateStore(store, N);
        IntegrationConfig cfg;
        cfg.type = IntegratorType::VelocityVerlet;
        cfg.gravity = gravity;

        auto start = Clock::now();
        for (uint32_t r = 0; r < Reps; ++r) {
            integrate(store, cfg, dt);
        }
        auto end = Clock::now();
        benchLoop("Dispatch: Verlet", static_cast<uint64_t>(N) * Reps, end - start);
    }

    // RK4
    {
        RigidBodyStore store(N);
        populateStore(store, N);
        IntegrationConfig cfg;
        cfg.type = IntegratorType::RK4;
        cfg.gravity = gravity;

        auto start = Clock::now();
        for (uint32_t r = 0; r < Reps; ++r) {
            integrate(store, cfg, dt);
        }
        auto end = Clock::now();
        benchLoop("Dispatch: RK4", static_cast<uint64_t>(N) * Reps, end - start);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("===============================================\n");
    std::printf("  Pulse Integration Benchmarks (Module 12)\n");
    std::printf("===============================================\n");

    bench_euler_velocity();
    bench_euler_position();
    bench_euler_full();
    bench_verlet_full();
    bench_rk4_full();
    bench_comparison();

    std::printf("\n=== Integration benchmarks complete ===\n");
    return 0;
}
