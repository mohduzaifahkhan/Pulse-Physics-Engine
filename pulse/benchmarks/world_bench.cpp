/**
 * @file world_bench.cpp
 * @brief End-to-end pipeline benchmarks for the Pulse World module (Module 13).
 *
 * Benchmarks:
 *  1. Free fall (N bodies, no collisions) — integration throughput
 *  2. Sphere rain (N spheres onto a floor) — full pipeline
 *  3. Dense pile (N overlapping spheres) — broadphase + solver stress
 *  4. Sleep efficiency (mostly sleeping world)
 *  5. Create/destroy churn — body lifecycle overhead
 */

#include <pulse/world/world_common.h>
#include <pulse/world/world.h>

#include <pulse/math/vec3.h>
#include <pulse/math/mat3.h>
#include <pulse/math/transform.h>
#include <pulse/shapes/sphere.h>
#include <pulse/shapes/box.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

using namespace pulse;

// ── Helpers ──────────────────────────────────────────────────────────────────

static BodyDef makeDynSphere(Vec3 pos, float r = 0.5f, float mass = 1.0f) {
    BodyDef def;
    def.type = BodyType::Dynamic;
    def.initialTransform = Transform(pos);
    def.mass = mass;
    def.shapeType = ShapeType::Sphere;
    def.linearDamping = 0.0f;
    def.angularDamping = 0.0f;
    float I = 0.4f * mass * r * r;
    def.localInertia = Mat3(I, 0, 0, 0, I, 0, 0, 0, I);
    return def;
}

static BodyDef makeStatFloor(Vec3 pos) {
    BodyDef def;
    def.type = BodyType::Static;
    def.initialTransform = Transform(pos);
    def.mass = 0.0f;
    def.shapeType = ShapeType::Box;
    return def;
}

using Clock = std::chrono::high_resolution_clock;

static double elapsedMs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// =============================================================================
// BENCH 1: Free Fall (no collisions)
// =============================================================================

static void benchFreeFall(uint32_t N, uint32_t steps) {
    WorldConfig cfg;
    cfg.maxBodies = N + 16;
    cfg.maxPairs = N * 4;
    PhysicsWorld world(cfg);

    Sphere sphere(0.5f);
    for (uint32_t i = 0; i < N; ++i) {
        float x = static_cast<float>(i % 100) * 3.0f;
        float y = 50.0f + static_cast<float>(i / 100) * 3.0f;
        float z = static_cast<float>((i / 10) % 10) * 3.0f;
        world.createBody(makeDynSphere(Vec3(x, y, z)), &sphere);
    }

    float dt = 1.0f / 60.0f;
    auto start = Clock::now();
    for (uint32_t s = 0; s < steps; ++s) {
        world.singleStep(dt);
    }
    auto end = Clock::now();

    double totalMs = elapsedMs(start, end);
    double perStepMs = totalMs / steps;
    double perBodyNs = (totalMs * 1e6) / (static_cast<double>(steps) * N);

    std::printf("  FreeFall %5u bodies x %3u steps: %8.2f ms total, %6.3f ms/step, %6.1f ns/body/step\n",
                N, steps, totalMs, perStepMs, perBodyNs);
}

// =============================================================================
// BENCH 2: Sphere Rain (collisions with static floor)
// =============================================================================

static void benchSphereRain(uint32_t N, uint32_t steps) {
    WorldConfig cfg;
    cfg.maxBodies = N + 16;
    cfg.maxPairs = N * 8;
    PhysicsWorld world(cfg);

    // Static floor.
    Box floorBox(Vec3(100.0f, 0.5f, 100.0f));
    world.createBody(makeStatFloor(Vec3(0, -0.5f, 0)), &floorBox);

    Sphere sphere(0.5f);
    for (uint32_t i = 0; i < N; ++i) {
        float x = static_cast<float>(i % 20) * 1.5f - 15.0f;
        float y = 2.0f + static_cast<float>(i / 20) * 1.5f;
        float z = static_cast<float>((i / 5) % 10) * 1.5f - 7.5f;
        world.createBody(makeDynSphere(Vec3(x, y, z)), &sphere);
    }

    float dt = 1.0f / 60.0f;
    auto start = Clock::now();
    for (uint32_t s = 0; s < steps; ++s) {
        world.singleStep(dt);
    }
    auto end = Clock::now();

    double totalMs = elapsedMs(start, end);
    double perStepMs = totalMs / steps;

    std::printf("  SphereRain %5u bodies x %3u steps: %8.2f ms total, %6.3f ms/step\n",
                N, steps, totalMs, perStepMs);
}

// =============================================================================
// BENCH 3: Dense Pile (many overlapping bodies — solver stress)
// =============================================================================

static void benchDensePile(uint32_t N, uint32_t steps) {
    WorldConfig cfg;
    cfg.gravity = Vec3(0, -10, 0);
    cfg.maxBodies = N + 16;
    cfg.maxPairs = N * N / 2; // Worst case.
    if (cfg.maxPairs > 100000) cfg.maxPairs = 100000;
    PhysicsWorld world(cfg);

    Sphere sphere(0.3f);
    for (uint32_t i = 0; i < N; ++i) {
        float x = static_cast<float>(i % 5) * 0.5f;
        float y = 1.0f + static_cast<float>(i / 5) * 0.5f;
        float z = static_cast<float>((i / 3) % 4) * 0.5f;
        world.createBody(makeDynSphere(Vec3(x, y, z), 0.3f), &sphere);
    }

    float dt = 1.0f / 60.0f;
    auto start = Clock::now();
    for (uint32_t s = 0; s < steps; ++s) {
        world.singleStep(dt);
    }
    auto end = Clock::now();

    double totalMs = elapsedMs(start, end);
    double perStepMs = totalMs / steps;

    std::printf("  DensePile %5u bodies x %3u steps: %8.2f ms total, %6.3f ms/step\n",
                N, steps, totalMs, perStepMs);
}

// =============================================================================
// BENCH 4: Sleep Efficiency
// =============================================================================

static void benchSleepEfficiency(uint32_t N, uint32_t steps) {
    WorldConfig cfg;
    cfg.gravity = Vec3::zero(); // Bodies are stationary.
    cfg.maxBodies = N + 16;
    cfg.maxPairs = N * 2;
    cfg.sleepConfig.timeToSleep = 0.1f; // Quick sleep.
    PhysicsWorld world(cfg);

    Sphere sphere(0.5f);
    for (uint32_t i = 0; i < N; ++i) {
        float x = static_cast<float>(i % 50) * 3.0f;
        float y = static_cast<float>(i / 50) * 3.0f;
        world.createBody(makeDynSphere(Vec3(x, y, 0)), &sphere);
    }

    // Let bodies sleep first (warmup).
    float dt = 1.0f / 60.0f;
    for (int w = 0; w < 30; ++w) {
        world.singleStep(dt);
    }

    // Now benchmark with mostly-sleeping bodies.
    auto start = Clock::now();
    for (uint32_t s = 0; s < steps; ++s) {
        world.singleStep(dt);
    }
    auto end = Clock::now();

    double totalMs = elapsedMs(start, end);
    double perStepMs = totalMs / steps;

    std::printf("  SleepWorld %5u bodies x %3u steps: %8.2f ms total, %6.3f ms/step (mostly sleeping)\n",
                N, steps, totalMs, perStepMs);
}

// =============================================================================
// BENCH 5: Create/Destroy Churn
// =============================================================================

static void benchCreateDestroyChurn(uint32_t N, uint32_t cycles) {
    WorldConfig cfg;
    cfg.maxBodies = N + 16;
    cfg.maxPairs = N * 4;
    PhysicsWorld world(cfg);
    Sphere sphere(0.5f);

    auto start = Clock::now();
    for (uint32_t c = 0; c < cycles; ++c) {
        // Create N bodies.
        BodyHandle* handles = static_cast<BodyHandle*>(std::malloc(N * sizeof(BodyHandle)));
        for (uint32_t i = 0; i < N; ++i) {
            handles[i] = world.createBody(
                makeDynSphere(Vec3(static_cast<float>(i), 0, 0)),
                &sphere);
        }
        // Destroy all.
        for (uint32_t i = 0; i < N; ++i) {
            world.destroyBody(handles[i]);
        }
        std::free(handles);
    }
    auto end = Clock::now();

    double totalMs = elapsedMs(start, end);
    double perCycleMs = totalMs / cycles;
    double perOpUs = (totalMs * 1000.0) / (static_cast<double>(cycles) * N * 2);

    std::printf("  Churn %5u bodies x %3u cycles: %8.2f ms total, %6.3f ms/cycle, %6.2f µs/create-or-destroy\n",
                N, cycles, totalMs, perCycleMs, perOpUs);
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    std::printf("═══════════════════════════════════════════════════════════════════\n");
    std::printf("  Pulse Physics Engine — World Module Benchmarks (Module 13)\n");
    std::printf("═══════════════════════════════════════════════════════════════════\n\n");

    std::printf("── Free Fall (Integration Throughput, No Collisions) ──\n");
    benchFreeFall(100, 100);
    benchFreeFall(500, 100);
    benchFreeFall(1000, 100);
    benchFreeFall(5000, 50);

    std::printf("\n── Sphere Rain (Full Pipeline, Static Floor) ──\n");
    benchSphereRain(50, 100);
    benchSphereRain(200, 60);
    benchSphereRain(500, 30);

    std::printf("\n── Dense Pile (Solver Stress) ──\n");
    benchDensePile(20, 60);
    benchDensePile(50, 30);
    benchDensePile(100, 20);

    std::printf("\n── Sleep Efficiency (Mostly Sleeping World) ──\n");
    benchSleepEfficiency(100, 100);
    benchSleepEfficiency(500, 100);
    benchSleepEfficiency(1000, 50);

    std::printf("\n── Create/Destroy Churn ──\n");
    benchCreateDestroyChurn(100, 100);
    benchCreateDestroyChurn(500, 50);
    benchCreateDestroyChurn(1000, 20);

    std::printf("\n═══════════════════════════════════════════════════════════════════\n");
    std::printf("  All benchmarks complete.\n");
    std::printf("═══════════════════════════════════════════════════════════════════\n");

    return 0;
}
