/**
 * @file solver_bench.cpp
 * @brief Performance benchmarks for the Pulse solver module.
 *
 * Measures throughput for:
 *  1. Velocity iteration — 1000 contacts
 *  2. Position iteration — 1000 contacts
 *  3. Full solve pass — 100 manifolds × 4 contacts
 *  4. Warm-start vs cold-start convergence
 *  5. Scaling — 100 / 1000 / 10000 contacts
 */

#include <pulse/solver/solver_common.h>
#include <pulse/solver/contact_solver.h>
#include <pulse/solver/solver.h>

#include <pulse/contact/contact_common.h>
#include <pulse/contact/persistent_contact.h>
#include <pulse/contact/contact_manifold_persistent.h>
#include <pulse/narrowphase/narrowphase_common.h>
#include <pulse/math/math_common.h>
#include <pulse/math/vec3.h>

#include <cstdio>
#include <cstdint>
#include <chrono>
#include <cmath>

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
    std::printf("  %-40s %8.2f ms  %10.0f ops/sec  (%llu ops)\n",
                name, ms, perSec, static_cast<unsigned long long>(ops));
    return {ms, perSec, ops};
}

// ── Test data generation ─────────────────────────────────────────────────────

/// Create a random-ish SolverBody array
static void createBodies(SolverBody* bodies, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        float x = static_cast<float>(i % 100) * 2.0f;
        float y = static_cast<float>(i / 100) * 2.0f;
        bodies[i].position = Vec3(x, y, 0);
        bodies[i].linearVelocity = Vec3(0, -1.0f, 0);
        bodies[i].angularVelocity = Vec3::zero();
        bodies[i].invMass = 1.0f;
        float I = 0.4f * 1.0f * 0.5f * 0.5f;
        bodies[i].invInertia = Vec3(1.0f / I, 1.0f / I, 1.0f / I);
        bodies[i].restitution = 0.3f;
        bodies[i].friction = 0.5f;
        bodies[i].bodyId = i;
    }
    // Make body 0 static (the ground)
    bodies[0].invMass = 0.0f;
    bodies[0].invInertia = Vec3::zero();
    bodies[0].linearVelocity = Vec3::zero();
    bodies[0].position = Vec3(0, -1, 0);
    bodies[0].friction = 0.5f;
}

/// Create manifolds: each between body i and body 0 (ground)
static void createManifolds(PersistentManifold* manifolds, uint32_t count,
                             uint32_t contactsPerManifold) {
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t bodyId = i + 1;  // Skip ground (body 0)
        manifolds[i] = PersistentManifold(bodyId, 0);

        ContactManifold cm;
        for (uint32_t c = 0; c < contactsPerManifold && c < 4; ++c) {
            float offsetX = (c % 2 == 0) ? -0.5f : 0.5f;
            float offsetZ = (c / 2 == 0) ? -0.5f : 0.5f;
            ContactPoint cp(
                Vec3(offsetX, 0, offsetZ),
                Vec3(offsetX, -1, offsetZ),
                Vec3(0, 1, 0),
                0.01f
            );
            cm.addPoint(cp);
        }
        manifolds[i].mergeContacts(cm, 0.04f * 0.04f);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Benchmark 1: Velocity Iteration Throughput
// ═════════════════════════════════════════════════════════════════════════════

static void bench_velocity_iteration_1000() {
    const uint32_t numManifolds = 1000;
    const uint32_t numBodies = numManifolds + 1;  // +1 for ground

    SolverBody* bodies = new SolverBody[numBodies];
    PersistentManifold* manifolds = new PersistentManifold[numManifolds];

    createBodies(bodies, numBodies);
    createManifolds(manifolds, numManifolds, 1);

    ContactSolver solver;
    SolverConfig config;
    config.warmStarting = false;
    solver.initialize(bodies, numBodies, manifolds, numManifolds, config, 1.0f / 60.0f);

    const uint32_t iterations = 1000;
    auto start = Clock::now();
    for (uint32_t i = 0; i < iterations; ++i) {
        solver.solveVelocityConstraints(bodies);
    }
    auto end = Clock::now();

    uint64_t ops = static_cast<uint64_t>(iterations) * numManifolds;
    benchLoop("Velocity iter (1000 contacts)", ops, end - start);

    delete[] bodies;
    delete[] manifolds;
}

// ═════════════════════════════════════════════════════════════════════════════
// Benchmark 2: Position Iteration Throughput
// ═════════════════════════════════════════════════════════════════════════════

static void bench_position_iteration_1000() {
    const uint32_t numManifolds = 1000;
    const uint32_t numBodies = numManifolds + 1;

    SolverBody* bodies = new SolverBody[numBodies];
    PersistentManifold* manifolds = new PersistentManifold[numManifolds];

    createBodies(bodies, numBodies);
    createManifolds(manifolds, numManifolds, 1);

    ContactSolver solver;
    SolverConfig config;
    config.warmStarting = false;
    solver.initialize(bodies, numBodies, manifolds, numManifolds, config, 1.0f / 60.0f);

    const uint32_t iterations = 1000;
    auto start = Clock::now();
    for (uint32_t i = 0; i < iterations; ++i) {
        solver.solvePositionConstraints(bodies, config);
    }
    auto end = Clock::now();

    uint64_t ops = static_cast<uint64_t>(iterations) * numManifolds;
    benchLoop("Position iter (1000 contacts)", ops, end - start);

    delete[] bodies;
    delete[] manifolds;
}

// ═════════════════════════════════════════════════════════════════════════════
// Benchmark 3: Full Solve Pass (100 manifolds × 4 contacts)
// ═════════════════════════════════════════════════════════════════════════════

static void bench_full_solve_400_contacts() {
    const uint32_t numManifolds = 100;
    const uint32_t numBodies = numManifolds + 1;

    SolverBody* bodies = new SolverBody[numBodies];
    PersistentManifold* manifolds = new PersistentManifold[numManifolds];

    createBodies(bodies, numBodies);
    createManifolds(manifolds, numManifolds, 4);  // 4 contacts per manifold

    SolverConfig config;
    config.warmStarting = false;

    const uint32_t passes = 500;
    auto start = Clock::now();
    for (uint32_t i = 0; i < passes; ++i) {
        // Reset velocities for each pass
        for (uint32_t b = 1; b < numBodies; ++b) {
            bodies[b].linearVelocity = Vec3(0, -1.0f, 0);
        }
        solve(bodies, numBodies, manifolds, numManifolds, config, 1.0f / 60.0f);
    }
    auto end = Clock::now();

    benchLoop("Full solve (400 contacts)", passes, end - start);

    delete[] bodies;
    delete[] manifolds;
}

// ═════════════════════════════════════════════════════════════════════════════
// Benchmark 4: Warm-Start vs Cold-Start
// ═════════════════════════════════════════════════════════════════════════════

static void bench_warm_vs_cold() {
    const uint32_t numManifolds = 200;
    const uint32_t numBodies = numManifolds + 1;

    SolverBody* bodiesCold = new SolverBody[numBodies];
    SolverBody* bodiesWarm = new SolverBody[numBodies];
    PersistentManifold* manifoldsCold = new PersistentManifold[numManifolds];
    PersistentManifold* manifoldsWarm = new PersistentManifold[numManifolds];

    SolverConfig configCold;
    configCold.warmStarting = false;

    SolverConfig configWarm;
    configWarm.warmStarting = true;

    const uint32_t passes = 300;

    // Cold start
    auto start = Clock::now();
    for (uint32_t i = 0; i < passes; ++i) {
        createBodies(bodiesCold, numBodies);
        createManifolds(manifoldsCold, numManifolds, 2);
        for (uint32_t b = 1; b < numBodies; ++b)
            bodiesCold[b].linearVelocity = Vec3(0, -2.0f, 0);
        solve(bodiesCold, numBodies, manifoldsCold, numManifolds, configCold, 1.0f / 60.0f);
    }
    auto end = Clock::now();
    benchLoop("Cold start (200 manifolds)", passes, end - start);

    // Warm start (reuse manifolds with impulses)
    createBodies(bodiesWarm, numBodies);
    createManifolds(manifoldsWarm, numManifolds, 2);

    // Prime with one solve
    for (uint32_t b = 1; b < numBodies; ++b)
        bodiesWarm[b].linearVelocity = Vec3(0, -2.0f, 0);
    solve(bodiesWarm, numBodies, manifoldsWarm, numManifolds, configWarm, 1.0f / 60.0f);

    start = Clock::now();
    for (uint32_t i = 0; i < passes; ++i) {
        for (uint32_t b = 1; b < numBodies; ++b)
            bodiesWarm[b].linearVelocity = Vec3(0, -2.0f, 0);
        solve(bodiesWarm, numBodies, manifoldsWarm, numManifolds, configWarm, 1.0f / 60.0f);
    }
    end = Clock::now();
    benchLoop("Warm start (200 manifolds)", passes, end - start);

    delete[] bodiesCold;
    delete[] bodiesWarm;
    delete[] manifoldsCold;
    delete[] manifoldsWarm;
}

// ═════════════════════════════════════════════════════════════════════════════
// Benchmark 5: Scaling
// ═════════════════════════════════════════════════════════════════════════════

static void bench_scaling() {
    const uint32_t sizes[] = { 100, 1000, 5000 };
    const uint32_t passes[] = { 1000, 200, 50 };

    SolverConfig config;
    config.warmStarting = false;

    for (int s = 0; s < 3; ++s) {
        uint32_t numManifolds = sizes[s];
        uint32_t numBodies = numManifolds + 1;
        uint32_t numPasses = passes[s];

        SolverBody* bodies = new SolverBody[numBodies];
        PersistentManifold* manifolds = new PersistentManifold[numManifolds];

        createBodies(bodies, numBodies);
        createManifolds(manifolds, numManifolds, 1);

        auto start = Clock::now();
        for (uint32_t i = 0; i < numPasses; ++i) {
            for (uint32_t b = 1; b < numBodies; ++b)
                bodies[b].linearVelocity = Vec3(0, -1.0f, 0);
            solve(bodies, numBodies, manifolds, numManifolds, config, 1.0f / 60.0f);
        }
        auto end = Clock::now();

        char label[128];
        std::snprintf(label, sizeof(label), "Scaling: %u contacts", numManifolds);
        benchLoop(label, numPasses, end - start);

        delete[] bodies;
        delete[] manifolds;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("========================================\n");
    std::printf("  Pulse Solver Module — Benchmarks\n");
    std::printf("========================================\n\n");

    std::printf("--- Velocity Iteration ---\n");
    bench_velocity_iteration_1000();

    std::printf("\n--- Position Iteration ---\n");
    bench_position_iteration_1000();

    std::printf("\n--- Full Solve Pass ---\n");
    bench_full_solve_400_contacts();

    std::printf("\n--- Warm vs Cold Start ---\n");
    bench_warm_vs_cold();

    std::printf("\n--- Scaling ---\n");
    bench_scaling();

    std::printf("\nDone.\n");
    return 0;
}
