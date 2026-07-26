/**
 * @file constraints_bench.cpp
 * @brief Performance benchmarks for the Pulse constraints module (Module 10).
 *
 * Measures throughput for:
 *  1. Distance constraint solve iteration — 1000 constraints
 *  2. Hinge constraint solve iteration — 1000 constraints
 *  3. Full mixed constraint solve pass
 *  4. Scaling — 100 / 1000 / 10000 distance constraints
 *  5. Soft vs rigid constraint comparison
 *  6. 6-DoF vs specialized constraint overhead
 */

#include <pulse/constraints/constraint_common.h>
#include <pulse/constraints/distance_constraint.h>
#include <pulse/constraints/hinge_constraint.h>
#include <pulse/constraints/slider_constraint.h>
#include <pulse/constraints/six_dof_constraint.h>
#include <pulse/constraints/spring_constraint.h>
#include <pulse/constraints/motor_constraint.h>

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

// ── Test data generation ─────────────────────────────────────────────────────

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
        bodies[i].restitution = 0.0f;
        bodies[i].friction = 0.5f;
        bodies[i].bodyId = i;
    }
    // Body 0 is static ground
    bodies[0].invMass = 0.0f;
    bodies[0].invInertia = Vec3::zero();
    bodies[0].linearVelocity = Vec3::zero();
    bodies[0].position = Vec3(0, -1, 0);
}

// ═════════════════════════════════════════════════════════════════════════════
// Bench 1: Distance constraint solve iteration (1000 constraints)
// ═════════════════════════════════════════════════════════════════════════════

static void bench_distance_iteration() {
    std::printf("\n=== Bench 1: Distance constraint velocity iteration (1000) ===\n");

    constexpr uint32_t N = 1001; // 1000 dynamic + 1 static
    constexpr uint32_t C = 1000;
    constexpr uint32_t ITERS = 10;
    constexpr uint32_t REPS = 100;

    std::vector<SolverBody> bodies(N);
    createBodies(bodies.data(), N);

    std::vector<DistanceConstraint> constraints(C);
    SolverConfig config;
    float dt = 1.0f / 60.0f;

    for (uint32_t i = 0; i < C; ++i) {
        constraints[i] = DistanceConstraint(Vec3::zero(), Vec3::zero(), 1.0f, 0, i + 1);
        constraints[i].initialize(bodies.data(), N, config, dt);
    }

    auto start = Clock::now();
    for (uint32_t rep = 0; rep < REPS; ++rep) {
        for (uint32_t iter = 0; iter < ITERS; ++iter) {
            for (uint32_t i = 0; i < C; ++i) {
                constraints[i].solveVelocity(bodies.data());
            }
        }
    }
    auto end = Clock::now();

    uint64_t ops = static_cast<uint64_t>(REPS) * ITERS * C;
    benchLoop("Distance constraint solve", ops, end - start);
}

// ═════════════════════════════════════════════════════════════════════════════
// Bench 2: Hinge constraint solve iteration (1000 constraints)
// ═════════════════════════════════════════════════════════════════════════════

static void bench_hinge_iteration() {
    std::printf("\n=== Bench 2: Hinge constraint velocity iteration (1000) ===\n");

    constexpr uint32_t N = 1001;
    constexpr uint32_t C = 1000;
    constexpr uint32_t ITERS = 10;
    constexpr uint32_t REPS = 50;

    std::vector<SolverBody> bodies(N);
    createBodies(bodies.data(), N);

    std::vector<HingeConstraint> constraints(C);
    SolverConfig config;
    float dt = 1.0f / 60.0f;

    for (uint32_t i = 0; i < C; ++i) {
        constraints[i] = HingeConstraint(Vec3::zero(), Vec3::zero(),
                                         Vec3::unitY(), Vec3::unitY(), 0, i + 1);
        constraints[i].initialize(bodies.data(), N, config, dt);
    }

    auto start = Clock::now();
    for (uint32_t rep = 0; rep < REPS; ++rep) {
        for (uint32_t iter = 0; iter < ITERS; ++iter) {
            for (uint32_t i = 0; i < C; ++i) {
                constraints[i].solveVelocity(bodies.data());
            }
        }
    }
    auto end = Clock::now();

    uint64_t ops = static_cast<uint64_t>(REPS) * ITERS * C;
    benchLoop("Hinge constraint solve (5 rows)", ops, end - start);
}

// ═════════════════════════════════════════════════════════════════════════════
// Bench 3: Mixed constraint solve pass
// ═════════════════════════════════════════════════════════════════════════════

static void bench_mixed_constraints() {
    std::printf("\n=== Bench 3: Mixed constraint solve pass ===\n");

    constexpr uint32_t N = 501;
    constexpr uint32_t ITERS = 10;
    constexpr uint32_t REPS = 100;

    std::vector<SolverBody> bodies(N);
    createBodies(bodies.data(), N);

    SolverConfig config;
    float dt = 1.0f / 60.0f;

    // Mix: 200 distance + 100 hinge + 100 slider + 100 motor
    constexpr uint32_t nDist = 200, nHinge = 100, nSlider = 100, nMotor = 100;

    std::vector<DistanceConstraint> dists(nDist);
    std::vector<HingeConstraint> hinges(nHinge);
    std::vector<SliderConstraint> sliders(nSlider);
    std::vector<MotorConstraint> motors(nMotor);

    uint32_t bodyIdx = 1;
    for (uint32_t i = 0; i < nDist; ++i, ++bodyIdx) {
        dists[i] = DistanceConstraint(Vec3::zero(), Vec3::zero(), 1.0f, 0, bodyIdx);
        dists[i].initialize(bodies.data(), N, config, dt);
    }
    for (uint32_t i = 0; i < nHinge; ++i, ++bodyIdx) {
        hinges[i] = HingeConstraint(Vec3::zero(), Vec3::zero(),
                                    Vec3::unitY(), Vec3::unitY(), 0, bodyIdx);
        hinges[i].initialize(bodies.data(), N, config, dt);
    }
    for (uint32_t i = 0; i < nSlider; ++i, ++bodyIdx) {
        sliders[i] = SliderConstraint(Vec3::zero(), Vec3::zero(),
                                      Vec3::unitX(), 0, bodyIdx);
        sliders[i].initialize(bodies.data(), N, config, dt);
    }
    for (uint32_t i = 0; i < nMotor; ++i, ++bodyIdx) {
        motors[i] = MotorConstraint(Vec3::unitY(), 100.0f, MotorMode::Angular, 0, bodyIdx);
        motors[i].targetVelocity = 3.0f;
        motors[i].initialize(bodies.data(), N, config, dt);
    }

    auto start = Clock::now();
    for (uint32_t rep = 0; rep < REPS; ++rep) {
        for (uint32_t iter = 0; iter < ITERS; ++iter) {
            for (uint32_t i = 0; i < nDist; ++i)
                dists[i].solveVelocity(bodies.data());
            for (uint32_t i = 0; i < nHinge; ++i)
                hinges[i].solveVelocity(bodies.data());
            for (uint32_t i = 0; i < nSlider; ++i)
                sliders[i].solveVelocity(bodies.data());
            for (uint32_t i = 0; i < nMotor; ++i)
                motors[i].solveVelocity(bodies.data());
        }
    }
    auto end = Clock::now();

    uint64_t ops = static_cast<uint64_t>(REPS) * ITERS * (nDist + nHinge + nSlider + nMotor);
    benchLoop("Mixed constraint solve pass", ops, end - start);
}

// ═════════════════════════════════════════════════════════════════════════════
// Bench 4: Scaling — distance constraints (100 / 1000 / 10000)
// ═════════════════════════════════════════════════════════════════════════════

static void bench_scaling(uint32_t count) {
    char label[128];
    std::snprintf(label, sizeof(label), "Distance scaling (%u constraints)", count);

    uint32_t N = count + 1;
    std::vector<SolverBody> bodies(N);
    createBodies(bodies.data(), N);

    std::vector<DistanceConstraint> constraints(count);
    SolverConfig config;
    float dt = 1.0f / 60.0f;

    for (uint32_t i = 0; i < count; ++i) {
        constraints[i] = DistanceConstraint(Vec3::zero(), Vec3::zero(), 1.0f, 0, i + 1);
        constraints[i].initialize(bodies.data(), N, config, dt);
    }

    constexpr uint32_t ITERS = 10;
    uint32_t REPS = (count <= 100) ? 1000 : (count <= 1000) ? 100 : 10;

    auto start = Clock::now();
    for (uint32_t rep = 0; rep < REPS; ++rep) {
        for (uint32_t iter = 0; iter < ITERS; ++iter) {
            for (uint32_t i = 0; i < count; ++i) {
                constraints[i].solveVelocity(bodies.data());
            }
        }
    }
    auto end = Clock::now();

    uint64_t ops = static_cast<uint64_t>(REPS) * ITERS * count;
    benchLoop(label, ops, end - start);
}

// ═════════════════════════════════════════════════════════════════════════════
// Bench 5: Soft vs rigid constraint comparison
// ═════════════════════════════════════════════════════════════════════════════

static void bench_soft_vs_rigid() {
    std::printf("\n=== Bench 5: Soft vs rigid constraint comparison (1000) ===\n");

    constexpr uint32_t N = 1001;
    constexpr uint32_t C = 1000;
    constexpr uint32_t ITERS = 10;
    constexpr uint32_t REPS = 100;

    SolverConfig config;
    float dt = 1.0f / 60.0f;

    // Rigid
    {
        std::vector<SolverBody> bodies(N);
        createBodies(bodies.data(), N);

        std::vector<DistanceConstraint> constraints(C);
        for (uint32_t i = 0; i < C; ++i) {
            constraints[i] = DistanceConstraint(Vec3::zero(), Vec3::zero(), 1.0f, 0, i + 1);
            constraints[i].initialize(bodies.data(), N, config, dt);
        }

        auto start = Clock::now();
        for (uint32_t rep = 0; rep < REPS; ++rep) {
            for (uint32_t iter = 0; iter < ITERS; ++iter) {
                for (uint32_t i = 0; i < C; ++i)
                    constraints[i].solveVelocity(bodies.data());
            }
        }
        auto end = Clock::now();
        benchLoop("Rigid distance constraint", static_cast<uint64_t>(REPS) * ITERS * C, end - start);
    }

    // Soft
    {
        std::vector<SolverBody> bodies(N);
        createBodies(bodies.data(), N);

        std::vector<DistanceConstraint> constraints(C);
        for (uint32_t i = 0; i < C; ++i) {
            constraints[i] = DistanceConstraint(Vec3::zero(), Vec3::zero(), 1.0f, 0, i + 1);
            constraints[i].softParams = SoftConstraintParams(30.0f, 0.7f);
            constraints[i].initialize(bodies.data(), N, config, dt);
        }

        auto start = Clock::now();
        for (uint32_t rep = 0; rep < REPS; ++rep) {
            for (uint32_t iter = 0; iter < ITERS; ++iter) {
                for (uint32_t i = 0; i < C; ++i)
                    constraints[i].solveVelocity(bodies.data());
            }
        }
        auto end = Clock::now();
        benchLoop("Soft distance constraint", static_cast<uint64_t>(REPS) * ITERS * C, end - start);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Bench 6: 6-DoF vs specialized constraint overhead
// ═════════════════════════════════════════════════════════════════════════════

static void bench_six_dof_vs_specialized() {
    std::printf("\n=== Bench 6: 6-DoF vs specialized (1000) ===\n");

    constexpr uint32_t N = 1001;
    constexpr uint32_t C = 1000;
    constexpr uint32_t ITERS = 10;
    constexpr uint32_t REPS = 50;

    SolverConfig config;
    float dt = 1.0f / 60.0f;

    // Specialized hinge
    {
        std::vector<SolverBody> bodies(N);
        createBodies(bodies.data(), N);

        std::vector<HingeConstraint> constraints(C);
        for (uint32_t i = 0; i < C; ++i) {
            constraints[i] = HingeConstraint(Vec3::zero(), Vec3::zero(),
                                             Vec3::unitY(), Vec3::unitY(), 0, i + 1);
            constraints[i].initialize(bodies.data(), N, config, dt);
        }

        auto start = Clock::now();
        for (uint32_t rep = 0; rep < REPS; ++rep) {
            for (uint32_t iter = 0; iter < ITERS; ++iter) {
                for (uint32_t i = 0; i < C; ++i)
                    constraints[i].solveVelocity(bodies.data());
            }
        }
        auto end = Clock::now();
        benchLoop("Specialized hinge (5 rows)", static_cast<uint64_t>(REPS) * ITERS * C, end - start);
    }

    // 6-DoF configured as hinge (5 locked DoFs)
    {
        std::vector<SolverBody> bodies(N);
        createBodies(bodies.data(), N);

        std::vector<SixDofConstraint> constraints(C);
        for (uint32_t i = 0; i < C; ++i) {
            constraints[i] = SixDofConstraint(Vec3::zero(), Vec3::zero(), 0, i + 1);
            constraints[i].lockDof(DofIndex::TransX);
            constraints[i].lockDof(DofIndex::TransY);
            constraints[i].lockDof(DofIndex::TransZ);
            constraints[i].lockDof(DofIndex::RotX);
            constraints[i].lockDof(DofIndex::RotZ);
            // RotY free (hinge-like)
            constraints[i].initialize(bodies.data(), N, config, dt);
        }

        auto start = Clock::now();
        for (uint32_t rep = 0; rep < REPS; ++rep) {
            for (uint32_t iter = 0; iter < ITERS; ++iter) {
                for (uint32_t i = 0; i < C; ++i)
                    constraints[i].solveVelocity(bodies.data());
            }
        }
        auto end = Clock::now();
        benchLoop("6-DoF as hinge (5 rows)", static_cast<uint64_t>(REPS) * ITERS * C, end - start);
    }
}

// ═════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("=== Pulse Constraints Benchmarks ===\n");

    bench_distance_iteration();
    bench_hinge_iteration();
    bench_mixed_constraints();

    std::printf("\n=== Bench 4: Scaling ===\n");
    bench_scaling(100);
    bench_scaling(1000);
    bench_scaling(10000);

    bench_soft_vs_rigid();
    bench_six_dof_vs_specialized();

    std::printf("\n=== Benchmarks complete ===\n");
    return 0;
}
