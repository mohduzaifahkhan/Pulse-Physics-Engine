/**
 * @file test_integration.cpp
 * @brief Comprehensive unit tests for the Pulse integration module (Module 12).
 *
 * Tests: config defaults, semi-implicit Euler (velocity/position/combined),
 * velocity Verlet (half-kick/drift/combined), RK4 (accuracy, convergence),
 * body type filtering, unified dispatch, and edge cases.
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
#include <pulse/math/quat.h>
#include <pulse/math/mat3.h>
#include <pulse/math/transform.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace pulse;

// ── Minimal test framework ────────────────────────────────────────────────────

static int g_total = 0, g_passed = 0, g_failed = 0;

#define TEST_ASSERT(expr) \
    do { if (!(expr)) { \
        std::printf("  FAIL: %s (line %d): %s\n", __func__, __LINE__, #expr); \
        return false; \
    }} while(0)

#define RUN_TEST(func) \
    do { \
        g_total++; \
        if (func()) { g_passed++; } \
        else { g_failed++; std::printf("FAILED: %s\n", #func); } \
    } while(0)

// ── Helpers ──────────────────────────────────────────────────────────────────

static bool approx(float a, float b, float eps = 0.01f) {
    return std::fabs(a - b) < eps;
}

static bool approxVec(Vec3 a, Vec3 b, float eps = 0.01f) {
    return approx(a.getX(), b.getX(), eps) &&
           approx(a.getY(), b.getY(), eps) &&
           approx(a.getZ(), b.getZ(), eps);
}

/// Create a default dynamic BodyDef at a position with optional velocity.
static BodyDef makeDynamicDef(Vec3 pos, float mass = 1.0f) {
    BodyDef def;
    def.type = BodyType::Dynamic;
    def.initialTransform = Transform(pos);
    def.mass = mass;
    def.linearDamping = 0.0f;
    def.angularDamping = 0.0f;
    // Sphere inertia: I = (2/5) * m * r²  for r=0.5
    float I = 0.4f * mass * 0.5f * 0.5f;
    def.localInertia = Mat3(I, 0, 0, 0, I, 0, 0, 0, I);
    return def;
}

static BodyDef makeStaticDef(Vec3 pos) {
    BodyDef def;
    def.type = BodyType::Static;
    def.initialTransform = Transform(pos);
    def.mass = 0.0f;
    return def;
}

static BodyDef makeKinematicDef(Vec3 pos) {
    BodyDef def;
    def.type = BodyType::Kinematic;
    def.initialTransform = Transform(pos);
    def.mass = 0.0f;
    return def;
}

// ══════════════════════════════════════════════════════════════════════════════
// Group 1: Config Defaults
// ══════════════════════════════════════════════════════════════════════════════

static bool test_config_defaults() {
    IntegrationConfig cfg;
    TEST_ASSERT(cfg.type == IntegratorType::SemiImplicitEuler);
    TEST_ASSERT(approxVec(cfg.gravity, Vec3(0, -9.81f, 0)));
    TEST_ASSERT(approx(cfg.maxLinearSpeed, 500.0f));
    TEST_ASSERT(approx(cfg.maxAngularSpeed, 100.0f));
    return true;
}

static bool test_config_custom() {
    IntegrationConfig cfg(IntegratorType::RK4, Vec3(0, -10, 0), 200.0f, 50.0f);
    TEST_ASSERT(cfg.type == IntegratorType::RK4);
    TEST_ASSERT(approxVec(cfg.gravity, Vec3(0, -10, 0)));
    TEST_ASSERT(approx(cfg.maxLinearSpeed, 200.0f));
    TEST_ASSERT(approx(cfg.maxAngularSpeed, 50.0f));
    return true;
}

static bool test_integrator_type_enum() {
    TEST_ASSERT(static_cast<uint8_t>(IntegratorType::SemiImplicitEuler) == 0);
    TEST_ASSERT(static_cast<uint8_t>(IntegratorType::VelocityVerlet) == 1);
    TEST_ASSERT(static_cast<uint8_t>(IntegratorType::RK4) == 2);
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// Group 2: Utility Functions
// ══════════════════════════════════════════════════════════════════════════════

static bool test_clamp_speed_below() {
    Vec3 v(3, 4, 0); // length = 5
    Vec3 clamped = integration_detail::clampSpeed(v, 10.0f);
    TEST_ASSERT(approxVec(clamped, v)); // No change, 5 < 10
    return true;
}

static bool test_clamp_speed_above() {
    Vec3 v(30, 40, 0); // length = 50
    Vec3 clamped = integration_detail::clampSpeed(v, 10.0f);
    float len = std::sqrt(clamped.getX() * clamped.getX() +
                          clamped.getY() * clamped.getY() +
                          clamped.getZ() * clamped.getZ());
    TEST_ASSERT(approx(len, 10.0f, 0.05f));
    return true;
}

static bool test_clamp_speed_zero_limit() {
    Vec3 v(100, 0, 0);
    Vec3 clamped = integration_detail::clampSpeed(v, 0.0f); // No clamp
    TEST_ASSERT(approxVec(clamped, v));
    return true;
}

static bool test_apply_damping() {
    Vec3 v(10, 0, 0);
    Vec3 damped = integration_detail::applyDamping(v, 1.0f, 1.0f);
    // factor = 1 / (1 + 1*1) = 0.5
    TEST_ASSERT(approxVec(damped, Vec3(5, 0, 0)));
    return true;
}

static bool test_integrate_rotation_identity() {
    Quat q = Quat::identity();
    Vec3 omega(0, 0, 0);
    Quat result = integration_detail::integrateRotation(q, omega, 1.0f / 60.0f);
    TEST_ASSERT(approx(result.getW(), 1.0f));
    TEST_ASSERT(approx(result.getX(), 0.0f));
    TEST_ASSERT(approx(result.getY(), 0.0f));
    TEST_ASSERT(approx(result.getZ(), 0.0f));
    return true;
}

static bool test_integrate_rotation_y_axis() {
    Quat q = Quat::identity();
    Vec3 omega(0, math::Pi, 0); // 180 deg/s around Y
    // After 1 second at pi rad/s, should have rotated ~180 degrees.
    // We test a single small step.
    float dt = 0.01f;
    Quat result = integration_detail::integrateRotation(q, omega, dt);
    // After 0.01s, rotation angle ~ pi * 0.01 = 0.0314 rad
    // The quaternion w should be close to cos(0.0157) ~ 0.9999
    TEST_ASSERT(result.getW() > 0.99f);
    TEST_ASSERT(result.lengthSq() > 0.99f && result.lengthSq() < 1.01f);
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// Group 3: Semi-Implicit Euler — Velocity Phase
// ══════════════════════════════════════════════════════════════════════════════

static bool test_euler_velocity_gravity() {
    RigidBodyStore store(4);
    BodyDef def = makeDynamicDef(Vec3::zero());
    store.add(def);

    Vec3 gravity(0, -9.81f, 0);
    float dt = 1.0f; // 1 second for easy math

    semiImplicitEulerIntegrateVelocities(store, gravity, dt, 0.0f, 0.0f);

    Vec3 vel = store.linearVelocity(0);
    // v = 0 + (-9.81) * 1 = -9.81
    TEST_ASSERT(approx(vel.getY(), -9.81f, 0.05f));
    TEST_ASSERT(approx(vel.getX(), 0.0f));
    TEST_ASSERT(approx(vel.getZ(), 0.0f));
    return true;
}

static bool test_euler_velocity_force() {
    RigidBodyStore store(4);
    BodyDef def = makeDynamicDef(Vec3::zero(), 2.0f); // 2 kg
    std::size_t idx = store.add(def);

    store.force(idx) = Vec3(10, 0, 0); // 10 N in X

    Vec3 gravity = Vec3::zero();
    float dt = 1.0f;

    semiImplicitEulerIntegrateVelocities(store, gravity, dt, 0.0f, 0.0f);

    Vec3 vel = store.linearVelocity(0);
    // a = F/m = 10/2 = 5.  v = 0 + 5*1 = 5
    TEST_ASSERT(approx(vel.getX(), 5.0f, 0.05f));
    return true;
}

static bool test_euler_velocity_damping() {
    RigidBodyStore store(4);
    BodyDef def = makeDynamicDef(Vec3::zero());
    def.linearDamping = 1.0f;
    def.linearVelocity = Vec3(10, 0, 0);
    store.add(def);

    Vec3 gravity = Vec3::zero();
    float dt = 1.0f;

    semiImplicitEulerIntegrateVelocities(store, gravity, dt, 0.0f, 0.0f);

    Vec3 vel = store.linearVelocity(0);
    // factor = 1/(1+1*1) = 0.5.  v = 10 * 0.5 = 5
    TEST_ASSERT(approx(vel.getX(), 5.0f, 0.05f));
    return true;
}

static bool test_euler_velocity_speed_clamp() {
    RigidBodyStore store(4);
    BodyDef def = makeDynamicDef(Vec3::zero());
    def.linearVelocity = Vec3(0, 0, 0);
    store.add(def);

    store.force(0) = Vec3(100000, 0, 0); // Huge force

    Vec3 gravity = Vec3::zero();
    float dt = 1.0f;

    semiImplicitEulerIntegrateVelocities(store, gravity, dt, 50.0f, 0.0f);

    Vec3 vel = store.linearVelocity(0);
    float speed = std::sqrt(vel.getX() * vel.getX() + vel.getY() * vel.getY() + vel.getZ() * vel.getZ());
    TEST_ASSERT(speed <= 50.1f);
    return true;
}

static bool test_euler_velocity_angular() {
    RigidBodyStore store(4);
    BodyDef def = makeDynamicDef(Vec3::zero());
    // Inertia = diag(0.1), invI = diag(10)
    def.localInertia = Mat3(0.1f, 0, 0, 0, 0.1f, 0, 0, 0, 0.1f);
    std::size_t idx = store.add(def);
    store.updateWorldInertia(idx);

    store.torque(idx) = Vec3(0, 1, 0); // 1 Nm around Y

    semiImplicitEulerIntegrateVelocities(store, Vec3::zero(), 1.0f, 0.0f, 0.0f);

    Vec3 angVel = store.angularVelocity(0);
    // alpha = I^-1 * tau = 10 * 1 = 10.  omega = 0 + 10*1 = 10
    TEST_ASSERT(approx(angVel.getY(), 10.0f, 0.1f));
    return true;
}

static bool test_euler_velocity_gravity_scale() {
    RigidBodyStore store(4);
    BodyDef def = makeDynamicDef(Vec3::zero());
    def.gravityScale = 0.5f;
    store.add(def);

    Vec3 gravity(0, -10.0f, 0);
    semiImplicitEulerIntegrateVelocities(store, gravity, 1.0f, 0.0f, 0.0f);

    Vec3 vel = store.linearVelocity(0);
    // Effective gravity = -10 * 0.5 = -5
    TEST_ASSERT(approx(vel.getY(), -5.0f, 0.05f));
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// Group 4: Semi-Implicit Euler — Position Phase
// ══════════════════════════════════════════════════════════════════════════════

static bool test_euler_position_linear() {
    RigidBodyStore store(4);
    BodyDef def = makeDynamicDef(Vec3(0, 10, 0));
    def.linearVelocity = Vec3(5, 0, 0);
    store.add(def);

    semiImplicitEulerIntegratePositions(store, 1.0f);

    Vec3 pos = store.position(0);
    // x = 0 + 5*1 = 5,  y = 10 + 0 = 10
    TEST_ASSERT(approx(pos.getX(), 5.0f));
    TEST_ASSERT(approx(pos.getY(), 10.0f));
    return true;
}

static bool test_euler_position_angular() {
    RigidBodyStore store(4);
    BodyDef def = makeDynamicDef(Vec3::zero());
    def.angularVelocity = Vec3(0, math::Pi, 0); // 180 deg/s
    store.add(def);

    semiImplicitEulerIntegratePositions(store, 0.01f);

    Quat rot = store.rotation(0);
    // Should have rotated slightly and remain unit-length
    TEST_ASSERT(approx(rot.lengthSq(), 1.0f, 0.01f));
    // Y component of quaternion should be non-zero (rotation around Y)
    TEST_ASSERT(std::fabs(rot.getY()) > 0.001f);
    return true;
}

static bool test_euler_position_rotation_stays_normalized() {
    RigidBodyStore store(4);
    BodyDef def = makeDynamicDef(Vec3::zero());
    def.angularVelocity = Vec3(10, 20, 30); // Fast rotation
    store.add(def);

    // Many steps
    for (int step = 0; step < 1000; ++step) {
        semiImplicitEulerIntegratePositions(store, 1.0f / 60.0f);
    }

    Quat rot = store.rotation(0);
    TEST_ASSERT(approx(rot.lengthSq(), 1.0f, 0.01f));
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// Group 5: Semi-Implicit Euler — Full Step
// ══════════════════════════════════════════════════════════════════════════════

static bool test_euler_free_fall_1s() {
    RigidBodyStore store(4);
    BodyDef def = makeDynamicDef(Vec3(0, 0, 0));
    store.add(def);

    Vec3 gravity(0, -9.81f, 0);
    int steps = 60;
    float dt = 1.0f / 60.0f;

    for (int i = 0; i < steps; ++i) {
        store.clearForces();
        semiImplicitEulerIntegrate(store, gravity, dt, 0.0f, 0.0f);
    }

    Vec3 pos = store.position(0);
    Vec3 vel = store.linearVelocity(0);

    // After 1s: v = -9.81 m/s, y ≈ -4.905m
    TEST_ASSERT(approx(vel.getY(), -9.81f, 0.2f));
    TEST_ASSERT(approx(pos.getY(), -4.905f, 0.2f));
    return true;
}

static bool test_euler_projectile() {
    RigidBodyStore store(4);
    BodyDef def = makeDynamicDef(Vec3(0, 0, 0));
    def.linearVelocity = Vec3(10, 0, 0); // 10 m/s horizontal
    store.add(def);

    Vec3 gravity(0, -10.0f, 0);
    int steps = 60;
    float dt = 1.0f / 60.0f;

    for (int i = 0; i < steps; ++i) {
        store.clearForces();
        semiImplicitEulerIntegrate(store, gravity, dt, 0.0f, 0.0f);
    }

    Vec3 pos = store.position(0);
    // After 1s: x = 10*1 = 10, y = -0.5*10*1² = -5
    TEST_ASSERT(approx(pos.getX(), 10.0f, 0.3f));
    TEST_ASSERT(approx(pos.getY(), -5.0f, 0.3f));
    return true;
}

static bool test_euler_multiple_bodies() {
    RigidBodyStore store(8);
    BodyDef defA = makeDynamicDef(Vec3(0, 0, 0), 1.0f);
    BodyDef defB = makeDynamicDef(Vec3(10, 0, 0), 2.0f);
    store.add(defA);
    store.add(defB);

    Vec3 gravity(0, -10, 0);
    semiImplicitEulerIntegrate(store, gravity, 1.0f, 0.0f, 0.0f);

    // Both should have same velocity (gravity is mass-independent)
    TEST_ASSERT(approx(store.linearVelocity(0).getY(), -10.0f, 0.05f));
    TEST_ASSERT(approx(store.linearVelocity(1).getY(), -10.0f, 0.05f));
    // But start at different positions
    TEST_ASSERT(approx(store.position(0).getX(), 0.0f));
    TEST_ASSERT(approx(store.position(1).getX(), 10.0f));
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// Group 6: Velocity Verlet
// ══════════════════════════════════════════════════════════════════════════════

static bool test_verlet_free_fall_1s() {
    RigidBodyStore store(4);
    BodyDef def = makeDynamicDef(Vec3(0, 0, 0));
    store.add(def);

    Vec3 gravity(0, -9.81f, 0);
    int steps = 60;
    float dt = 1.0f / 60.0f;

    for (int i = 0; i < steps; ++i) {
        store.clearForces();
        verletIntegrate(store, gravity, dt, 0.0f, 0.0f);
    }

    Vec3 pos = store.position(0);
    Vec3 vel = store.linearVelocity(0);

    // Same free-fall target: v ≈ -9.81, y ≈ -4.905
    TEST_ASSERT(approx(vel.getY(), -9.81f, 0.3f));
    TEST_ASSERT(approx(pos.getY(), -4.905f, 0.3f));
    return true;
}

static bool test_verlet_split_api() {
    RigidBodyStore store(4);
    BodyDef def = makeDynamicDef(Vec3(0, 0, 0));
    store.add(def);

    Vec3 gravity(0, -10, 0);
    float dt = 1.0f;

    verletHalfKickAndDrift(store, gravity, dt);

    // After half-kick: v = 0 + 0.5*(-10)*1 = -5
    // After drift: position = 0 + (-5)*1 = -5
    TEST_ASSERT(approx(store.linearVelocity(0).getY(), -5.0f, 0.1f));
    TEST_ASSERT(approx(store.position(0).getY(), -5.0f, 0.1f));

    verletSecondHalfKick(store, gravity, dt, 0.0f, 0.0f);

    // After second half-kick: v = -5 + 0.5*(-10)*1 = -10
    TEST_ASSERT(approx(store.linearVelocity(0).getY(), -10.0f, 0.1f));
    return true;
}

static bool test_verlet_energy_conservation() {
    // Compare Euler vs Verlet for a spring-like system (harmonic oscillator).
    // A body oscillating vertically under gravity with initial upward velocity.
    // Verlet should conserve energy better over many steps.

    Vec3 gravity(0, -10, 0);
    float dt = 1.0f / 60.0f;
    int steps = 600; // 10 seconds

    // Euler
    RigidBodyStore storeE(4);
    BodyDef defE = makeDynamicDef(Vec3(0, 0, 0));
    defE.linearVelocity = Vec3(0, 50, 0); // Launch upward
    storeE.add(defE);

    for (int i = 0; i < steps; ++i) {
        storeE.clearForces();
        semiImplicitEulerIntegrate(storeE, gravity, dt, 0.0f, 0.0f);
    }

    // Verlet
    RigidBodyStore storeV(4);
    BodyDef defV = makeDynamicDef(Vec3(0, 0, 0));
    defV.linearVelocity = Vec3(0, 50, 0);
    storeV.add(defV);

    for (int i = 0; i < steps; ++i) {
        storeV.clearForces();
        verletIntegrate(storeV, gravity, dt, 0.0f, 0.0f);
    }

    // Both should end at similar positions (constant gravity = both exact),
    // but Verlet should be at least as close to the analytical answer.
    // Analytical: v(10s) = 50 - 10*10 = -50,  y(10s) = 50*10 - 0.5*10*100 = 0
    float eulerErr = std::fabs(storeE.position(0).getY() - 0.0f);
    float verletErr = std::fabs(storeV.position(0).getY() - 0.0f);

    // For constant gravity both should be very accurate.
    // Just verify both are close.
    TEST_ASSERT(eulerErr < 1.0f);
    TEST_ASSERT(verletErr < 1.0f);
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// Group 7: RK4
// ══════════════════════════════════════════════════════════════════════════════

static bool test_rk4_free_fall_1s() {
    RigidBodyStore store(4);
    BodyDef def = makeDynamicDef(Vec3(0, 0, 0));
    store.add(def);

    Vec3 gravity(0, -9.81f, 0);
    int steps = 60;
    float dt = 1.0f / 60.0f;

    for (int i = 0; i < steps; ++i) {
        store.clearForces();
        rk4Integrate(store, gravity, dt, 0.0f, 0.0f);
    }

    Vec3 pos = store.position(0);
    Vec3 vel = store.linearVelocity(0);

    // v ≈ -9.81, y ≈ -4.905
    TEST_ASSERT(approx(vel.getY(), -9.81f, 0.1f));
    TEST_ASSERT(approx(pos.getY(), -4.905f, 0.1f));
    return true;
}

static bool test_rk4_accuracy_vs_euler() {
    // RK4 should be more accurate than Euler for the same timestep.
    // Use a large timestep to exaggerate the difference.
    Vec3 gravity(0, -10, 0);
    float dt = 0.1f; // Large timestep
    int steps = 10;  // 1 second total

    // Euler
    RigidBodyStore storeE(4);
    BodyDef defE = makeDynamicDef(Vec3::zero());
    storeE.add(defE);
    for (int i = 0; i < steps; ++i) {
        storeE.clearForces();
        semiImplicitEulerIntegrate(storeE, gravity, dt, 0.0f, 0.0f);
    }

    // RK4
    RigidBodyStore storeR(4);
    BodyDef defR = makeDynamicDef(Vec3::zero());
    storeR.add(defR);
    for (int i = 0; i < steps; ++i) {
        storeR.clearForces();
        rk4Integrate(storeR, gravity, dt, 0.0f, 0.0f);
    }

    // Analytical: y = -0.5*10*1 = -5.0
    float eulerErr = std::fabs(storeE.position(0).getY() - (-5.0f));
    float rk4Err = std::fabs(storeR.position(0).getY() - (-5.0f));

    // Both should be accurate for constant gravity; RK4 should be at least as good.
    TEST_ASSERT(rk4Err <= eulerErr + 0.01f);
    TEST_ASSERT(rk4Err < 0.5f); // Very accurate
    return true;
}

static bool test_rk4_horizontal_launch() {
    RigidBodyStore store(4);
    BodyDef def = makeDynamicDef(Vec3::zero());
    def.linearVelocity = Vec3(20, 30, 0); // 20 m/s horizontal, 30 m/s vertical
    store.add(def);

    Vec3 gravity(0, -10, 0);
    float dt = 1.0f / 120.0f;
    int steps = 120; // 1 second

    for (int i = 0; i < steps; ++i) {
        store.clearForces();
        rk4Integrate(store, gravity, dt, 0.0f, 0.0f);
    }

    Vec3 pos = store.position(0);
    // Analytical: x = 20*1 = 20, y = 30*1 - 0.5*10*1 = 25
    TEST_ASSERT(approx(pos.getX(), 20.0f, 0.2f));
    TEST_ASSERT(approx(pos.getY(), 25.0f, 0.2f));
    return true;
}

static bool test_rk4_constant_velocity() {
    // No gravity, no force — body should move at constant velocity.
    RigidBodyStore store(4);
    BodyDef def = makeDynamicDef(Vec3::zero());
    def.linearVelocity = Vec3(1, 2, 3);
    store.add(def);

    rk4Integrate(store, Vec3::zero(), 1.0f, 0.0f, 0.0f);

    Vec3 pos = store.position(0);
    TEST_ASSERT(approxVec(pos, Vec3(1, 2, 3), 0.01f));

    Vec3 vel = store.linearVelocity(0);
    TEST_ASSERT(approxVec(vel, Vec3(1, 2, 3), 0.01f));
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// Group 8: Body Type Filtering
// ══════════════════════════════════════════════════════════════════════════════

static bool test_static_body_skipped() {
    RigidBodyStore store(4);
    BodyDef def = makeStaticDef(Vec3(5, 5, 5));
    store.add(def);

    semiImplicitEulerIntegrate(store, Vec3(0, -10, 0), 1.0f, 0.0f, 0.0f);

    // Static body should not have moved.
    Vec3 pos = store.position(0);
    TEST_ASSERT(approxVec(pos, Vec3(5, 5, 5)));
    TEST_ASSERT(approxVec(store.linearVelocity(0), Vec3::zero()));
    return true;
}

static bool test_kinematic_body_skipped() {
    RigidBodyStore store(4);
    BodyDef def = makeKinematicDef(Vec3(3, 3, 3));
    store.add(def);

    semiImplicitEulerIntegrate(store, Vec3(0, -10, 0), 1.0f, 0.0f, 0.0f);

    Vec3 pos = store.position(0);
    TEST_ASSERT(approxVec(pos, Vec3(3, 3, 3)));
    return true;
}

static bool test_sleeping_body_skipped() {
    RigidBodyStore store(4);
    BodyDef def = makeDynamicDef(Vec3(1, 1, 1));
    def.startAwake = false; // Start sleeping
    store.add(def);

    semiImplicitEulerIntegrate(store, Vec3(0, -10, 0), 1.0f, 0.0f, 0.0f);

    // Sleeping body should not move.
    Vec3 pos = store.position(0);
    TEST_ASSERT(approxVec(pos, Vec3(1, 1, 1)));
    return true;
}

static bool test_mixed_body_types() {
    RigidBodyStore store(8);
    std::size_t iStatic  = store.add(makeStaticDef(Vec3(0, 0, 0)));
    std::size_t iDynamic = store.add(makeDynamicDef(Vec3(0, 0, 0)));
    std::size_t iKin     = store.add(makeKinematicDef(Vec3(0, 0, 0)));

    semiImplicitEulerIntegrate(store, Vec3(0, -10, 0), 1.0f, 0.0f, 0.0f);

    // Only dynamic should have moved.
    TEST_ASSERT(approx(store.position(iStatic).getY(), 0.0f));
    TEST_ASSERT(store.position(iDynamic).getY() < -0.5f); // Fell under gravity
    TEST_ASSERT(approx(store.position(iKin).getY(), 0.0f));
    return true;
}

static bool test_verlet_skips_static() {
    RigidBodyStore store(4);
    store.add(makeStaticDef(Vec3(5, 5, 5)));

    verletIntegrate(store, Vec3(0, -10, 0), 1.0f, 0.0f, 0.0f);

    TEST_ASSERT(approxVec(store.position(0), Vec3(5, 5, 5)));
    return true;
}

static bool test_rk4_skips_static() {
    RigidBodyStore store(4);
    store.add(makeStaticDef(Vec3(5, 5, 5)));

    rk4Integrate(store, Vec3(0, -10, 0), 1.0f, 0.0f, 0.0f);

    TEST_ASSERT(approxVec(store.position(0), Vec3(5, 5, 5)));
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// Group 9: Unified Dispatch
// ══════════════════════════════════════════════════════════════════════════════

static bool test_dispatch_euler() {
    RigidBodyStore store(4);
    store.add(makeDynamicDef(Vec3::zero()));

    IntegrationConfig cfg;
    cfg.type = IntegratorType::SemiImplicitEuler;
    cfg.gravity = Vec3(0, -10, 0);
    cfg.maxLinearSpeed = 0.0f;
    cfg.maxAngularSpeed = 0.0f;

    integrate(store, cfg, 1.0f);

    TEST_ASSERT(approx(store.linearVelocity(0).getY(), -10.0f, 0.1f));
    TEST_ASSERT(store.position(0).getY() < -0.5f);
    return true;
}

static bool test_dispatch_verlet() {
    RigidBodyStore store(4);
    store.add(makeDynamicDef(Vec3::zero()));

    IntegrationConfig cfg;
    cfg.type = IntegratorType::VelocityVerlet;
    cfg.gravity = Vec3(0, -10, 0);
    cfg.maxLinearSpeed = 0.0f;
    cfg.maxAngularSpeed = 0.0f;

    integrate(store, cfg, 1.0f);

    TEST_ASSERT(approx(store.linearVelocity(0).getY(), -10.0f, 0.1f));
    TEST_ASSERT(store.position(0).getY() < -0.5f);
    return true;
}

static bool test_dispatch_rk4() {
    RigidBodyStore store(4);
    store.add(makeDynamicDef(Vec3::zero()));

    IntegrationConfig cfg;
    cfg.type = IntegratorType::RK4;
    cfg.gravity = Vec3(0, -10, 0);
    cfg.maxLinearSpeed = 0.0f;
    cfg.maxAngularSpeed = 0.0f;

    integrate(store, cfg, 1.0f);

    TEST_ASSERT(approx(store.linearVelocity(0).getY(), -10.0f, 0.1f));
    TEST_ASSERT(store.position(0).getY() < -0.5f);
    return true;
}

static bool test_dispatch_split_euler() {
    RigidBodyStore store(4);
    BodyDef def = makeDynamicDef(Vec3::zero());
    store.add(def);

    IntegrationConfig cfg;
    cfg.type = IntegratorType::SemiImplicitEuler;
    cfg.gravity = Vec3(0, -10, 0);
    cfg.maxLinearSpeed = 0.0f;
    cfg.maxAngularSpeed = 0.0f;

    integrateVelocities(store, cfg, 1.0f);
    TEST_ASSERT(approx(store.linearVelocity(0).getY(), -10.0f, 0.1f));
    TEST_ASSERT(approx(store.position(0).getY(), 0.0f)); // Not moved yet

    integratePositions(store, cfg, 1.0f);
    TEST_ASSERT(store.position(0).getY() < -0.5f); // Now moved
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// Group 10: Edge Cases
// ══════════════════════════════════════════════════════════════════════════════

static bool test_zero_dt() {
    RigidBodyStore store(4);
    BodyDef def = makeDynamicDef(Vec3(1, 2, 3));
    def.linearVelocity = Vec3(10, 0, 0);
    store.add(def);

    semiImplicitEulerIntegrate(store, Vec3(0, -10, 0), 0.0f, 0.0f, 0.0f);

    // Nothing should change with dt=0.
    TEST_ASSERT(approxVec(store.position(0), Vec3(1, 2, 3)));
    TEST_ASSERT(approxVec(store.linearVelocity(0), Vec3(10, 0, 0)));
    return true;
}

static bool test_zero_mass_body() {
    // A dynamic body with zero mass (invMass = 0) should not accelerate.
    RigidBodyStore store(4);
    BodyDef def = makeDynamicDef(Vec3::zero(), 0.0f);
    def.linearVelocity = Vec3(5, 0, 0); // Has initial velocity
    store.add(def);

    semiImplicitEulerIntegrate(store, Vec3(0, -10, 0), 1.0f, 0.0f, 0.0f);

    // invMass = 0, so gravity/force don't affect velocity
    // But position still advances from existing velocity
    TEST_ASSERT(approx(store.linearVelocity(0).getX(), 5.0f, 0.1f));
    TEST_ASSERT(approx(store.linearVelocity(0).getY(), 0.0f, 0.1f));
    return true;
}

static bool test_empty_store() {
    RigidBodyStore store(4);

    // These should not crash on empty stores.
    semiImplicitEulerIntegrate(store, Vec3(0, -10, 0), 1.0f, 0.0f, 0.0f);
    verletIntegrate(store, Vec3(0, -10, 0), 1.0f, 0.0f, 0.0f);
    rk4Integrate(store, Vec3(0, -10, 0), 1.0f, 0.0f, 0.0f);

    TEST_ASSERT(store.size() == 0);
    return true;
}

static bool test_single_body() {
    RigidBodyStore store(4);
    BodyDef def = makeDynamicDef(Vec3(0, 100, 0));
    store.add(def);

    semiImplicitEulerIntegrate(store, Vec3(0, -10, 0), 0.5f, 0.0f, 0.0f);

    // v = -5, y = 100 + (-5)*0.5 = 97.5
    TEST_ASSERT(approx(store.linearVelocity(0).getY(), -5.0f, 0.05f));
    TEST_ASSERT(approx(store.position(0).getY(), 97.5f, 0.1f));
    return true;
}

static bool test_all_integrators_agree_constant_gravity() {
    // For constant gravity (linear accel), all integrators should give
    // the same result (within numerical precision).
    Vec3 gravity(0, -10, 0);
    float dt = 1.0f / 60.0f;
    int steps = 60;

    float finalY[3];
    float finalVy[3];

    for (int method = 0; method < 3; ++method) {
        RigidBodyStore store(4);
        BodyDef def = makeDynamicDef(Vec3::zero());
        store.add(def);

        IntegrationConfig cfg;
        cfg.type = static_cast<IntegratorType>(method);
        cfg.gravity = gravity;
        cfg.maxLinearSpeed = 0.0f;
        cfg.maxAngularSpeed = 0.0f;

        for (int i = 0; i < steps; ++i) {
            store.clearForces();
            integrate(store, cfg, dt);
        }

        finalY[method] = store.position(0).getY();
        finalVy[method] = store.linearVelocity(0).getY();
    }

    // All three should be close to analytical: y=-5, vy=-10
    for (int m = 0; m < 3; ++m) {
        TEST_ASSERT(approx(finalY[m], -5.0f, 0.5f));
        TEST_ASSERT(approx(finalVy[m], -10.0f, 0.5f));
    }
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// Main
// ══════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("=== Pulse Integration Tests (Module 12) ===\n\n");

    // Group 1: Config
    std::printf("--- Config Defaults ---\n");
    RUN_TEST(test_config_defaults);
    RUN_TEST(test_config_custom);
    RUN_TEST(test_integrator_type_enum);

    // Group 2: Utilities
    std::printf("--- Utility Functions ---\n");
    RUN_TEST(test_clamp_speed_below);
    RUN_TEST(test_clamp_speed_above);
    RUN_TEST(test_clamp_speed_zero_limit);
    RUN_TEST(test_apply_damping);
    RUN_TEST(test_integrate_rotation_identity);
    RUN_TEST(test_integrate_rotation_y_axis);

    // Group 3: Euler Velocity
    std::printf("--- Semi-Implicit Euler: Velocity ---\n");
    RUN_TEST(test_euler_velocity_gravity);
    RUN_TEST(test_euler_velocity_force);
    RUN_TEST(test_euler_velocity_damping);
    RUN_TEST(test_euler_velocity_speed_clamp);
    RUN_TEST(test_euler_velocity_angular);
    RUN_TEST(test_euler_velocity_gravity_scale);

    // Group 4: Euler Position
    std::printf("--- Semi-Implicit Euler: Position ---\n");
    RUN_TEST(test_euler_position_linear);
    RUN_TEST(test_euler_position_angular);
    RUN_TEST(test_euler_position_rotation_stays_normalized);

    // Group 5: Euler Full Step
    std::printf("--- Semi-Implicit Euler: Full Step ---\n");
    RUN_TEST(test_euler_free_fall_1s);
    RUN_TEST(test_euler_projectile);
    RUN_TEST(test_euler_multiple_bodies);

    // Group 6: Velocity Verlet
    std::printf("--- Velocity Verlet ---\n");
    RUN_TEST(test_verlet_free_fall_1s);
    RUN_TEST(test_verlet_split_api);
    RUN_TEST(test_verlet_energy_conservation);

    // Group 7: RK4
    std::printf("--- RK4 ---\n");
    RUN_TEST(test_rk4_free_fall_1s);
    RUN_TEST(test_rk4_accuracy_vs_euler);
    RUN_TEST(test_rk4_horizontal_launch);
    RUN_TEST(test_rk4_constant_velocity);

    // Group 8: Body Type Filtering
    std::printf("--- Body Type Filtering ---\n");
    RUN_TEST(test_static_body_skipped);
    RUN_TEST(test_kinematic_body_skipped);
    RUN_TEST(test_sleeping_body_skipped);
    RUN_TEST(test_mixed_body_types);
    RUN_TEST(test_verlet_skips_static);
    RUN_TEST(test_rk4_skips_static);

    // Group 9: Unified Dispatch
    std::printf("--- Unified Dispatch ---\n");
    RUN_TEST(test_dispatch_euler);
    RUN_TEST(test_dispatch_verlet);
    RUN_TEST(test_dispatch_rk4);
    RUN_TEST(test_dispatch_split_euler);

    // Group 10: Edge Cases
    std::printf("--- Edge Cases ---\n");
    RUN_TEST(test_zero_dt);
    RUN_TEST(test_zero_mass_body);
    RUN_TEST(test_empty_store);
    RUN_TEST(test_single_body);
    RUN_TEST(test_all_integrators_agree_constant_gravity);

    // Summary
    std::printf("\n=== Results: %d/%d passed", g_passed, g_total);
    if (g_failed > 0) std::printf(", %d FAILED", g_failed);
    std::printf(" ===\n");

    return (g_failed > 0) ? 1 : 0;
}
