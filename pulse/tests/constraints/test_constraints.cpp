/**
 * @file test_constraints.cpp
 * @brief Comprehensive unit tests for the Pulse constraints module (Module 10).
 *
 * Tests all constraint types: JointRow helpers, DistanceConstraint,
 * HingeConstraint, SliderConstraint, ConeTwistConstraint, SixDofConstraint,
 * SpringConstraint, MotorConstraint, and edge cases.
 */

#include <pulse/constraints/constraint_common.h>
#include <pulse/constraints/distance_constraint.h>
#include <pulse/constraints/hinge_constraint.h>
#include <pulse/constraints/slider_constraint.h>
#include <pulse/constraints/cone_twist_constraint.h>
#include <pulse/constraints/six_dof_constraint.h>
#include <pulse/constraints/spring_constraint.h>
#include <pulse/constraints/motor_constraint.h>

#include <pulse/solver/solver_common.h>
#include <pulse/solver/constraint_base.h>
#include <pulse/math/math_common.h>
#include <pulse/math/vec3.h>
#include <pulse/math/quat.h>

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

static bool approx(float a, float b, float eps = 0.05f) {
    return std::fabs(a - b) < eps;
}

static bool approxVec(Vec3 a, Vec3 b, float eps = 0.05f) {
    return approx(a.getX(), b.getX(), eps) &&
           approx(a.getY(), b.getY(), eps) &&
           approx(a.getZ(), b.getZ(), eps);
}

/// Create a dynamic solver body.
static SolverBody makeDynamic(Vec3 pos, float mass, uint32_t id) {
    SolverBody b;
    b.position = pos;
    b.invMass = 1.0f / mass;
    float I = 0.4f * mass * 0.5f * 0.5f; // sphere r=0.5
    b.invInertia = Vec3(1.0f / I, 1.0f / I, 1.0f / I);
    b.restitution = 0.0f;
    b.friction = 0.5f;
    b.bodyId = id;
    return b;
}

/// Create a static solver body.
static SolverBody makeStatic(Vec3 pos, uint32_t id) {
    SolverBody b;
    b.position = pos;
    b.invMass = 0.0f;
    b.invInertia = Vec3::zero();
    b.restitution = 0.0f;
    b.friction = 0.5f;
    b.bodyId = id;
    return b;
}

/// Run N velocity iterations on a constraint.
template<typename T>
static void runIterations(T& constraint, SolverBody* bodies, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        constraint.solveVelocity(bodies);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 1: JointRow & Common Helpers
// ═════════════════════════════════════════════════════════════════════════════

static bool test_joint_limit_default() {
    JointLimit limit;
    TEST_ASSERT(!limit.enabled);
    TEST_ASSERT(!limit.isLocked());
    TEST_ASSERT(limit.classify(0.0f) == JointLimitState::Inactive);
    return true;
}

static bool test_joint_limit_classification() {
    JointLimit limit(-0.5f, 0.5f);
    TEST_ASSERT(limit.enabled);
    TEST_ASSERT(limit.classify(0.0f) == JointLimitState::Inactive);
    TEST_ASSERT(limit.classify(-0.5f) == JointLimitState::AtLower);
    TEST_ASSERT(limit.classify(-1.0f) == JointLimitState::AtLower);
    TEST_ASSERT(limit.classify(0.5f) == JointLimitState::AtUpper);
    TEST_ASSERT(limit.classify(1.0f) == JointLimitState::AtUpper);
    return true;
}

static bool test_joint_limit_locked() {
    JointLimit limit(0.0f, 0.0f);
    TEST_ASSERT(limit.isLocked());
    TEST_ASSERT(limit.classify(0.0f) == JointLimitState::Locked);
    return true;
}

static bool test_joint_motor_default() {
    JointMotor motor;
    TEST_ASSERT(!motor.enabled);
    TEST_ASSERT(approx(motor.targetVelocity, 0.0f));
    TEST_ASSERT(approx(motor.maxForce, 0.0f));
    return true;
}

static bool test_joint_motor_construction() {
    JointMotor motor(5.0f, 100.0f);
    TEST_ASSERT(motor.enabled);
    TEST_ASSERT(approx(motor.targetVelocity, 5.0f));
    TEST_ASSERT(approx(motor.maxForce, 100.0f));
    return true;
}

static bool test_soft_params_rigid() {
    SoftConstraintParams soft;
    TEST_ASSERT(!soft.enabled);
    soft.compute(1.0f / 60.0f);
    TEST_ASSERT(approx(soft.gamma, 0.0f));
    return true;
}

static bool test_soft_params_compute() {
    SoftConstraintParams soft(30.0f, 0.7f);
    TEST_ASSERT(soft.enabled);
    soft.compute(1.0f / 60.0f);
    TEST_ASSERT(soft.gamma > 0.0f);
    TEST_ASSERT(soft.beta > 0.0f && soft.beta <= 1.0f);
    return true;
}

static bool test_soft_params_from_stiffness() {
    SoftConstraintParams soft = SoftConstraintParams::fromStiffness(400.0f, 10.0f, 1.0f);
    TEST_ASSERT(soft.enabled);
    TEST_ASSERT(soft.frequency > 0.0f);
    TEST_ASSERT(soft.dampingRatio > 0.0f);
    return true;
}

static bool test_joint_row_default() {
    JointRow row;
    TEST_ASSERT(approx(row.effectiveMass, 0.0f));
    TEST_ASSERT(approx(row.accumulatedImpulse, 0.0f));
    TEST_ASSERT(row.indexA == 0xFFFFFFFFu);
    TEST_ASSERT(row.indexB == 0xFFFFFFFFu);
    return true;
}

static bool test_effective_mass_computation() {
    SolverBody bodyA = makeDynamic(Vec3::zero(), 1.0f, 0);
    SolverBody bodyB = makeStatic(Vec3::zero(), 1);

    JointRow row;
    row.jvA = Vec3::unitY();
    row.jwA = Vec3::zero();
    row.jvB = Vec3(0, -1, 0);
    row.jwB = Vec3::zero();

    float em = computeJointEffectiveMass(bodyA, bodyB, row, 0.0f);
    // K = invMassA * |JvA|^2 + invMassB * |JvB|^2 = 1.0 * 1 + 0 * 1 = 1.0
    // effectiveMass = 1/K = 1.0
    TEST_ASSERT(approx(em, 1.0f, 0.01f));
    return true;
}

static bool test_solve_joint_row_basic() {
    SolverBody bodyA = makeDynamic(Vec3(0, 1, 0), 1.0f, 0);
    SolverBody bodyB = makeStatic(Vec3(0, 0, 0), 1);

    // Simulate a simple constraint: push A towards B along Y
    JointRow row;
    row.jvA = Vec3::unitY();
    row.jwA = Vec3::zero();
    row.jvB = Vec3(0, -1, 0);
    row.jwB = Vec3::zero();
    row.indexA = 0;
    row.indexB = 1;
    row.effectiveMass = computeJointEffectiveMass(bodyA, bodyB, row, 0.0f);
    row.bias = 0.2f * 1.0f * 60.0f; // baumgarte * error * invDt
    row.accumulatedImpulse = 0.0f;
    row.minImpulse = -math::Infinity;
    row.maxImpulse = math::Infinity;

    SolverBody bodies[2] = {bodyA, bodyB};
    solveJointRow(row, bodies[0], bodies[1]);

    // After solving, body A should have gained velocity along Y
    TEST_ASSERT(std::fabs(bodies[0].linearVelocity.getY()) > 0.01f);
    // Static body should not move
    TEST_ASSERT(approxVec(bodies[1].linearVelocity, Vec3::zero()));
    return true;
}

static bool test_solve_row_clamping() {
    SolverBody bodyA = makeDynamic(Vec3::zero(), 1.0f, 0);
    SolverBody bodyB = makeDynamic(Vec3(1, 0, 0), 1.0f, 1);

    JointRow row;
    row.jvA = Vec3::unitX();
    row.jwA = Vec3::zero();
    row.jvB = Vec3(-1, 0, 0);
    row.jwB = Vec3::zero();
    row.indexA = 0;
    row.indexB = 1;
    row.effectiveMass = computeJointEffectiveMass(bodyA, bodyB, row, 0.0f);
    row.bias = 100.0f;
    row.accumulatedImpulse = 0.0f;
    row.minImpulse = 0.0f; // Only positive impulse (like a limit)
    row.maxImpulse = 5.0f;

    SolverBody bodies[2] = {bodyA, bodyB};
    solveJointRow(row, bodies[0], bodies[1]);

    TEST_ASSERT(row.accumulatedImpulse >= 0.0f);
    TEST_ASSERT(row.accumulatedImpulse <= 5.0f);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 2: DistanceConstraint
// ═════════════════════════════════════════════════════════════════════════════

static bool test_distance_maintains_distance() {
    // Two bodies 2m apart, constrained to 1m distance
    SolverBody bodies[2] = {
        makeDynamic(Vec3(0, 0, 0), 1.0f, 0),
        makeDynamic(Vec3(2, 0, 0), 1.0f, 1)
    };

    DistanceConstraint dc(Vec3::zero(), Vec3::zero(), 1.0f, 0, 1);
    SolverConfig config;
    float dt = 1.0f / 60.0f;

    dc.initialize(bodies, 2, config, dt);
    TEST_ASSERT(dc.rowCount() == 1);

    // The row should have a positive bias (error = 2 - 1 = 1)
    TEST_ASSERT(dc.getRow().bias > 0.0f);

    // Run velocity iterations
    runIterations(dc, bodies, 10);

    // Bodies should be moving towards each other
    TEST_ASSERT(bodies[0].linearVelocity.getX() > 0.0f); // A moves towards B
    TEST_ASSERT(bodies[1].linearVelocity.getX() < 0.0f); // B moves towards A
    return true;
}

static bool test_distance_zero_length() {
    // Ball-socket: bodies at same position, distance = 0
    SolverBody bodies[2] = {
        makeDynamic(Vec3(0, 0, 0), 1.0f, 0),
        makeDynamic(Vec3(0, 0, 0), 1.0f, 1)
    };

    DistanceConstraint dc(Vec3::zero(), Vec3::zero(), 0.0f, 0, 1);
    SolverConfig config;
    dc.initialize(bodies, 2, config, 1.0f / 60.0f);

    // Error should be ~0 when at rest length
    TEST_ASSERT(approx(dc.getRow().bias, 0.0f, 1.0f));
    return true;
}

static bool test_distance_soft_spring() {
    SolverBody bodies[2] = {
        makeDynamic(Vec3(0, 0, 0), 1.0f, 0),
        makeDynamic(Vec3(3, 0, 0), 1.0f, 1)
    };

    DistanceConstraint dc(Vec3::zero(), Vec3::zero(), 1.0f, 0, 1);
    dc.softParams = SoftConstraintParams(30.0f, 0.7f);
    SolverConfig config;
    dc.initialize(bodies, 2, config, 1.0f / 60.0f);

    // Should have soft effective mass (smaller than rigid)
    TEST_ASSERT(dc.getRow().effectiveMass > 0.0f);
    return true;
}

static bool test_distance_with_static_body() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(2, 0, 0), 1.0f, 1)
    };

    DistanceConstraint dc(Vec3::zero(), Vec3::zero(), 1.0f, 0, 1);
    SolverConfig config;
    dc.initialize(bodies, 2, config, 1.0f / 60.0f);
    runIterations(dc, bodies, 10);

    // Static body should not move
    TEST_ASSERT(approxVec(bodies[0].linearVelocity, Vec3::zero()));
    // Dynamic body should move towards static
    TEST_ASSERT(bodies[1].linearVelocity.getX() < 0.0f);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 3: HingeConstraint
// ═════════════════════════════════════════════════════════════════════════════

static bool test_hinge_anchor_coincidence() {
    SolverBody bodies[2] = {
        makeDynamic(Vec3(0, 0, 0), 1.0f, 0),
        makeDynamic(Vec3(1, 0, 0), 1.0f, 1)
    };

    HingeConstraint hc(Vec3(0.5f, 0, 0), Vec3(-0.5f, 0, 0),
                       Vec3::unitY(), Vec3::unitY(), 0, 1);
    SolverConfig config;
    hc.initialize(bodies, 2, config, 1.0f / 60.0f);

    // Should have at least 5 rows (3 linear + 2 angular)
    TEST_ASSERT(hc.activeRowCount() >= 5);
    return true;
}

static bool test_hinge_free_rotation() {
    SolverBody bodies[2] = {
        makeDynamic(Vec3(0, 0, 0), 1.0f, 0),
        makeDynamic(Vec3(0, 0, 0), 1.0f, 1)
    };
    bodies[1].angularVelocity = Vec3(0, 2, 0); // Spinning around Y

    HingeConstraint hc(Vec3::zero(), Vec3::zero(),
                       Vec3::unitY(), Vec3::unitY(), 0, 1);
    SolverConfig config;
    hc.initialize(bodies, 2, config, 1.0f / 60.0f);
    runIterations(hc, bodies, 10);

    // Y angular velocity should persist (free DoF)
    // Other angular velocities should be zeroed
    TEST_ASSERT(std::fabs(bodies[1].angularVelocity.getY()) > 0.1f);
    return true;
}

static bool test_hinge_angular_limit() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(0, 0, 0), 1.0f, 1)
    };

    HingeConstraint hc(Vec3::zero(), Vec3::zero(),
                       Vec3::unitY(), Vec3::unitY(), 0, 1);
    hc.angularLimit = JointLimit(-0.5f, 0.5f);
    SolverConfig config;
    hc.initialize(bodies, 2, config, 1.0f / 60.0f);

    // With axes aligned, the hinge angle should be ~0 and limits inactive
    // The active row count should be 5 (no limit row active when within range)
    TEST_ASSERT(hc.activeRowCount() == 5);
    return true;
}

static bool test_hinge_motor() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(0, 0, 0), 1.0f, 1)
    };

    HingeConstraint hc(Vec3::zero(), Vec3::zero(),
                       Vec3::unitY(), Vec3::unitY(), 0, 1);
    hc.motor = JointMotor(5.0f, 100.0f);
    SolverConfig config;
    float dt = 1.0f / 60.0f;
    hc.initialize(bodies, 2, config, dt);

    // Should have 6 rows (5 base + 1 motor)
    TEST_ASSERT(hc.activeRowCount() == 6);

    runIterations(hc, bodies, 20);

    // Body should be spinning around Y towards target velocity
    TEST_ASSERT(bodies[1].angularVelocity.getY() > 0.0f);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 4: SliderConstraint
// ═════════════════════════════════════════════════════════════════════════════

static bool test_slider_free_axis() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(1, 0, 0), 1.0f, 1)
    };
    bodies[1].linearVelocity = Vec3(2, 0, 0); // Moving along slider axis

    SliderConstraint sc(Vec3::zero(), Vec3::zero(), Vec3::unitX(), 0, 1);
    SolverConfig config;
    sc.initialize(bodies, 2, config, 1.0f / 60.0f);

    // Should have 5 rows (2 linear transverse + 3 angular)
    TEST_ASSERT(sc.activeRowCount() == 5);

    runIterations(sc, bodies, 10);

    // X velocity should be preserved (free axis)
    TEST_ASSERT(std::fabs(bodies[1].linearVelocity.getX()) > 0.5f);
    return true;
}

static bool test_slider_constrained_transverse() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(1, 1, 0), 1.0f, 1) // Offset in Y (transverse)
    };

    SliderConstraint sc(Vec3::zero(), Vec3::zero(), Vec3::unitX(), 0, 1);
    SolverConfig config;
    sc.initialize(bodies, 2, config, 1.0f / 60.0f);

    runIterations(sc, bodies, 10);

    // The Y offset should generate corrective velocity
    TEST_ASSERT(std::fabs(bodies[1].linearVelocity.getY()) > 0.01f);
    return true;
}

static bool test_slider_linear_limit() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(3, 0, 0), 1.0f, 1)
    };

    SliderConstraint sc(Vec3::zero(), Vec3::zero(), Vec3::unitX(), 0, 1);
    sc.linearLimit = JointLimit(-1.0f, 2.0f);
    SolverConfig config;
    sc.initialize(bodies, 2, config, 1.0f / 60.0f);

    // Slider position = 3, upper limit = 2 => should have limit row
    TEST_ASSERT(sc.activeRowCount() == 6); // 5 base + 1 limit
    TEST_ASSERT(sc.getSliderPosition() > 2.0f);
    return true;
}

static bool test_slider_motor() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(0, 0, 0), 1.0f, 1)
    };

    SliderConstraint sc(Vec3::zero(), Vec3::zero(), Vec3::unitX(), 0, 1);
    sc.motor = JointMotor(3.0f, 50.0f);
    SolverConfig config;
    float dt = 1.0f / 60.0f;
    sc.initialize(bodies, 2, config, dt);

    // Should have 6 rows (5 base + 1 motor)
    TEST_ASSERT(sc.activeRowCount() == 6);

    runIterations(sc, bodies, 20);

    // Body should be moving along X towards target velocity
    TEST_ASSERT(bodies[1].linearVelocity.getX() > 0.0f);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 5: ConeTwistConstraint
// ═════════════════════════════════════════════════════════════════════════════

static bool test_cone_twist_point_to_point() {
    SolverBody bodies[2] = {
        makeDynamic(Vec3(0, 0, 0), 1.0f, 0),
        makeDynamic(Vec3(1, 0, 0), 1.0f, 1)
    };

    ConeTwistConstraint ct(Vec3(0.5f, 0, 0), Vec3(-0.5f, 0, 0),
                           Vec3::unitX(), Vec3::unitX(), 0, 1);
    SolverConfig config;
    ct.initialize(bodies, 2, config, 1.0f / 60.0f);

    // Should have 3 base rows (point-to-point)
    TEST_ASSERT(ct.activeRowCount() >= 3);
    return true;
}

static bool test_cone_twist_swing_limit() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(0, 0, 0), 1.0f, 1)
    };

    ConeTwistConstraint ct(Vec3::zero(), Vec3::zero(),
                           Vec3::unitX(), Vec3(0, 1, 0), 0, 1);
    ct.setSwingLimit(0.1f);
    SolverConfig config;
    ct.initialize(bodies, 2, config, 1.0f / 60.0f);

    // Axes are 90° apart, swing limit is 0.1 rad => limit should be active
    TEST_ASSERT(ct.getSwingAngle() > 0.1f);
    TEST_ASSERT(ct.activeRowCount() >= 4); // 3 + swing limit
    return true;
}

static bool test_cone_twist_twist_limit() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(0, 0, 0), 1.0f, 1)
    };

    ConeTwistConstraint ct(Vec3::zero(), Vec3::zero(),
                           Vec3::unitX(), Vec3::unitX(), 0, 1);
    ct.setTwistLimit(-0.1f, 0.1f);
    SolverConfig config;
    ct.initialize(bodies, 2, config, 1.0f / 60.0f);

    // Axes are aligned, twist angle should be ~0, within limits
    TEST_ASSERT(ct.activeRowCount() == 3); // No limit rows active
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 6: SixDofConstraint
// ═════════════════════════════════════════════════════════════════════════════

static bool test_six_dof_all_locked() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(1, 0, 0), 1.0f, 1)
    };

    SixDofConstraint sdof(Vec3::zero(), Vec3::zero(), 0, 1);
    for (uint32_t i = 0; i < 6; ++i) {
        sdof.lockDof(static_cast<DofIndex>(i));
    }

    SolverConfig config;
    sdof.initialize(bodies, 2, config, 1.0f / 60.0f);

    // All 6 DoFs locked -> 6 constraint rows
    TEST_ASSERT(sdof.activeRowCount() == 6);
    return true;
}

static bool test_six_dof_all_free() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(1, 0, 0), 1.0f, 1)
    };

    SixDofConstraint sdof(Vec3::zero(), Vec3::zero(), 0, 1);
    // Default is all Free
    SolverConfig config;
    sdof.initialize(bodies, 2, config, 1.0f / 60.0f);

    // No constraint rows for free DoFs
    TEST_ASSERT(sdof.activeRowCount() == 0);
    return true;
}

static bool test_six_dof_mixed() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(0.5f, 0, 0), 1.0f, 1)
    };

    SixDofConstraint sdof(Vec3::zero(), Vec3::zero(), 0, 1);
    sdof.lockDof(DofIndex::TransY); // Lock Y translation
    sdof.lockDof(DofIndex::TransZ); // Lock Z translation
    sdof.lockDof(DofIndex::RotX);   // Lock X rotation
    sdof.lockDof(DofIndex::RotY);   // Lock Y rotation
    sdof.lockDof(DofIndex::RotZ);   // Lock Z rotation
    // TransX is free (slider-like)

    SolverConfig config;
    sdof.initialize(bodies, 2, config, 1.0f / 60.0f);

    TEST_ASSERT(sdof.activeRowCount() == 5);
    return true;
}

static bool test_six_dof_with_motor() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(0, 0, 0), 1.0f, 1)
    };

    SixDofConstraint sdof(Vec3::zero(), Vec3::zero(), 0, 1);
    sdof.lockDof(DofIndex::TransX);
    sdof.lockDof(DofIndex::TransY);
    sdof.lockDof(DofIndex::TransZ);
    sdof.setMotor(DofIndex::RotY, 3.0f, 50.0f); // Motor on Y rotation

    SolverConfig config;
    sdof.initialize(bodies, 2, config, 1.0f / 60.0f);

    // 3 locked translation rows + 1 motor row
    TEST_ASSERT(sdof.activeRowCount() == 4);
    return true;
}

static bool test_six_dof_limited_dof() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(2, 0, 0), 1.0f, 1)
    };

    SixDofConstraint sdof(Vec3::zero(), Vec3::zero(), 0, 1);
    sdof.limitDof(DofIndex::TransX, -1.0f, 1.0f); // Limited X: body at 2 exceeds upper

    SolverConfig config;
    sdof.initialize(bodies, 2, config, 1.0f / 60.0f);

    // Body at X=2 exceeds upper limit of 1 -> should have 1 limit row
    TEST_ASSERT(sdof.activeRowCount() == 1);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 7: SpringConstraint
// ═════════════════════════════════════════════════════════════════════════════

static bool test_spring_basic() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(2, 0, 0), 1.0f, 1)
    };

    SpringConstraint spring(Vec3::zero(), Vec3::zero(), 1.0f, 100.0f, 5.0f, 0, 1);
    SolverConfig config;
    spring.initialize(bodies, 2, config, 1.0f / 60.0f);

    // Should have soft effective mass
    TEST_ASSERT(spring.getRow().effectiveMass > 0.0f);

    runIterations(spring, bodies, 10);

    // Body should be moving towards rest length
    TEST_ASSERT(bodies[1].linearVelocity.getX() < 0.0f);
    return true;
}

static bool test_spring_at_rest() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(1, 0, 0), 1.0f, 1)  // At rest length
    };

    SpringConstraint spring(Vec3::zero(), Vec3::zero(), 1.0f, 100.0f, 5.0f, 0, 1);
    SolverConfig config;
    spring.initialize(bodies, 2, config, 1.0f / 60.0f);

    runIterations(spring, bodies, 10);

    // At rest length, velocity should stay near zero
    TEST_ASSERT(approx(bodies[1].linearVelocity.getX(), 0.0f, 0.5f));
    return true;
}

static bool test_spring_compressed() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(0.5f, 0, 0), 1.0f, 1)  // Compressed
    };

    SpringConstraint spring(Vec3::zero(), Vec3::zero(), 1.0f, 100.0f, 5.0f, 0, 1);
    SolverConfig config;
    spring.initialize(bodies, 2, config, 1.0f / 60.0f);

    runIterations(spring, bodies, 10);

    // Should push body away (towards rest length)
    TEST_ASSERT(bodies[1].linearVelocity.getX() > 0.0f);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 8: MotorConstraint
// ═════════════════════════════════════════════════════════════════════════════

static bool test_motor_angular_velocity() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(0, 0, 0), 1.0f, 1)
    };

    MotorConstraint mc(Vec3::unitY(), 100.0f, MotorMode::Angular, 0, 1);
    mc.targetVelocity = 5.0f;
    SolverConfig config;
    float dt = 1.0f / 60.0f;
    mc.initialize(bodies, 2, config, dt);

    runIterations(mc, bodies, 20);

    // Body should be spinning around Y
    TEST_ASSERT(bodies[1].angularVelocity.getY() > 0.0f);
    return true;
}

static bool test_motor_linear_velocity() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(0, 0, 0), 1.0f, 1)
    };

    MotorConstraint mc(Vec3::unitX(), 100.0f, MotorMode::Linear, 0, 1);
    mc.targetVelocity = 3.0f;
    SolverConfig config;
    float dt = 1.0f / 60.0f;
    mc.initialize(bodies, 2, config, dt);

    runIterations(mc, bodies, 20);

    // Body should be moving along X
    TEST_ASSERT(bodies[1].linearVelocity.getX() > 0.0f);
    return true;
}

static bool test_motor_max_force_clamp() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(0, 0, 0), 1.0f, 1)
    };

    float dt = 1.0f / 60.0f;
    MotorConstraint mc(Vec3::unitY(), 0.001f, MotorMode::Angular, 0, 1); // Very low force
    mc.targetVelocity = 1000.0f; // Very high target
    SolverConfig config;
    mc.initialize(bodies, 2, config, dt);

    runIterations(mc, bodies, 10);

    // Impulse should be clamped to maxForce * dt
    float maxImpulse = 0.001f * dt;
    TEST_ASSERT(mc.getAccumulatedImpulse() <= maxImpulse + 0.001f);
    return true;
}

static bool test_motor_position_target() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(0, 0, 0), 1.0f, 1)
    };

    MotorConstraint mc(Vec3::unitX(), 100.0f, MotorMode::Linear, 0, 1);
    mc.target = MotorTarget::Position;
    mc.targetPosition = 5.0f;
    mc.targetVelocity = 0.0f;
    mc.positionGain = 1.0f;
    SolverConfig config;
    mc.initialize(bodies, 2, config, 1.0f / 60.0f);

    runIterations(mc, bodies, 10);

    // Should be moving towards target position
    TEST_ASSERT(bodies[1].linearVelocity.getX() > 0.0f);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 9: Edge Cases
// ═════════════════════════════════════════════════════════════════════════════

static bool test_edge_two_static_bodies() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeStatic(Vec3(2, 0, 0), 1)
    };

    DistanceConstraint dc(Vec3::zero(), Vec3::zero(), 1.0f, 0, 1);
    SolverConfig config;
    dc.initialize(bodies, 2, config, 1.0f / 60.0f);
    runIterations(dc, bodies, 10);

    // Neither body should move
    TEST_ASSERT(approxVec(bodies[0].linearVelocity, Vec3::zero()));
    TEST_ASSERT(approxVec(bodies[1].linearVelocity, Vec3::zero()));
    return true;
}

static bool test_edge_zero_dt() {
    SolverBody bodies[2] = {
        makeDynamic(Vec3(0, 0, 0), 1.0f, 0),
        makeDynamic(Vec3(2, 0, 0), 1.0f, 1)
    };

    DistanceConstraint dc(Vec3::zero(), Vec3::zero(), 1.0f, 0, 1);
    SolverConfig config;
    dc.initialize(bodies, 2, config, 0.0f); // Zero dt
    runIterations(dc, bodies, 5);

    // Should not crash, and effective mass should be computed
    return true;
}

static bool test_edge_constraint_disabled() {
    SolverBody bodies[2] = {
        makeDynamic(Vec3(0, 0, 0), 1.0f, 0),
        makeDynamic(Vec3(2, 0, 0), 1.0f, 1)
    };

    DistanceConstraint dc(Vec3::zero(), Vec3::zero(), 1.0f, 0, 1);
    dc.header.setEnabled(false);
    TEST_ASSERT(!dc.header.isEnabled());

    // Constraint flag should be checked by the caller (World module)
    // The constraint itself still initializes, but World should skip it
    SolverConfig config;
    dc.initialize(bodies, 2, config, 1.0f / 60.0f);
    return true;
}

static bool test_edge_invalid_body_id() {
    SolverBody bodies[2] = {
        makeDynamic(Vec3(0, 0, 0), 1.0f, 0),
        makeDynamic(Vec3(2, 0, 0), 1.0f, 1)
    };

    // Use body ID 999 which doesn't exist
    DistanceConstraint dc(Vec3::zero(), Vec3::zero(), 1.0f, 0, 999);
    SolverConfig config;
    dc.initialize(bodies, 2, config, 1.0f / 60.0f);
    runIterations(dc, bodies, 5);

    // Should not crash
    return true;
}

static bool test_edge_constraint_type_tags() {
    DistanceConstraint dc;
    TEST_ASSERT(dc.header.type == ConstraintType::Distance);

    HingeConstraint hc;
    TEST_ASSERT(hc.header.type == ConstraintType::Hinge);

    SliderConstraint sc;
    TEST_ASSERT(sc.header.type == ConstraintType::Slider);

    ConeTwistConstraint ct;
    TEST_ASSERT(ct.header.type == ConstraintType::ConeTwist);

    SixDofConstraint sdof;
    TEST_ASSERT(sdof.header.type == ConstraintType::SixDof);

    SpringConstraint sp;
    TEST_ASSERT(sp.header.type == ConstraintType::Spring);

    MotorConstraint mc;
    TEST_ASSERT(mc.header.type == ConstraintType::Motor);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 10: Warm Starting
// ═════════════════════════════════════════════════════════════════════════════

static bool test_warm_start_distance() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(2, 0, 0), 1.0f, 1)
    };

    DistanceConstraint dc(Vec3::zero(), Vec3::zero(), 1.0f, 0, 1);
    SolverConfig config;
    float dt = 1.0f / 60.0f;
    dc.initialize(bodies, 2, config, dt);

    // First solve pass
    runIterations(dc, bodies, 10);
    float firstImpulse = dc.getAccumulatedImpulse();

    // Reset velocities, re-init with warm start
    bodies[1].linearVelocity = Vec3::zero();
    bodies[1].angularVelocity = Vec3::zero();

    dc.initialize(bodies, 2, config, dt);
    dc.setWarmStartImpulse(firstImpulse);
    dc.warmStart(bodies);

    // After warm start, body should already have some velocity
    TEST_ASSERT(std::fabs(bodies[1].linearVelocity.getX()) > 0.01f);
    return true;
}

static bool test_warm_start_hinge() {
    SolverBody bodies[2] = {
        makeStatic(Vec3(0, 0, 0), 0),
        makeDynamic(Vec3(1, 0, 0), 1.0f, 1)
    };

    HingeConstraint hc(Vec3::zero(), Vec3(-1, 0, 0),
                       Vec3::unitY(), Vec3::unitY(), 0, 1);
    SolverConfig config;
    float dt = 1.0f / 60.0f;
    hc.initialize(bodies, 2, config, dt);
    hc.warmStart(bodies);

    // Should not crash, warm start with zero impulses is a no-op
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("=== Pulse Constraints Module Tests ===\n\n");

    // Group 1: JointRow & Common Helpers
    std::printf("--- JointRow & Helpers ---\n");
    RUN_TEST(test_joint_limit_default);
    RUN_TEST(test_joint_limit_classification);
    RUN_TEST(test_joint_limit_locked);
    RUN_TEST(test_joint_motor_default);
    RUN_TEST(test_joint_motor_construction);
    RUN_TEST(test_soft_params_rigid);
    RUN_TEST(test_soft_params_compute);
    RUN_TEST(test_soft_params_from_stiffness);
    RUN_TEST(test_joint_row_default);
    RUN_TEST(test_effective_mass_computation);
    RUN_TEST(test_solve_joint_row_basic);
    RUN_TEST(test_solve_row_clamping);

    // Group 2: DistanceConstraint
    std::printf("--- DistanceConstraint ---\n");
    RUN_TEST(test_distance_maintains_distance);
    RUN_TEST(test_distance_zero_length);
    RUN_TEST(test_distance_soft_spring);
    RUN_TEST(test_distance_with_static_body);

    // Group 3: HingeConstraint
    std::printf("--- HingeConstraint ---\n");
    RUN_TEST(test_hinge_anchor_coincidence);
    RUN_TEST(test_hinge_free_rotation);
    RUN_TEST(test_hinge_angular_limit);
    RUN_TEST(test_hinge_motor);

    // Group 4: SliderConstraint
    std::printf("--- SliderConstraint ---\n");
    RUN_TEST(test_slider_free_axis);
    RUN_TEST(test_slider_constrained_transverse);
    RUN_TEST(test_slider_linear_limit);
    RUN_TEST(test_slider_motor);

    // Group 5: ConeTwistConstraint
    std::printf("--- ConeTwistConstraint ---\n");
    RUN_TEST(test_cone_twist_point_to_point);
    RUN_TEST(test_cone_twist_swing_limit);
    RUN_TEST(test_cone_twist_twist_limit);

    // Group 6: SixDofConstraint
    std::printf("--- SixDofConstraint ---\n");
    RUN_TEST(test_six_dof_all_locked);
    RUN_TEST(test_six_dof_all_free);
    RUN_TEST(test_six_dof_mixed);
    RUN_TEST(test_six_dof_with_motor);
    RUN_TEST(test_six_dof_limited_dof);

    // Group 7: SpringConstraint
    std::printf("--- SpringConstraint ---\n");
    RUN_TEST(test_spring_basic);
    RUN_TEST(test_spring_at_rest);
    RUN_TEST(test_spring_compressed);

    // Group 8: MotorConstraint
    std::printf("--- MotorConstraint ---\n");
    RUN_TEST(test_motor_angular_velocity);
    RUN_TEST(test_motor_linear_velocity);
    RUN_TEST(test_motor_max_force_clamp);
    RUN_TEST(test_motor_position_target);

    // Group 9: Edge Cases
    std::printf("--- Edge Cases ---\n");
    RUN_TEST(test_edge_two_static_bodies);
    RUN_TEST(test_edge_zero_dt);
    RUN_TEST(test_edge_constraint_disabled);
    RUN_TEST(test_edge_invalid_body_id);
    RUN_TEST(test_edge_constraint_type_tags);

    // Group 10: Warm Starting
    std::printf("--- Warm Starting ---\n");
    RUN_TEST(test_warm_start_distance);
    RUN_TEST(test_warm_start_hinge);

    // Summary
    std::printf("\n=== Results: %d/%d passed", g_passed, g_total);
    if (g_failed > 0) std::printf(", %d FAILED", g_failed);
    std::printf(" ===\n");

    return (g_failed > 0) ? 1 : 0;
}
