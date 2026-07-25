/**
 * @file test_solver.cpp
 * @brief Comprehensive unit tests for the Pulse solver module.
 *
 * Tests SolverConfig, SolverBody, VelocityConstraint, PositionConstraint,
 * ContactSolver (initialize, warm-start, velocity/position solving),
 * the top-level solve() facade, restitution, friction, split-impulse,
 * multi-contact scenarios, and edge cases.
 */

#include <pulse/solver/solver_common.h>
#include <pulse/solver/constraint_base.h>
#include <pulse/solver/contact_solver.h>
#include <pulse/solver/solver.h>

#include <pulse/contact/contact_common.h>
#include <pulse/contact/persistent_contact.h>
#include <pulse/contact/contact_manifold_persistent.h>
#include <pulse/narrowphase/narrowphase_common.h>
#include <pulse/math/math_common.h>
#include <pulse/math/vec3.h>

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

/// Create a SolverBody for a dynamic sphere.
static SolverBody makeDynamicBody(Vec3 pos, float mass, float radius,
                                   float rest, float fric, uint32_t id) {
    // Sphere inertia: I = 2/5 * m * r^2
    float I = 0.4f * mass * radius * radius;
    SolverBody b;
    b.position = pos;
    b.invMass = 1.0f / mass;
    b.invInertia = Vec3(1.0f / I, 1.0f / I, 1.0f / I);
    b.restitution = rest;
    b.friction = fric;
    b.bodyId = id;
    return b;
}

/// Create a static (infinite mass) body.
static SolverBody makeStaticBody(Vec3 pos, float fric, uint32_t id) {
    SolverBody b;
    b.position = pos;
    b.invMass = 0.0f;
    b.invInertia = Vec3::zero();
    b.restitution = 0.0f;
    b.friction = fric;
    b.bodyId = id;
    return b;
}

/// Create a PersistentManifold with one contact point between two bodies.
static PersistentManifold makeOneContactManifold(
    uint32_t idA, uint32_t idB,
    Vec3 posOnA, Vec3 posOnB, Vec3 normal, float depth)
{
    PersistentManifold pm(idA, idB);
    ContactPoint cp(posOnA, posOnB, normal, depth);
    ContactManifold cm;
    cm.addPoint(cp);
    pm.mergeContacts(cm, 0.04f * 0.04f);
    return pm;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 1: SolverConfig
// ═════════════════════════════════════════════════════════════════════════════

static bool test_config_defaults() {
    SolverConfig config;
    TEST_ASSERT(config.velocityIterations == 8);
    TEST_ASSERT(config.positionIterations == 3);
    TEST_ASSERT(approx(config.baumgarte, 0.2f));
    TEST_ASSERT(approx(config.slop, 0.005f));
    TEST_ASSERT(approx(config.restitutionThreshold, 1.0f));
    TEST_ASSERT(approx(config.maxPositionCorrection, 0.2f));
    TEST_ASSERT(config.warmStarting == true);
    return true;
}

static bool test_config_custom() {
    SolverConfig config(16, 6, 0.3f, 0.01f, 0.5f, 0.1f, false);
    TEST_ASSERT(config.velocityIterations == 16);
    TEST_ASSERT(config.positionIterations == 6);
    TEST_ASSERT(approx(config.baumgarte, 0.3f));
    TEST_ASSERT(approx(config.slop, 0.01f));
    TEST_ASSERT(approx(config.restitutionThreshold, 0.5f));
    TEST_ASSERT(approx(config.maxPositionCorrection, 0.1f));
    TEST_ASSERT(config.warmStarting == false);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 2: SolverBody
// ═════════════════════════════════════════════════════════════════════════════

static bool test_body_default_is_static() {
    SolverBody body;
    TEST_ASSERT(body.isStatic());
    TEST_ASSERT(body.invMass < math::Epsilon);
    TEST_ASSERT(approxVec(body.linearVelocity, Vec3::zero()));
    TEST_ASSERT(approxVec(body.angularVelocity, Vec3::zero()));
    return true;
}

static bool test_body_dynamic() {
    SolverBody body = makeDynamicBody(Vec3(1, 2, 3), 10.0f, 0.5f, 0.5f, 0.3f, 42);
    TEST_ASSERT(!body.isStatic());
    TEST_ASSERT(approx(body.invMass, 0.1f));
    TEST_ASSERT(body.bodyId == 42);
    TEST_ASSERT(approxVec(body.position, Vec3(1, 2, 3)));
    return true;
}

static bool test_body_apply_linear_impulse() {
    SolverBody body = makeDynamicBody(Vec3::zero(), 2.0f, 0.5f, 0.0f, 0.0f, 0);
    body.applyLinearImpulse(Vec3(4.0f, 0.0f, 0.0f));
    // v += impulse * invMass = (4, 0, 0) * 0.5 = (2, 0, 0)
    TEST_ASSERT(approx(body.linearVelocity.getX(), 2.0f));
    return true;
}

static bool test_body_apply_angular_impulse() {
    SolverBody body = makeDynamicBody(Vec3::zero(), 1.0f, 1.0f, 0.0f, 0.0f, 0);
    Vec3 torque(0.0f, 1.0f, 0.0f);
    body.applyAngularImpulse(torque);
    // ω += invI * torque; invI = 1/(0.4*1*1) = 2.5
    TEST_ASSERT(approx(body.angularVelocity.getY(), 2.5f, 0.1f));
    return true;
}

static bool test_body_static_ignores_impulse() {
    SolverBody body = makeStaticBody(Vec3::zero(), 0.5f, 0);
    body.applyLinearImpulse(Vec3(100.0f, 0.0f, 0.0f));
    TEST_ASSERT(approxVec(body.linearVelocity, Vec3::zero()));
    return true;
}

static bool test_body_convenience_constructor() {
    // mass=5, inertia=(2,3,4), restitution=0.7, friction=0.5, id=99
    SolverBody body(Vec3(1, 2, 3), 5.0f, Vec3(2.0f, 3.0f, 4.0f), 0.7f, 0.5f, 99);
    TEST_ASSERT(!body.isStatic());
    TEST_ASSERT(approx(body.invMass, 0.2f));
    TEST_ASSERT(approx(body.invInertia.getX(), 0.5f));
    TEST_ASSERT(approx(body.restitution, 0.7f));
    TEST_ASSERT(body.bodyId == 99);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 3: Constraint Types
// ═════════════════════════════════════════════════════════════════════════════

static bool test_constraint_header_default() {
    ConstraintHeader h;
    TEST_ASSERT(h.type == ConstraintType::Contact);
    TEST_ASSERT(h.bodyIdA == 0xFFFFFFFFu);
    TEST_ASSERT(h.isEnabled());
    return true;
}

static bool test_constraint_header_custom() {
    ConstraintHeader h(ConstraintType::Hinge, 10, 20);
    TEST_ASSERT(h.type == ConstraintType::Hinge);
    TEST_ASSERT(h.bodyIdA == 10);
    TEST_ASSERT(h.bodyIdB == 20);
    return true;
}

static bool test_constraint_enable_disable() {
    ConstraintHeader h;
    TEST_ASSERT(h.isEnabled());
    h.setEnabled(false);
    TEST_ASSERT(!h.isEnabled());
    h.setEnabled(true);
    TEST_ASSERT(h.isEnabled());
    return true;
}

static bool test_velocity_constraint_default() {
    VelocityConstraint vc;
    TEST_ASSERT(approx(vc.normalMass, 0.0f));
    TEST_ASSERT(approx(vc.normalImpulse, 0.0f));
    TEST_ASSERT(approx(vc.tangentImpulse0, 0.0f));
    TEST_ASSERT(approx(vc.tangentImpulse1, 0.0f));
    return true;
}

static bool test_position_constraint_default() {
    PositionConstraint pc;
    TEST_ASSERT(approx(pc.penetration, 0.0f));
    TEST_ASSERT(approx(pc.normalMass, 0.0f));
    return true;
}

static bool test_solver_stats_default() {
    SolverStats stats;
    TEST_ASSERT(stats.velocityIterationsUsed == 0);
    TEST_ASSERT(stats.positionIterationsUsed == 0);
    TEST_ASSERT(stats.totalContacts == 0);
    TEST_ASSERT(stats.positionSolved == false);
    return true;
}

static bool test_contact_constraint_group_default() {
    ContactConstraintGroup g;
    TEST_ASSERT(g.header.type == ConstraintType::Contact);
    TEST_ASSERT(g.count == 0);
    TEST_ASSERT(g.startIndex == 0);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 4: ContactSolver Initialization
// ═════════════════════════════════════════════════════════════════════════════

static bool test_solver_init_empty() {
    ContactSolver solver;
    SolverBody bodies[1];
    bodies[0] = makeStaticBody(Vec3::zero(), 0.5f, 0);

    solver.initialize(bodies, 1, nullptr, 0, SolverConfig(), 1.0f / 60.0f);
    TEST_ASSERT(solver.velocityConstraintCount() == 0);
    TEST_ASSERT(solver.positionConstraintCount() == 0);
    TEST_ASSERT(solver.groupCount() == 0);
    return true;
}

static bool test_solver_init_one_contact() {
    // Two spheres colliding head-on
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(-0.5f, 0, 0), 1.0f, 0.5f, 0.5f, 0.3f, 0);
    bodies[1] = makeDynamicBody(Vec3(0.5f, 0, 0), 1.0f, 0.5f, 0.5f, 0.3f, 1);

    PersistentManifold pm = makeOneContactManifold(
        0, 1,
        Vec3(0.0f, 0, 0), Vec3(0.0f, 0, 0),
        Vec3(1.0f, 0, 0), 0.02f
    );

    ContactSolver solver;
    solver.initialize(bodies, 2, &pm, 1, SolverConfig(), 1.0f / 60.0f);

    TEST_ASSERT(solver.velocityConstraintCount() == 1);
    TEST_ASSERT(solver.positionConstraintCount() == 1);
    TEST_ASSERT(solver.groupCount() == 1);

    // Effective mass should be positive
    TEST_ASSERT(solver.velocityConstraint(0).normalMass > 0.0f);
    return true;
}

static bool test_solver_init_multiple_contacts() {
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(0, 1, 0), 1.0f, 0.5f, 0.0f, 0.5f, 0);
    bodies[1] = makeStaticBody(Vec3(0, 0, 0), 0.5f, 1);

    // 4-contact manifold (like a box resting on a plane)
    PersistentManifold pm(0, 1);
    ContactManifold cm;
    cm.addPoint(ContactPoint(Vec3(-0.5f, 0.5f, -0.5f), Vec3(-0.5f, 0, -0.5f), Vec3(0, 1, 0), 0.01f));
    cm.addPoint(ContactPoint(Vec3(0.5f, 0.5f, -0.5f), Vec3(0.5f, 0, -0.5f), Vec3(0, 1, 0), 0.01f));
    cm.addPoint(ContactPoint(Vec3(-0.5f, 0.5f, 0.5f), Vec3(-0.5f, 0, 0.5f), Vec3(0, 1, 0), 0.01f));
    cm.addPoint(ContactPoint(Vec3(0.5f, 0.5f, 0.5f), Vec3(0.5f, 0, 0.5f), Vec3(0, 1, 0), 0.01f));
    pm.mergeContacts(cm, 0.04f * 0.04f);

    ContactSolver solver;
    solver.initialize(bodies, 2, &pm, 1, SolverConfig(), 1.0f / 60.0f);

    TEST_ASSERT(solver.velocityConstraintCount() == 4);
    TEST_ASSERT(solver.positionConstraintCount() == 4);
    return true;
}

static bool test_solver_init_body_lookup() {
    // Bodies with non-sequential IDs
    SolverBody bodies[3];
    bodies[0] = makeDynamicBody(Vec3::zero(), 1.0f, 0.5f, 0.0f, 0.5f, 100);
    bodies[1] = makeDynamicBody(Vec3(1, 0, 0), 1.0f, 0.5f, 0.0f, 0.5f, 200);
    bodies[2] = makeStaticBody(Vec3(0, -1, 0), 0.5f, 300);

    PersistentManifold pm = makeOneContactManifold(
        100, 300,
        Vec3(0, -0.5f, 0), Vec3(0, -1.0f, 0),
        Vec3(0, 1, 0), 0.01f
    );

    ContactSolver solver;
    solver.initialize(bodies, 3, &pm, 1, SolverConfig(), 1.0f / 60.0f);

    TEST_ASSERT(solver.velocityConstraintCount() == 1);
    // The constraint should have found the correct body indices
    const VelocityConstraint& vc = solver.velocityConstraint(0);
    TEST_ASSERT(vc.indexA == 0);  // bodies[0] has id=100
    TEST_ASSERT(vc.indexB == 2);  // bodies[2] has id=300
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 5: Contact Resolution
// ═════════════════════════════════════════════════════════════════════════════

static bool test_two_spheres_separate() {
    // Two equal spheres approaching head-on along X
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(-0.48f, 0, 0), 1.0f, 0.5f, 0.0f, 0.0f, 0);
    bodies[0].linearVelocity = Vec3(2.0f, 0, 0);  // Moving right
    bodies[1] = makeDynamicBody(Vec3(0.48f, 0, 0), 1.0f, 0.5f, 0.0f, 0.0f, 1);
    bodies[1].linearVelocity = Vec3(-2.0f, 0, 0);  // Moving left

    // Normal from B toward A = from right to left = (-1, 0, 0)
    PersistentManifold pm = makeOneContactManifold(
        0, 1,
        Vec3(0.0f, 0, 0), Vec3(0.0f, 0, 0),
        Vec3(-1.0f, 0, 0), 0.04f
    );

    SolverConfig config;
    config.warmStarting = false;
    float dt = 1.0f / 60.0f;

    SolverStats stats = solve(bodies, 2, &pm, 1, config, dt);

    // After solving, bodies should no longer be approaching each other
    // Body 0 should have velocity ≤ 0 in X (stopped or bounced back)
    // Body 1 should have velocity ≥ 0 in X
    float relVelX = bodies[1].linearVelocity.getX() - bodies[0].linearVelocity.getX();
    TEST_ASSERT(relVelX >= -0.1f);  // Should be separating or at rest

    TEST_ASSERT(stats.totalContacts == 1);
    TEST_ASSERT(stats.velocityIterationsUsed > 0);
    return true;
}

static bool test_sphere_on_static_plane() {
    // Dynamic sphere sitting on a static infinite plane
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(0, 0.49f, 0), 1.0f, 0.5f, 0.0f, 0.5f, 0);
    bodies[0].linearVelocity = Vec3(0, -1.0f, 0);  // Falling
    bodies[1] = makeStaticBody(Vec3(0, 0, 0), 0.5f, 1);

    PersistentManifold pm = makeOneContactManifold(
        0, 1,
        Vec3(0, 0, 0), Vec3(0, 0, 0),
        Vec3(0, 1, 0), 0.01f
    );

    SolverConfig config;
    config.warmStarting = false;
    float dt = 1.0f / 60.0f;

    solve(bodies, 2, &pm, 1, config, dt);

    // Dynamic sphere's downward velocity should be stopped
    TEST_ASSERT(bodies[0].linearVelocity.getY() >= -0.1f);
    // Static body should not move
    TEST_ASSERT(approxVec(bodies[1].linearVelocity, Vec3::zero()));
    TEST_ASSERT(approxVec(bodies[1].position, Vec3(0, 0, 0)));
    return true;
}

static bool test_normal_impulse_non_negative() {
    // Separating bodies should produce zero normal impulse
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(-0.5f, 0, 0), 1.0f, 0.5f, 0.0f, 0.0f, 0);
    bodies[0].linearVelocity = Vec3(-2.0f, 0, 0);  // Moving away
    bodies[1] = makeDynamicBody(Vec3(0.5f, 0, 0), 1.0f, 0.5f, 0.0f, 0.0f, 1);
    bodies[1].linearVelocity = Vec3(2.0f, 0, 0);  // Moving away

    PersistentManifold pm = makeOneContactManifold(
        0, 1,
        Vec3(0, 0, 0), Vec3(0, 0, 0),
        Vec3(1, 0, 0), 0.001f
    );

    SolverConfig config;
    config.warmStarting = false;
    solve(bodies, 2, &pm, 1, config, 1.0f / 60.0f);

    // Normal impulse in manifold should be ≥ 0
    TEST_ASSERT(pm.contacts[0].normalImpulse >= -math::Epsilon);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 6: Restitution
// ═════════════════════════════════════════════════════════════════════════════

static bool test_restitution_bounce() {
    // Ball bouncing off a static floor with e=1.0 (perfect bounce)
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(0, 0.49f, 0), 1.0f, 0.5f, 1.0f, 0.0f, 0);
    bodies[0].linearVelocity = Vec3(0, -5.0f, 0);  // Fast enough for restitution
    bodies[1] = makeStaticBody(Vec3(0, 0, 0), 0.0f, 1);
    bodies[1].restitution = 1.0f;

    PersistentManifold pm = makeOneContactManifold(
        0, 1,
        Vec3(0, 0, 0), Vec3(0, 0, 0),
        Vec3(0, 1, 0), 0.01f
    );

    SolverConfig config;
    config.warmStarting = false;
    config.restitutionThreshold = 0.5f;  // Low threshold so restitution activates

    solve(bodies, 2, &pm, 1, config, 1.0f / 60.0f);

    // With e=1.0, ball should bounce back at approximately +5 m/s
    TEST_ASSERT(bodies[0].linearVelocity.getY() > 3.0f);
    return true;
}

static bool test_restitution_zero_no_bounce() {
    // Ball with e=0 should come to rest (no bounce)
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(0, 0.49f, 0), 1.0f, 0.5f, 0.0f, 0.0f, 0);
    bodies[0].linearVelocity = Vec3(0, -5.0f, 0);
    bodies[1] = makeStaticBody(Vec3(0, 0, 0), 0.0f, 1);

    PersistentManifold pm = makeOneContactManifold(
        0, 1,
        Vec3(0, 0, 0), Vec3(0, 0, 0),
        Vec3(0, 1, 0), 0.01f
    );

    SolverConfig config;
    config.warmStarting = false;
    config.restitutionThreshold = 0.5f;

    solve(bodies, 2, &pm, 1, config, 1.0f / 60.0f);

    // With e=0, ball should stop (velocity near zero or slightly positive from correction)
    TEST_ASSERT(bodies[0].linearVelocity.getY() >= -0.5f);
    TEST_ASSERT(bodies[0].linearVelocity.getY() < 2.0f);
    return true;
}

static bool test_restitution_threshold() {
    // Slow impact (below threshold) should not apply restitution
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(0, 0.49f, 0), 1.0f, 0.5f, 1.0f, 0.0f, 0);
    bodies[0].linearVelocity = Vec3(0, -0.3f, 0);  // Below threshold
    bodies[1] = makeStaticBody(Vec3(0, 0, 0), 0.0f, 1);
    bodies[1].restitution = 1.0f;

    PersistentManifold pm = makeOneContactManifold(
        0, 1,
        Vec3(0, 0, 0), Vec3(0, 0, 0),
        Vec3(0, 1, 0), 0.01f
    );

    SolverConfig config;
    config.warmStarting = false;
    config.restitutionThreshold = 1.0f;  // Threshold higher than closing speed

    solve(bodies, 2, &pm, 1, config, 1.0f / 60.0f);

    // Should not bounce (restitution suppressed)
    TEST_ASSERT(bodies[0].linearVelocity.getY() < 1.0f);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 7: Friction
// ═════════════════════════════════════════════════════════════════════════════

static bool test_friction_slows_sliding() {
    // Object sliding on a surface with friction.
    // Needs a closing velocity so the normal constraint activates,
    // providing the normal impulse that enables friction (F_t ≤ μ·F_n).
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(0, 0.49f, 0), 1.0f, 0.5f, 0.0f, 0.8f, 0);
    bodies[0].linearVelocity = Vec3(5.0f, -1.0f, 0);  // Sliding along X + falling into surface
    bodies[1] = makeStaticBody(Vec3(0, 0, 0), 0.8f, 1);

    PersistentManifold pm = makeOneContactManifold(
        0, 1,
        Vec3(0, 0, 0), Vec3(0, 0, 0),
        Vec3(0, 1, 0), 0.01f
    );

    SolverConfig config;
    config.warmStarting = false;

    float velBefore = bodies[0].linearVelocity.getX();
    solve(bodies, 2, &pm, 1, config, 1.0f / 60.0f);
    float velAfter = bodies[0].linearVelocity.getX();

    // Friction should reduce the tangential velocity
    TEST_ASSERT(velAfter < velBefore);
    return true;
}

static bool test_zero_friction_no_slowdown() {
    // Object sliding on a frictionless surface
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(0, 0.49f, 0), 1.0f, 0.5f, 0.0f, 0.0f, 0);
    bodies[0].linearVelocity = Vec3(5.0f, 0, 0);
    bodies[1] = makeStaticBody(Vec3(0, 0, 0), 0.0f, 1);

    PersistentManifold pm = makeOneContactManifold(
        0, 1,
        Vec3(0, 0, 0), Vec3(0, 0, 0),
        Vec3(0, 1, 0), 0.01f
    );

    SolverConfig config;
    config.warmStarting = false;

    solve(bodies, 2, &pm, 1, config, 1.0f / 60.0f);

    // With zero friction, tangential velocity should be preserved
    TEST_ASSERT(approx(bodies[0].linearVelocity.getX(), 5.0f, 0.5f));
    return true;
}

static bool test_friction_cone_limit() {
    // Verify friction impulse ≤ μ * normal impulse (Coulomb cone)
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(0, 0.49f, 0), 1.0f, 0.5f, 0.0f, 0.3f, 0);
    bodies[0].linearVelocity = Vec3(100.0f, -5.0f, 0);  // Fast slide + contact
    bodies[1] = makeStaticBody(Vec3(0, 0, 0), 0.3f, 1);

    PersistentManifold pm = makeOneContactManifold(
        0, 1,
        Vec3(0, 0, 0), Vec3(0, 0, 0),
        Vec3(0, 1, 0), 0.01f
    );

    SolverConfig config;
    config.warmStarting = false;

    solve(bodies, 2, &pm, 1, config, 1.0f / 60.0f);

    // Tangent impulse should be bounded by friction * normal impulse
    float ni = pm.contacts[0].normalImpulse;
    float ti0 = std::fabs(pm.contacts[0].tangentImpulse0);
    float ti1 = std::fabs(pm.contacts[0].tangentImpulse1);
    float maxFriction = 0.3f * ni;  // μ = sqrt(0.3 * 0.3) = 0.3

    // Allow some numerical tolerance
    TEST_ASSERT(ti0 <= maxFriction + 0.1f);
    TEST_ASSERT(ti1 <= maxFriction + 0.1f);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 8: Split-Impulse Position Correction
// ═════════════════════════════════════════════════════════════════════════════

static bool test_position_correction_resolves_overlap() {
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(0, 0.4f, 0), 1.0f, 0.5f, 0.0f, 0.0f, 0);
    bodies[1] = makeStaticBody(Vec3(0, 0, 0), 0.0f, 1);

    PersistentManifold pm = makeOneContactManifold(
        0, 1,
        Vec3(0, 0, 0), Vec3(0, 0, 0),
        Vec3(0, 1, 0), 0.1f  // 10cm penetration
    );

    SolverConfig config;
    config.warmStarting = false;

    float posBefore = bodies[0].position.getY();
    solve(bodies, 2, &pm, 1, config, 1.0f / 60.0f);
    float posAfter = bodies[0].position.getY();

    // Position should have been pushed upward
    TEST_ASSERT(posAfter >= posBefore);
    return true;
}

static bool test_position_no_velocity_artifact() {
    // Split-impulse should correct position without adding velocity
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(0, 0.45f, 0), 1.0f, 0.5f, 0.0f, 0.0f, 0);
    bodies[0].linearVelocity = Vec3::zero();  // At rest
    bodies[1] = makeStaticBody(Vec3(0, 0, 0), 0.0f, 1);

    PersistentManifold pm = makeOneContactManifold(
        0, 1,
        Vec3(0, 0, 0), Vec3(0, 0, 0),
        Vec3(0, 1, 0), 0.05f
    );

    SolverConfig config;
    config.warmStarting = false;

    solve(bodies, 2, &pm, 1, config, 1.0f / 60.0f);

    // Velocity should remain approximately zero (position correction is separate)
    TEST_ASSERT(std::fabs(bodies[0].linearVelocity.getY()) < 1.0f);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 9: Warm Starting
// ═════════════════════════════════════════════════════════════════════════════

static bool test_warm_start_uses_cached_impulses() {
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(0, 0.49f, 0), 1.0f, 0.5f, 0.0f, 0.5f, 0);
    bodies[0].linearVelocity = Vec3(0, -2.0f, 0);
    bodies[1] = makeStaticBody(Vec3(0, 0, 0), 0.5f, 1);

    PersistentManifold pm = makeOneContactManifold(
        0, 1,
        Vec3(0, 0, 0), Vec3(0, 0, 0),
        Vec3(0, 1, 0), 0.01f
    );

    // First solve — builds up impulses
    SolverConfig config;
    config.warmStarting = false;
    solve(bodies, 2, &pm, 1, config, 1.0f / 60.0f);

    float storedImpulse = pm.contacts[0].normalImpulse;
    TEST_ASSERT(storedImpulse > 0.0f);

    // Second solve with warm starting — should use the stored impulse
    bodies[0].linearVelocity = Vec3(0, -2.0f, 0);
    config.warmStarting = true;
    solve(bodies, 2, &pm, 1, config, 1.0f / 60.0f);

    // The stored impulse should still be positive
    TEST_ASSERT(pm.contacts[0].normalImpulse > 0.0f);
    return true;
}

static bool test_warm_start_disabled() {
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(0, 0.49f, 0), 1.0f, 0.5f, 0.0f, 0.5f, 0);
    bodies[0].linearVelocity = Vec3(0, -2.0f, 0);
    bodies[1] = makeStaticBody(Vec3(0, 0, 0), 0.5f, 1);

    PersistentManifold pm = makeOneContactManifold(
        0, 1,
        Vec3(0, 0, 0), Vec3(0, 0, 0),
        Vec3(0, 1, 0), 0.01f
    );

    // Pre-fill with impulse data
    pm.contacts[0].normalImpulse = 100.0f;

    SolverConfig config;
    config.warmStarting = false;  // Disabled!

    // The solver should not use the pre-filled impulse
    ContactSolver solver;
    solver.initialize(bodies, 2, &pm, 1, config, 1.0f / 60.0f);

    // With warm starting disabled, initial impulse should be 0
    TEST_ASSERT(approx(solver.velocityConstraint(0).normalImpulse, 0.0f));
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 10: Multi-Contact Scenarios
// ═════════════════════════════════════════════════════════════════════════════

static bool test_box_on_plane_4_contacts() {
    // Box resting on a plane — 4 contact points
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(0, 0.5f, 0), 1.0f, 0.5f, 0.0f, 0.5f, 0);
    bodies[0].linearVelocity = Vec3(0, -1.0f, 0);
    bodies[1] = makeStaticBody(Vec3(0, 0, 0), 0.5f, 1);

    PersistentManifold pm(0, 1);
    ContactManifold cm;
    cm.addPoint(ContactPoint(Vec3(-0.5f, 0, -0.5f), Vec3(-0.5f, 0, -0.5f), Vec3(0, 1, 0), 0.01f));
    cm.addPoint(ContactPoint(Vec3(0.5f, 0, -0.5f), Vec3(0.5f, 0, -0.5f), Vec3(0, 1, 0), 0.01f));
    cm.addPoint(ContactPoint(Vec3(-0.5f, 0, 0.5f), Vec3(-0.5f, 0, 0.5f), Vec3(0, 1, 0), 0.01f));
    cm.addPoint(ContactPoint(Vec3(0.5f, 0, 0.5f), Vec3(0.5f, 0, 0.5f), Vec3(0, 1, 0), 0.01f));
    pm.mergeContacts(cm, 0.04f * 0.04f);

    SolverConfig config;
    config.warmStarting = false;

    SolverStats stats = solve(bodies, 2, &pm, 1, config, 1.0f / 60.0f);

    TEST_ASSERT(stats.totalContacts == 4);
    // Box should stop falling
    TEST_ASSERT(bodies[0].linearVelocity.getY() >= -0.5f);
    return true;
}

static bool test_multiple_manifolds() {
    // Three bodies: A-B and B-C both in contact
    SolverBody bodies[3];
    bodies[0] = makeDynamicBody(Vec3(0, 2, 0), 1.0f, 0.5f, 0.0f, 0.5f, 0);
    bodies[0].linearVelocity = Vec3(0, -1, 0);
    bodies[1] = makeDynamicBody(Vec3(0, 1, 0), 1.0f, 0.5f, 0.0f, 0.5f, 1);
    bodies[2] = makeStaticBody(Vec3(0, 0, 0), 0.5f, 2);

    PersistentManifold manifolds[2];
    manifolds[0] = makeOneContactManifold(0, 1, Vec3(0, 1.5f, 0), Vec3(0, 1.5f, 0), Vec3(0, 1, 0), 0.01f);
    manifolds[1] = makeOneContactManifold(1, 2, Vec3(0, 0.5f, 0), Vec3(0, 0, 0), Vec3(0, 1, 0), 0.01f);

    SolverConfig config;
    config.warmStarting = false;

    SolverStats stats = solve(bodies, 3, manifolds, 2, config, 1.0f / 60.0f);

    TEST_ASSERT(stats.totalContacts == 2);
    TEST_ASSERT(stats.totalManifolds == 2);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 11: Top-level solve() API
// ═════════════════════════════════════════════════════════════════════════════

static bool test_solve_returns_stats() {
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(0, 0.49f, 0), 1.0f, 0.5f, 0.0f, 0.0f, 0);
    bodies[0].linearVelocity = Vec3(0, -2.0f, 0);
    bodies[1] = makeStaticBody(Vec3(0, 0, 0), 0.0f, 1);

    PersistentManifold pm = makeOneContactManifold(
        0, 1, Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 1, 0), 0.01f);

    SolverConfig config;
    config.warmStarting = false;

    SolverStats stats = solve(bodies, 2, &pm, 1, config, 1.0f / 60.0f);

    TEST_ASSERT(stats.totalContacts == 1);
    TEST_ASSERT(stats.totalManifolds == 1);
    TEST_ASSERT(stats.velocityIterationsUsed == config.velocityIterations);
    TEST_ASSERT(stats.positionIterationsUsed > 0);
    return true;
}

static bool test_solve_empty_manifolds() {
    SolverBody bodies[1];
    bodies[0] = makeDynamicBody(Vec3::zero(), 1.0f, 0.5f, 0.0f, 0.0f, 0);

    SolverStats stats = solve(bodies, 1, nullptr, 0, SolverConfig(), 1.0f / 60.0f);

    TEST_ASSERT(stats.totalContacts == 0);
    TEST_ASSERT(stats.positionSolved == true);
    return true;
}

static bool test_solve_stores_impulses() {
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(0, 0.49f, 0), 1.0f, 0.5f, 0.0f, 0.5f, 0);
    bodies[0].linearVelocity = Vec3(0, -3.0f, 0);
    bodies[1] = makeStaticBody(Vec3(0, 0, 0), 0.5f, 1);

    PersistentManifold pm = makeOneContactManifold(
        0, 1, Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 1, 0), 0.01f);

    SolverConfig config;
    config.warmStarting = false;

    solve(bodies, 2, &pm, 1, config, 1.0f / 60.0f);

    // After solving, impulses should be stored in the manifold
    TEST_ASSERT(pm.contacts[0].normalImpulse > 0.0f);
    TEST_ASSERT(hasFlag(pm.contacts[0].flags, ContactFlags::HasImpulse));
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 12: Edge Cases
// ═════════════════════════════════════════════════════════════════════════════

static bool test_edge_two_static_bodies() {
    // Two static bodies — nothing should happen
    SolverBody bodies[2];
    bodies[0] = makeStaticBody(Vec3(-0.5f, 0, 0), 0.5f, 0);
    bodies[1] = makeStaticBody(Vec3(0.5f, 0, 0), 0.5f, 1);

    PersistentManifold pm = makeOneContactManifold(
        0, 1, Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(1, 0, 0), 0.01f);

    SolverConfig config;
    config.warmStarting = false;

    solve(bodies, 2, &pm, 1, config, 1.0f / 60.0f);

    TEST_ASSERT(approxVec(bodies[0].linearVelocity, Vec3::zero()));
    TEST_ASSERT(approxVec(bodies[1].linearVelocity, Vec3::zero()));
    return true;
}

static bool test_edge_single_iteration() {
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(0, 0.49f, 0), 1.0f, 0.5f, 0.0f, 0.0f, 0);
    bodies[0].linearVelocity = Vec3(0, -2.0f, 0);
    bodies[1] = makeStaticBody(Vec3(0, 0, 0), 0.0f, 1);

    PersistentManifold pm = makeOneContactManifold(
        0, 1, Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 1, 0), 0.01f);

    SolverConfig config(1, 1, 0.2f, 0.005f, 1.0f, 0.2f, false);

    SolverStats stats = solve(bodies, 2, &pm, 1, config, 1.0f / 60.0f);

    TEST_ASSERT(stats.velocityIterationsUsed == 1);
    // Even 1 iteration should help
    TEST_ASSERT(bodies[0].linearVelocity.getY() > -2.0f);
    return true;
}

static bool test_edge_zero_dt() {
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3::zero(), 1.0f, 0.5f, 0.0f, 0.0f, 0);
    bodies[1] = makeStaticBody(Vec3(0, -1, 0), 0.0f, 1);

    PersistentManifold pm = makeOneContactManifold(
        0, 1, Vec3(0, 0, 0), Vec3(0, -1, 0), Vec3(0, 1, 0), 0.01f);

    // Zero dt should not crash
    SolverStats stats = solve(bodies, 2, &pm, 1, SolverConfig(), 0.0f);
    TEST_ASSERT(stats.totalContacts == 1);
    return true;
}

static bool test_edge_very_large_penetration() {
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(0, -5, 0), 1.0f, 0.5f, 0.0f, 0.0f, 0);
    bodies[1] = makeStaticBody(Vec3(0, 0, 0), 0.0f, 1);

    PersistentManifold pm = makeOneContactManifold(
        0, 1, Vec3(0, -5, 0), Vec3(0, 0, 0), Vec3(0, 1, 0), 5.0f);

    SolverConfig config;
    config.warmStarting = false;

    // Should not crash even with extreme penetration
    SolverStats stats = solve(bodies, 2, &pm, 1, config, 1.0f / 60.0f);
    TEST_ASSERT(stats.totalContacts == 1);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 13: Helper Functions
// ═════════════════════════════════════════════════════════════════════════════

static bool test_integrate_positions() {
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(0, 0, 0), 1.0f, 0.5f, 0.0f, 0.0f, 0);
    bodies[0].linearVelocity = Vec3(10.0f, 0, 0);
    bodies[1] = makeStaticBody(Vec3(0, 0, 0), 0.0f, 1);

    float dt = 0.1f;
    integratePositions(bodies, 2, dt);

    // Dynamic body: x += 10 * 0.1 = 1.0
    TEST_ASSERT(approx(bodies[0].position.getX(), 1.0f));
    // Static body should not move
    TEST_ASSERT(approxVec(bodies[1].position, Vec3::zero()));
    return true;
}

static bool test_apply_gravity() {
    SolverBody bodies[2];
    bodies[0] = makeDynamicBody(Vec3(0, 10, 0), 1.0f, 0.5f, 0.0f, 0.0f, 0);
    bodies[1] = makeStaticBody(Vec3(0, 0, 0), 0.0f, 1);

    Vec3 gravity(0, -9.81f, 0);
    float dt = 1.0f / 60.0f;

    applyGravity(bodies, 2, gravity, dt);

    // Dynamic body should have gained downward velocity
    TEST_ASSERT(bodies[0].linearVelocity.getY() < 0.0f);
    TEST_ASSERT(approx(bodies[0].linearVelocity.getY(), -9.81f / 60.0f, 0.01f));
    // Static body unaffected
    TEST_ASSERT(approxVec(bodies[1].linearVelocity, Vec3::zero()));
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("========================================\n");
    std::printf("  Pulse Solver Module — Unit Tests\n");
    std::printf("========================================\n\n");

    // Group 1: SolverConfig
    std::printf("--- SolverConfig ---\n");
    RUN_TEST(test_config_defaults);
    RUN_TEST(test_config_custom);

    // Group 2: SolverBody
    std::printf("--- SolverBody ---\n");
    RUN_TEST(test_body_default_is_static);
    RUN_TEST(test_body_dynamic);
    RUN_TEST(test_body_apply_linear_impulse);
    RUN_TEST(test_body_apply_angular_impulse);
    RUN_TEST(test_body_static_ignores_impulse);
    RUN_TEST(test_body_convenience_constructor);

    // Group 3: Constraint Types
    std::printf("--- Constraint Types ---\n");
    RUN_TEST(test_constraint_header_default);
    RUN_TEST(test_constraint_header_custom);
    RUN_TEST(test_constraint_enable_disable);
    RUN_TEST(test_velocity_constraint_default);
    RUN_TEST(test_position_constraint_default);
    RUN_TEST(test_solver_stats_default);
    RUN_TEST(test_contact_constraint_group_default);

    // Group 4: ContactSolver Initialization
    std::printf("--- ContactSolver Initialization ---\n");
    RUN_TEST(test_solver_init_empty);
    RUN_TEST(test_solver_init_one_contact);
    RUN_TEST(test_solver_init_multiple_contacts);
    RUN_TEST(test_solver_init_body_lookup);

    // Group 5: Contact Resolution
    std::printf("--- Contact Resolution ---\n");
    RUN_TEST(test_two_spheres_separate);
    RUN_TEST(test_sphere_on_static_plane);
    RUN_TEST(test_normal_impulse_non_negative);

    // Group 6: Restitution
    std::printf("--- Restitution ---\n");
    RUN_TEST(test_restitution_bounce);
    RUN_TEST(test_restitution_zero_no_bounce);
    RUN_TEST(test_restitution_threshold);

    // Group 7: Friction
    std::printf("--- Friction ---\n");
    RUN_TEST(test_friction_slows_sliding);
    RUN_TEST(test_zero_friction_no_slowdown);
    RUN_TEST(test_friction_cone_limit);

    // Group 8: Split-Impulse Position Correction
    std::printf("--- Split-Impulse Position ---\n");
    RUN_TEST(test_position_correction_resolves_overlap);
    RUN_TEST(test_position_no_velocity_artifact);

    // Group 9: Warm Starting
    std::printf("--- Warm Starting ---\n");
    RUN_TEST(test_warm_start_uses_cached_impulses);
    RUN_TEST(test_warm_start_disabled);

    // Group 10: Multi-Contact Scenarios
    std::printf("--- Multi-Contact ---\n");
    RUN_TEST(test_box_on_plane_4_contacts);
    RUN_TEST(test_multiple_manifolds);

    // Group 11: Top-level solve() API
    std::printf("--- solve() API ---\n");
    RUN_TEST(test_solve_returns_stats);
    RUN_TEST(test_solve_empty_manifolds);
    RUN_TEST(test_solve_stores_impulses);

    // Group 12: Edge Cases
    std::printf("--- Edge Cases ---\n");
    RUN_TEST(test_edge_two_static_bodies);
    RUN_TEST(test_edge_single_iteration);
    RUN_TEST(test_edge_zero_dt);
    RUN_TEST(test_edge_very_large_penetration);

    // Group 13: Helper Functions
    std::printf("--- Helper Functions ---\n");
    RUN_TEST(test_integrate_positions);
    RUN_TEST(test_apply_gravity);

    // Summary
    std::printf("\n=== Results: %d/%d passed", g_passed, g_total);
    if (g_failed > 0) std::printf(", %d FAILED", g_failed);
    std::printf(" ===\n");

    return (g_failed > 0) ? 1 : 0;
}
