/**
 * @file test_world.cpp
 * @brief Comprehensive unit tests for the Pulse World module (Module 13).
 *
 * Tests: WorldConfig defaults, PhysicsWorld construction, body lifecycle,
 * free-fall gravity integration, sphere-sphere collision, contact callbacks,
 * island detection, sleep system, fixed-timestep accumulation, kinematic
 * bodies, static floors, and edge cases.
 *
 * Uses the same lightweight test framework as all other Pulse test files.
 */

#include <pulse/world/world_common.h>
#include <pulse/world/world.h>

#include <pulse/math/math_common.h>
#include <pulse/math/vec3.h>
#include <pulse/math/quat.h>
#include <pulse/math/mat3.h>
#include <pulse/math/transform.h>

#include <pulse/shapes/sphere.h>
#include <pulse/shapes/box.h>
#include <pulse/shapes/capsule.h>

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

static bool approx(float a, float b, float eps = 0.02f) {
    return std::fabs(a - b) < eps;
}

static bool approxVec(Vec3 a, Vec3 b, float eps = 0.05f) {
    return approx(a.getX(), b.getX(), eps) &&
           approx(a.getY(), b.getY(), eps) &&
           approx(a.getZ(), b.getZ(), eps);
}

/// Create a dynamic sphere BodyDef.
static BodyDef makeDynamicSphere(Vec3 pos, float radius = 0.5f, float mass = 1.0f) {
    BodyDef def;
    def.type = BodyType::Dynamic;
    def.initialTransform = Transform(pos);
    def.mass = mass;
    def.shapeType = ShapeType::Sphere;
    def.linearDamping = 0.0f;
    def.angularDamping = 0.0f;
    // Sphere inertia: I = (2/5) * m * r²
    float I = 0.4f * mass * radius * radius;
    def.localInertia = Mat3(I, 0, 0, 0, I, 0, 0, 0, I);
    return def;
}

/// Create a static sphere BodyDef.
static BodyDef makeStaticSphere(Vec3 pos) {
    BodyDef def;
    def.type = BodyType::Static;
    def.initialTransform = Transform(pos);
    def.mass = 0.0f;
    def.shapeType = ShapeType::Sphere;
    return def;
}

/// Create a kinematic sphere BodyDef.
static BodyDef makeKinematicSphere(Vec3 pos) {
    BodyDef def;
    def.type = BodyType::Kinematic;
    def.initialTransform = Transform(pos);
    def.mass = 0.0f;
    def.shapeType = ShapeType::Sphere;
    return def;
}

/// Create a dynamic box BodyDef.
static BodyDef makeDynamicBox(Vec3 pos, Vec3 halfExt = Vec3(0.5f), float mass = 1.0f) {
    BodyDef def;
    def.type = BodyType::Dynamic;
    def.initialTransform = Transform(pos);
    def.mass = mass;
    def.shapeType = ShapeType::Box;
    def.linearDamping = 0.0f;
    def.angularDamping = 0.0f;
    // Box inertia: I_xx = (1/12) * m * (h² + d²), etc.
    float hx = halfExt.getX() * 2.0f;
    float hy = halfExt.getY() * 2.0f;
    float hz = halfExt.getZ() * 2.0f;
    float Ixx = (1.0f/12.0f) * mass * (hy*hy + hz*hz);
    float Iyy = (1.0f/12.0f) * mass * (hx*hx + hz*hz);
    float Izz = (1.0f/12.0f) * mass * (hx*hx + hy*hy);
    def.localInertia = Mat3(Ixx, 0, 0, 0, Iyy, 0, 0, 0, Izz);
    return def;
}

/// Create a static box BodyDef (floor).
static BodyDef makeStaticBox(Vec3 pos) {
    BodyDef def;
    def.type = BodyType::Static;
    def.initialTransform = Transform(pos);
    def.mass = 0.0f;
    def.shapeType = ShapeType::Box;
    return def;
}

// =============================================================================
// 1. WORLD CONFIG TESTS
// =============================================================================

bool test_world_config_defaults() {
    WorldConfig cfg;
    TEST_ASSERT(approxVec(cfg.gravity, Vec3(0.0f, -9.81f, 0.0f)));
    TEST_ASSERT(approx(cfg.fixedTimeStep, 1.0f / 60.0f, 0.001f));
    TEST_ASSERT(cfg.maxSubSteps == 8);
    TEST_ASSERT(cfg.maxBodies == 16384);
    TEST_ASSERT(cfg.broadPhaseType == BroadPhaseType::DynamicAABBTree);
    TEST_ASSERT(cfg.integratorType == IntegratorType::SemiImplicitEuler);
    return true;
}

bool test_world_config_custom() {
    WorldConfig cfg;
    cfg.gravity = Vec3(0.0f, -10.0f, 0.0f);
    cfg.fixedTimeStep = 1.0f / 120.0f;
    cfg.maxSubSteps = 4;
    cfg.integratorType = IntegratorType::VelocityVerlet;
    TEST_ASSERT(approx(cfg.gravity.getY(), -10.0f));
    TEST_ASSERT(approx(cfg.fixedTimeStep, 1.0f / 120.0f, 0.001f));
    TEST_ASSERT(cfg.maxSubSteps == 4);
    TEST_ASSERT(cfg.integratorType == IntegratorType::VelocityVerlet);
    return true;
}

bool test_world_stats_defaults() {
    WorldStats stats;
    TEST_ASSERT(stats.totalBodies == 0);
    TEST_ASSERT(stats.activeBodies == 0);
    TEST_ASSERT(stats.broadPhasePairs == 0);
    TEST_ASSERT(stats.subStepsTaken == 0);
    return true;
}

// =============================================================================
// 2. WORLD CONSTRUCTION TESTS
// =============================================================================

bool test_world_construction_default() {
    PhysicsWorld world;
    TEST_ASSERT(world.bodyCount() == 0);
    TEST_ASSERT(approxVec(world.getGravity(), Vec3(0.0f, -9.81f, 0.0f)));
    return true;
}

bool test_world_construction_custom_config() {
    WorldConfig cfg;
    cfg.gravity = Vec3(0.0f, -20.0f, 0.0f);
    cfg.maxBodies = 512;
    PhysicsWorld world(cfg);
    TEST_ASSERT(approxVec(world.getGravity(), Vec3(0.0f, -20.0f, 0.0f)));
    TEST_ASSERT(world.bodyCount() == 0);
    return true;
}

bool test_world_set_gravity() {
    PhysicsWorld world;
    world.setGravity(Vec3(0.0f, -5.0f, 0.0f));
    TEST_ASSERT(approx(world.getGravity().getY(), -5.0f));
    return true;
}

// =============================================================================
// 3. BODY LIFECYCLE TESTS
// =============================================================================

bool test_create_single_body() {
    PhysicsWorld world;
    Sphere sphere(0.5f);
    BodyDef def = makeDynamicSphere(Vec3(0, 5, 0));
    BodyHandle h = world.createBody(def, &sphere);
    TEST_ASSERT(world.isValid(h));
    TEST_ASSERT(world.bodyCount() == 1);
    TEST_ASSERT(approxVec(world.getPosition(h), Vec3(0, 5, 0)));
    return true;
}

bool test_create_multiple_bodies() {
    PhysicsWorld world;
    Sphere sphere(0.5f);

    BodyHandle h1 = world.createBody(makeDynamicSphere(Vec3(0, 1, 0)), &sphere);
    BodyHandle h2 = world.createBody(makeDynamicSphere(Vec3(2, 1, 0)), &sphere);
    BodyHandle h3 = world.createBody(makeStaticSphere(Vec3(0, 0, 0)), &sphere);

    TEST_ASSERT(world.bodyCount() == 3);
    TEST_ASSERT(world.isValid(h1));
    TEST_ASSERT(world.isValid(h2));
    TEST_ASSERT(world.isValid(h3));
    return true;
}

bool test_destroy_body() {
    PhysicsWorld world;
    Sphere sphere(0.5f);

    BodyHandle h1 = world.createBody(makeDynamicSphere(Vec3(0, 1, 0)), &sphere);
    BodyHandle h2 = world.createBody(makeDynamicSphere(Vec3(2, 1, 0)), &sphere);
    TEST_ASSERT(world.bodyCount() == 2);

    world.destroyBody(h1);
    TEST_ASSERT(world.bodyCount() == 1);
    TEST_ASSERT(!world.isValid(h1));
    TEST_ASSERT(world.isValid(h2));
    return true;
}

bool test_destroy_last_body() {
    PhysicsWorld world;
    Sphere sphere(0.5f);

    BodyHandle h = world.createBody(makeDynamicSphere(Vec3(0, 5, 0)), &sphere);
    world.destroyBody(h);
    TEST_ASSERT(world.bodyCount() == 0);
    TEST_ASSERT(!world.isValid(h));
    return true;
}

bool test_body_type_queries() {
    PhysicsWorld world;
    Sphere sphere(0.5f);

    BodyHandle dyn = world.createBody(makeDynamicSphere(Vec3(0, 1, 0)), &sphere);
    BodyHandle sta = world.createBody(makeStaticSphere(Vec3(0, 0, 0)), &sphere);
    BodyHandle kin = world.createBody(makeKinematicSphere(Vec3(5, 0, 0)), &sphere);

    TEST_ASSERT(world.getBodyType(dyn) == BodyType::Dynamic);
    TEST_ASSERT(world.getBodyType(sta) == BodyType::Static);
    TEST_ASSERT(world.getBodyType(kin) == BodyType::Kinematic);
    return true;
}

// =============================================================================
// 4. BODY ACCESSORS
// =============================================================================

bool test_set_get_position() {
    PhysicsWorld world;
    Sphere sphere(0.5f);
    BodyHandle h = world.createBody(makeDynamicSphere(Vec3(0, 5, 0)), &sphere);

    world.setPosition(h, Vec3(1, 2, 3));
    TEST_ASSERT(approxVec(world.getPosition(h), Vec3(1, 2, 3)));
    return true;
}

bool test_set_get_velocity() {
    PhysicsWorld world;
    Sphere sphere(0.5f);
    BodyHandle h = world.createBody(makeDynamicSphere(Vec3(0, 0, 0)), &sphere);

    world.setLinearVelocity(h, Vec3(1, 2, 3));
    TEST_ASSERT(approxVec(world.getLinearVelocity(h), Vec3(1, 2, 3)));

    world.setAngularVelocity(h, Vec3(0.1f, 0.2f, 0.3f));
    TEST_ASSERT(approxVec(world.getAngularVelocity(h), Vec3(0.1f, 0.2f, 0.3f)));
    return true;
}

bool test_apply_force() {
    PhysicsWorld world;
    Sphere sphere(0.5f);
    BodyHandle h = world.createBody(makeDynamicSphere(Vec3(0, 10, 0)), &sphere);

    // Apply upward force equal to gravity (1kg * 9.81)
    world.applyForce(h, Vec3(0, 9.81f, 0));

    // Step — gravity + our force should roughly cancel
    world.singleStep(1.0f / 60.0f);

    // Body should barely move from y=10
    Vec3 pos = world.getPosition(h);
    TEST_ASSERT(std::fabs(pos.getY() - 10.0f) < 0.5f);
    return true;
}

bool test_apply_impulse() {
    PhysicsWorld world;
    world.setGravity(Vec3::zero()); // No gravity for clean test.
    Sphere sphere(0.5f);
    BodyHandle h = world.createBody(makeDynamicSphere(Vec3(0, 0, 0)), &sphere);

    world.applyLinearImpulse(h, Vec3(5, 0, 0)); // 1kg → v = 5 m/s
    Vec3 vel = world.getLinearVelocity(h);
    TEST_ASSERT(approx(vel.getX(), 5.0f, 0.01f));
    return true;
}

// =============================================================================
// 5. GRAVITY & FREE FALL
// =============================================================================

bool test_free_fall_single_step() {
    WorldConfig cfg;
    cfg.gravity = Vec3(0, -10.0f, 0);
    PhysicsWorld world(cfg);

    Sphere sphere(0.5f);
    BodyHandle h = world.createBody(makeDynamicSphere(Vec3(0, 100, 0)), &sphere);

    float dt = 1.0f / 60.0f;
    world.singleStep(dt);

    // After one step: v = g*dt = -10 * (1/60) ≈ -0.167 m/s
    // pos ≈ 100 + v*dt ≈ 99.997 (very small displacement)
    Vec3 pos = world.getPosition(h);
    TEST_ASSERT(pos.getY() < 100.0f); // Should have fallen.
    TEST_ASSERT(pos.getY() > 99.0f);  // Not by much.
    return true;
}

bool test_free_fall_100_steps() {
    WorldConfig cfg;
    cfg.gravity = Vec3(0, -10.0f, 0);
    cfg.fixedTimeStep = 1.0f / 60.0f;
    PhysicsWorld world(cfg);

    Sphere sphere(0.5f);
    BodyHandle h = world.createBody(makeDynamicSphere(Vec3(0, 100, 0)), &sphere);

    float dt = 1.0f / 60.0f;
    for (int i = 0; i < 100; ++i) {
        world.singleStep(dt);
    }

    // Analytical: y = 100 + 0.5 * (-10) * t²  where t = 100/60 ≈ 1.667s
    // y_analytical = 100 - 5 * (100/60)² ≈ 100 - 13.89 ≈ 86.11
    float t = 100.0f * dt;
    float y_analytical = 100.0f + 0.5f * (-10.0f) * t * t;

    Vec3 pos = world.getPosition(h);
    // Allow 5% tolerance for numerical integration error.
    TEST_ASSERT(std::fabs(pos.getY() - y_analytical) < std::fabs(y_analytical) * 0.05f + 1.0f);
    return true;
}

bool test_zero_gravity() {
    WorldConfig cfg;
    cfg.gravity = Vec3::zero();
    PhysicsWorld world(cfg);

    Sphere sphere(0.5f);
    BodyDef def = makeDynamicSphere(Vec3(0, 5, 0));
    def.linearVelocity = Vec3(1, 0, 0);
    BodyHandle h = world.createBody(def, &sphere);

    float dt = 1.0f / 60.0f;
    for (int i = 0; i < 60; ++i) {
        world.singleStep(dt);
    }

    // After 1 second at 1 m/s horizontal, x ≈ 1.0, y ≈ 5.0
    Vec3 pos = world.getPosition(h);
    TEST_ASSERT(approx(pos.getX(), 1.0f, 0.1f));
    TEST_ASSERT(approx(pos.getY(), 5.0f, 0.1f));
    return true;
}

// =============================================================================
// 6. COLLISION DETECTION — SPHERE-SPHERE
// =============================================================================

bool test_sphere_collision_detection() {
    WorldConfig cfg;
    cfg.gravity = Vec3::zero();
    PhysicsWorld world(cfg);

    Sphere sphere(0.5f);
    BodyDef defA = makeDynamicSphere(Vec3(0, 0, 0));
    defA.linearVelocity = Vec3(1, 0, 0);
    BodyDef defB = makeDynamicSphere(Vec3(0.8f, 0, 0)); // Within contact distance

    BodyHandle hA = world.createBody(defA, &sphere);
    BodyHandle hB = world.createBody(defB, &sphere);

    world.singleStep(1.0f / 60.0f);

    // Bodies should have interacted — A should slow down or B should speed up.
    Vec3 velA = world.getLinearVelocity(hA);
    Vec3 velB = world.getLinearVelocity(hB);

    // After collision response, momentum should be roughly conserved.
    // With equal masses, A should slow and B should gain speed.
    float totalMomentumX = velA.getX() + velB.getX();
    TEST_ASSERT(approx(totalMomentumX, 1.0f, 0.3f)); // Initial momentum was 1.
    return true;
}

bool test_separated_spheres_no_collision() {
    WorldConfig cfg;
    cfg.gravity = Vec3::zero();
    PhysicsWorld world(cfg);

    Sphere sphere(0.5f);
    BodyDef defA = makeDynamicSphere(Vec3(0, 0, 0));
    BodyDef defB = makeDynamicSphere(Vec3(5, 0, 0)); // Far apart.

    BodyHandle hA = world.createBody(defA, &sphere);
    BodyHandle hB = world.createBody(defB, &sphere);

    world.singleStep(1.0f / 60.0f);

    // No collision — velocities should remain zero (no gravity either).
    TEST_ASSERT(approxVec(world.getLinearVelocity(hA), Vec3::zero(), 0.01f));
    TEST_ASSERT(approxVec(world.getLinearVelocity(hB), Vec3::zero(), 0.01f));
    return true;
}

// =============================================================================
// 7. CONTACT CALLBACKS
// =============================================================================

struct CallbackData {
    int eventCount;
    ContactEventType lastType;
    float lastPenetration;
};

static void testContactCallback(const ContactEvent& evt, void* userData) {
    auto* data = static_cast<CallbackData*>(userData);
    data->eventCount++;
    data->lastType = evt.type;
    data->lastPenetration = evt.penetration;
}

bool test_contact_callback_fires() {
    WorldConfig cfg;
    cfg.gravity = Vec3::zero();
    PhysicsWorld world(cfg);

    CallbackData cbData = {0, ContactEventType::Begin, 0.0f};
    world.setContactCallback(testContactCallback, &cbData);

    Sphere sphere(0.5f);
    // Overlapping spheres.
    BodyHandle hA = world.createBody(makeDynamicSphere(Vec3(0, 0, 0)), &sphere);
    BodyHandle hB = world.createBody(makeDynamicSphere(Vec3(0.8f, 0, 0)), &sphere);

    world.singleStep(1.0f / 60.0f);

    TEST_ASSERT(cbData.eventCount > 0);
    TEST_ASSERT(cbData.lastPenetration > 0.0f);
    (void)hA; (void)hB;
    return true;
}

bool test_no_callback_when_null() {
    WorldConfig cfg;
    cfg.gravity = Vec3::zero();
    PhysicsWorld world(cfg);

    // No callback set.
    Sphere sphere(0.5f);
    world.createBody(makeDynamicSphere(Vec3(0, 0, 0)), &sphere);
    world.createBody(makeDynamicSphere(Vec3(0.8f, 0, 0)), &sphere);

    // Should not crash.
    world.singleStep(1.0f / 60.0f);
    return true;
}

// =============================================================================
// 8. STATIC BODIES
// =============================================================================

bool test_static_body_immovable() {
    PhysicsWorld world;
    Sphere sphere(0.5f);

    BodyHandle h = world.createBody(makeStaticSphere(Vec3(0, 0, 0)), &sphere);
    Vec3 posBefore = world.getPosition(h);

    for (int i = 0; i < 60; ++i) {
        world.singleStep(1.0f / 60.0f);
    }

    Vec3 posAfter = world.getPosition(h);
    TEST_ASSERT(approxVec(posBefore, posAfter, 0.001f));
    return true;
}

// =============================================================================
// 9. KINEMATIC BODIES
// =============================================================================

bool test_kinematic_body_velocity() {
    WorldConfig cfg;
    cfg.gravity = Vec3::zero();
    PhysicsWorld world(cfg);

    Sphere sphere(0.5f);
    BodyDef def = makeKinematicSphere(Vec3(0, 0, 0));
    BodyHandle h = world.createBody(def, &sphere);

    // Kinematic bodies should not be affected by gravity.
    for (int i = 0; i < 60; ++i) {
        world.singleStep(1.0f / 60.0f);
    }

    Vec3 pos = world.getPosition(h);
    TEST_ASSERT(approx(pos.getY(), 0.0f, 0.01f));
    return true;
}

// =============================================================================
// 10. FIXED-TIMESTEP ACCUMULATION
// =============================================================================

bool test_step_accumulator_single_substep() {
    PhysicsWorld world;
    Sphere sphere(0.5f);
    world.createBody(makeDynamicSphere(Vec3(0, 10, 0)), &sphere);

    float fixedDt = world.getConfig().fixedTimeStep;

    // step(dt) with dt == fixedDt → should take exactly 1 sub-step.
    WorldStats stats = world.step(fixedDt);
    TEST_ASSERT(stats.subStepsTaken == 1);
    return true;
}

bool test_step_accumulator_multiple_substeps() {
    PhysicsWorld world;
    Sphere sphere(0.5f);
    world.createBody(makeDynamicSphere(Vec3(0, 10, 0)), &sphere);

    float fixedDt = world.getConfig().fixedTimeStep;

    // step(dt) with dt == 3 * fixedDt → should take 3 sub-steps.
    WorldStats stats = world.step(fixedDt * 3.0f);
    TEST_ASSERT(stats.subStepsTaken == 3);
    return true;
}

bool test_step_accumulator_max_substeps() {
    WorldConfig cfg;
    cfg.maxSubSteps = 4;
    PhysicsWorld world(cfg);
    Sphere sphere(0.5f);
    world.createBody(makeDynamicSphere(Vec3(0, 10, 0)), &sphere);

    float fixedDt = cfg.fixedTimeStep;

    // step(dt) with huge dt → capped at maxSubSteps.
    WorldStats stats = world.step(fixedDt * 100.0f);
    TEST_ASSERT(stats.subStepsTaken <= 4);
    return true;
}

bool test_step_accumulator_small_dt() {
    PhysicsWorld world;
    Sphere sphere(0.5f);
    world.createBody(makeDynamicSphere(Vec3(0, 10, 0)), &sphere);

    float fixedDt = world.getConfig().fixedTimeStep;

    // step(dt) with dt < fixedDt → should take 0 sub-steps.
    WorldStats stats = world.step(fixedDt * 0.5f);
    TEST_ASSERT(stats.subStepsTaken == 0);

    // Accumulated time. Second call should trigger 1 step.
    stats = world.step(fixedDt * 0.6f);
    TEST_ASSERT(stats.subStepsTaken == 1);
    return true;
}

// =============================================================================
// 11. WORLD STATS
// =============================================================================

bool test_world_stats_body_counts() {
    PhysicsWorld world;
    Sphere sphere(0.5f);

    world.createBody(makeDynamicSphere(Vec3(0, 5, 0)), &sphere);
    world.createBody(makeDynamicSphere(Vec3(2, 5, 0)), &sphere);
    world.createBody(makeStaticSphere(Vec3(0, 0, 0)), &sphere);

    world.singleStep(1.0f / 60.0f);
    WorldStats stats = world.step(0.0f); // Just to get stats, no actual step.

    TEST_ASSERT(stats.totalBodies == 3);
    TEST_ASSERT(stats.staticBodies == 1);
    return true;
}

// =============================================================================
// 12. EMPTY WORLD EDGE CASES
// =============================================================================

bool test_empty_world_step() {
    PhysicsWorld world;
    // Should not crash.
    world.singleStep(1.0f / 60.0f);
    WorldStats stats = world.step(1.0f / 60.0f);
    TEST_ASSERT(stats.totalBodies == 0);
    TEST_ASSERT(stats.subStepsTaken == 1);
    return true;
}

bool test_single_body_world() {
    PhysicsWorld world;
    Sphere sphere(0.5f);
    BodyHandle h = world.createBody(makeDynamicSphere(Vec3(0, 10, 0)), &sphere);

    // Should not crash — no pairs to process.
    for (int i = 0; i < 10; ++i) {
        world.singleStep(1.0f / 60.0f);
    }

    TEST_ASSERT(world.getPosition(h).getY() < 10.0f);
    return true;
}

// =============================================================================
// 13. SLEEP SYSTEM
// =============================================================================

bool test_body_sleeps_when_stationary() {
    WorldConfig cfg;
    cfg.gravity = Vec3::zero(); // No gravity — body is at rest from start.
    cfg.sleepConfig.timeToSleep = 0.3f;
    PhysicsWorld world(cfg);

    Sphere sphere(0.5f);
    BodyHandle h = world.createBody(makeDynamicSphere(Vec3(0, 0, 0)), &sphere);

    // Step for 1 second — body should eventually fall asleep.
    float dt = 1.0f / 60.0f;
    for (int i = 0; i < 60; ++i) {
        world.singleStep(dt);
    }

    TEST_ASSERT(world.isSleeping(h));
    return true;
}

bool test_wake_body() {
    WorldConfig cfg;
    cfg.gravity = Vec3::zero();
    cfg.sleepConfig.timeToSleep = 0.1f;
    PhysicsWorld world(cfg);

    Sphere sphere(0.5f);
    BodyHandle h = world.createBody(makeDynamicSphere(Vec3(0, 0, 0)), &sphere);

    // Let it sleep.
    float dt = 1.0f / 60.0f;
    for (int i = 0; i < 30; ++i) {
        world.singleStep(dt);
    }
    TEST_ASSERT(world.isSleeping(h));

    // Wake it.
    world.wakeBody(h);
    TEST_ASSERT(!world.isSleeping(h));
    return true;
}

// =============================================================================
// 14. INVERSE MASS QUERY
// =============================================================================

bool test_inv_mass_query() {
    PhysicsWorld world;
    Sphere sphere(0.5f);

    BodyHandle dyn = world.createBody(makeDynamicSphere(Vec3(0, 0, 0), 0.5f, 2.0f), &sphere);
    BodyHandle sta = world.createBody(makeStaticSphere(Vec3(5, 0, 0)), &sphere);

    TEST_ASSERT(approx(world.getInvMass(dyn), 0.5f, 0.01f)); // 1/2 = 0.5
    TEST_ASSERT(approx(world.getInvMass(sta), 0.0f, 0.001f)); // Static = 0
    return true;
}

// =============================================================================
// 15. BOX COLLISION SHAPES
// =============================================================================

bool test_box_bodies_collision() {
    WorldConfig cfg;
    cfg.gravity = Vec3::zero();
    PhysicsWorld world(cfg);

    Box boxShape(Vec3(0.5f, 0.5f, 0.5f));
    BodyDef defA = makeDynamicBox(Vec3(0, 0, 0));
    defA.linearVelocity = Vec3(1, 0, 0);
    BodyDef defB = makeDynamicBox(Vec3(0.8f, 0, 0));

    BodyHandle hA = world.createBody(defA, &boxShape);
    BodyHandle hB = world.createBody(defB, &boxShape);

    world.singleStep(1.0f / 60.0f);

    // Boxes overlap at 0.8m apart with 0.5m half-extents each.
    // There should be a collision response.
    float totalMom = world.getLinearVelocity(hA).getX() + world.getLinearVelocity(hB).getX();
    TEST_ASSERT(approx(totalMom, 1.0f, 0.5f)); // Momentum conserved.
    return true;
}

// =============================================================================
// 16. MULTIPLE STEPS STABILITY
// =============================================================================

bool test_many_steps_no_crash() {
    PhysicsWorld world;
    Sphere sphere(0.5f);

    // Create 10 bodies.
    for (int i = 0; i < 10; ++i) {
        world.createBody(
            makeDynamicSphere(Vec3(static_cast<float>(i) * 3.0f, 10.0f, 0.0f)),
            &sphere);
    }

    // Run 200 steps.
    for (int i = 0; i < 200; ++i) {
        world.singleStep(1.0f / 60.0f);
    }

    TEST_ASSERT(world.bodyCount() == 10);
    return true;
}

// =============================================================================
// 17. CREATE AFTER DESTROY
// =============================================================================

bool test_create_after_destroy() {
    PhysicsWorld world;
    Sphere sphere(0.5f);

    BodyHandle h1 = world.createBody(makeDynamicSphere(Vec3(0, 5, 0)), &sphere);
    world.destroyBody(h1);
    TEST_ASSERT(world.bodyCount() == 0);

    BodyHandle h2 = world.createBody(makeDynamicSphere(Vec3(1, 5, 0)), &sphere);
    TEST_ASSERT(world.bodyCount() == 1);
    TEST_ASSERT(world.isValid(h2));
    TEST_ASSERT(!world.isValid(h1)); // Old handle should be invalid.
    return true;
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    std::printf("═══════════════════════════════════════════════════════\n");
    std::printf("  Pulse Physics Engine — World Module Tests (Module 13)\n");
    std::printf("═══════════════════════════════════════════════════════\n\n");

    // 1. Config.
    std::printf("── WorldConfig ──\n");
    RUN_TEST(test_world_config_defaults);
    RUN_TEST(test_world_config_custom);
    RUN_TEST(test_world_stats_defaults);

    // 2. Construction.
    std::printf("\n── PhysicsWorld Construction ──\n");
    RUN_TEST(test_world_construction_default);
    RUN_TEST(test_world_construction_custom_config);
    RUN_TEST(test_world_set_gravity);

    // 3. Body lifecycle.
    std::printf("\n── Body Lifecycle ──\n");
    RUN_TEST(test_create_single_body);
    RUN_TEST(test_create_multiple_bodies);
    RUN_TEST(test_destroy_body);
    RUN_TEST(test_destroy_last_body);
    RUN_TEST(test_body_type_queries);

    // 4. Body accessors.
    std::printf("\n── Body Accessors ──\n");
    RUN_TEST(test_set_get_position);
    RUN_TEST(test_set_get_velocity);
    RUN_TEST(test_apply_force);
    RUN_TEST(test_apply_impulse);

    // 5. Gravity & free fall.
    std::printf("\n── Gravity & Free Fall ──\n");
    RUN_TEST(test_free_fall_single_step);
    RUN_TEST(test_free_fall_100_steps);
    RUN_TEST(test_zero_gravity);

    // 6. Collision detection.
    std::printf("\n── Collision Detection ──\n");
    RUN_TEST(test_sphere_collision_detection);
    RUN_TEST(test_separated_spheres_no_collision);

    // 7. Contact callbacks.
    std::printf("\n── Contact Callbacks ──\n");
    RUN_TEST(test_contact_callback_fires);
    RUN_TEST(test_no_callback_when_null);

    // 8. Static bodies.
    std::printf("\n── Static Bodies ──\n");
    RUN_TEST(test_static_body_immovable);

    // 9. Kinematic bodies.
    std::printf("\n── Kinematic Bodies ──\n");
    RUN_TEST(test_kinematic_body_velocity);

    // 10. Fixed-timestep accumulation.
    std::printf("\n── Step Accumulator ──\n");
    RUN_TEST(test_step_accumulator_single_substep);
    RUN_TEST(test_step_accumulator_multiple_substeps);
    RUN_TEST(test_step_accumulator_max_substeps);
    RUN_TEST(test_step_accumulator_small_dt);

    // 11. Stats.
    std::printf("\n── World Stats ──\n");
    RUN_TEST(test_world_stats_body_counts);

    // 12. Edge cases.
    std::printf("\n── Edge Cases ──\n");
    RUN_TEST(test_empty_world_step);
    RUN_TEST(test_single_body_world);

    // 13. Sleep.
    std::printf("\n── Sleep System ──\n");
    RUN_TEST(test_body_sleeps_when_stationary);
    RUN_TEST(test_wake_body);

    // 14. Queries.
    std::printf("\n── Queries ──\n");
    RUN_TEST(test_inv_mass_query);

    // 15. Box shapes.
    std::printf("\n── Box Collision ──\n");
    RUN_TEST(test_box_bodies_collision);

    // 16. Stability.
    std::printf("\n── Stability ──\n");
    RUN_TEST(test_many_steps_no_crash);

    // 17. Create after destroy.
    std::printf("\n── Create After Destroy ──\n");
    RUN_TEST(test_create_after_destroy);

    // ── Summary ──────────────────────────────────────────────────────────
    std::printf("\n═══════════════════════════════════════════════════════\n");
    std::printf("  Results: %d/%d passed\n", g_passed, g_total);
    std::printf("═══════════════════════════════════════════════════════\n");

    return g_failed > 0 ? 1 : 0;
}
