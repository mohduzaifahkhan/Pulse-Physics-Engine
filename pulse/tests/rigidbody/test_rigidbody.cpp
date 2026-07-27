/**
 * @file test_rigidbody.cpp
 * @brief Comprehensive unit tests for the Pulse rigid body module (Module 11).
 *
 * Tests: BodyDef & common types, BodyManager creation/destruction,
 * SoA access, body type behaviour, world inertia, SolverBody conversion,
 * island detection, sleep management, and edge cases.
 */

#include <pulse/rigidbody/rigid_body_common.h>
#include <pulse/rigidbody/rigid_body.h>
#include <pulse/rigidbody/body_manager.h>
#include <pulse/rigidbody/island_manager.h>
#include <pulse/rigidbody/sleep_manager.h>

#include <pulse/solver/solver_common.h>
#include <pulse/math/math_common.h>
#include <pulse/math/vec3.h>
#include <pulse/math/quat.h>
#include <pulse/math/mat3.h>
#include <pulse/math/transform.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

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

/// Create a default dynamic BodyDef at a position.
static BodyDef makeDynamicDef(Vec3 pos, float mass = 1.0f) {
    BodyDef def;
    def.type = BodyType::Dynamic;
    def.initialTransform = Transform(pos);
    def.mass = mass;
    // Sphere inertia: I = (2/5) * m * r²  for r=0.5
    float I = 0.4f * mass * 0.5f * 0.5f;
    def.localInertia = Mat3(I, 0, 0,
                            0, I, 0,
                            0, 0, I);
    def.restitution = 0.3f;
    def.friction = 0.4f;
    return def;
}

/// Create a static BodyDef at a position.
static BodyDef makeStaticDef(Vec3 pos) {
    BodyDef def;
    def.type = BodyType::Static;
    def.initialTransform = Transform(pos);
    def.mass = 0.0f;
    return def;
}

/// Create a kinematic BodyDef at a position.
static BodyDef makeKinematicDef(Vec3 pos) {
    BodyDef def;
    def.type = BodyType::Kinematic;
    def.initialTransform = Transform(pos);
    def.mass = 0.0f;
    return def;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 1: BodyDef & Common Types
// ═════════════════════════════════════════════════════════════════════════════

static bool test_body_def_defaults() {
    BodyDef def;
    TEST_ASSERT(def.type == BodyType::Dynamic);
    TEST_ASSERT(approx(def.mass, 1.0f));
    TEST_ASSERT(approx(def.restitution, 0.3f));
    TEST_ASSERT(approx(def.friction, 0.4f));
    TEST_ASSERT(approx(def.linearDamping, 0.01f));
    TEST_ASSERT(approx(def.angularDamping, 0.01f));
    TEST_ASSERT(approx(def.gravityScale, 1.0f));
    TEST_ASSERT(def.collisionLayer == 0x0001);
    TEST_ASSERT(def.collisionMask == 0xFFFF);
    TEST_ASSERT(!def.enableCCD);
    TEST_ASSERT(!def.fixedRotation);
    TEST_ASSERT(!def.isBullet);
    TEST_ASSERT(!def.isSensor);
    TEST_ASSERT(def.startAwake);
    return true;
}

static bool test_body_handle_null() {
    BodyHandle h;
    TEST_ASSERT(h.isNull());
    TEST_ASSERT(!h.isValid());
    TEST_ASSERT(h == BodyHandle::null());
    return true;
}

static bool test_body_flags_operations() {
    BodyFlags flags = BodyFlags::Active | BodyFlags::EnableGravity;
    TEST_ASSERT(hasFlag(flags, BodyFlags::Active));
    TEST_ASSERT(hasFlag(flags, BodyFlags::EnableGravity));
    TEST_ASSERT(!hasFlag(flags, BodyFlags::Sleeping));

    flags |= BodyFlags::Sleeping;
    TEST_ASSERT(hasFlag(flags, BodyFlags::Sleeping));

    flags &= ~BodyFlags::Sleeping;
    TEST_ASSERT(!hasFlag(flags, BodyFlags::Sleeping));
    return true;
}

static bool test_body_type_enum() {
    TEST_ASSERT(static_cast<uint8_t>(BodyType::Static) == 0);
    TEST_ASSERT(static_cast<uint8_t>(BodyType::Dynamic) == 1);
    TEST_ASSERT(static_cast<uint8_t>(BodyType::Kinematic) == 2);
    return true;
}

static bool test_body_config_defaults() {
    BodyConfig config;
    TEST_ASSERT(config.maxBodies == 16384);
    TEST_ASSERT(approx(config.gravity.getY(), -9.81f));
    return true;
}

static bool test_body_stats_default() {
    BodyStats stats;
    TEST_ASSERT(stats.totalBodies == 0);
    TEST_ASSERT(stats.activeBodies == 0);
    TEST_ASSERT(stats.sleepingBodies == 0);
    TEST_ASSERT(stats.staticBodies == 0);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 2: BodyManager Creation & Destruction
// ═════════════════════════════════════════════════════════════════════════════

static bool test_create_single_body() {
    BodyManager mgr(64);
    BodyDef def = makeDynamicDef(Vec3(1, 2, 3));
    BodyHandle h = mgr.createBody(def);
    TEST_ASSERT(!h.isNull());
    TEST_ASSERT(mgr.isValid(h));
    TEST_ASSERT(mgr.bodyCount() == 1);
    return true;
}

static bool test_create_multiple_bodies() {
    BodyManager mgr(64);
    std::vector<BodyHandle> handles;
    for (int i = 0; i < 100; ++i) {
        BodyHandle h = mgr.createBody(makeDynamicDef(
            Vec3(static_cast<float>(i), 0, 0)));
        handles.push_back(h);
    }
    TEST_ASSERT(mgr.bodyCount() == 100);
    for (auto& h : handles) {
        TEST_ASSERT(mgr.isValid(h));
    }
    return true;
}

static bool test_destroy_body() {
    BodyManager mgr(64);
    BodyHandle h = mgr.createBody(makeDynamicDef(Vec3(1, 2, 3)));
    TEST_ASSERT(mgr.bodyCount() == 1);
    mgr.destroyBody(h);
    TEST_ASSERT(mgr.bodyCount() == 0);
    TEST_ASSERT(!mgr.isValid(h)); // Handle invalidated
    return true;
}

static bool test_destroy_middle_body() {
    BodyManager mgr(64);
    BodyHandle h0 = mgr.createBody(makeDynamicDef(Vec3(0, 0, 0)));
    BodyHandle h1 = mgr.createBody(makeDynamicDef(Vec3(1, 0, 0)));
    BodyHandle h2 = mgr.createBody(makeDynamicDef(Vec3(2, 0, 0)));
    TEST_ASSERT(mgr.bodyCount() == 3);

    mgr.destroyBody(h1); // Destroy middle — triggers swap-and-pop
    TEST_ASSERT(mgr.bodyCount() == 2);
    TEST_ASSERT(!mgr.isValid(h1));
    TEST_ASSERT(mgr.isValid(h0));
    TEST_ASSERT(mgr.isValid(h2));

    // The remaining bodies should still be accessible.
    Vec3 p0 = mgr.getPosition(h0);
    Vec3 p2 = mgr.getPosition(h2);
    TEST_ASSERT(approxVec(p0, Vec3(0, 0, 0)));
    TEST_ASSERT(approxVec(p2, Vec3(2, 0, 0)));
    return true;
}

static bool test_generation_invalidation() {
    BodyManager mgr(64);
    BodyHandle h1 = mgr.createBody(makeDynamicDef(Vec3::zero()));
    BodyHandle stale = h1; // Copy the handle
    mgr.destroyBody(h1);

    // stale should now be invalid
    TEST_ASSERT(!mgr.isValid(stale));

    // Create a new body — it reuses the same slot but with a new generation.
    BodyHandle h2 = mgr.createBody(makeDynamicDef(Vec3(5, 0, 0)));
    TEST_ASSERT(mgr.isValid(h2));
    TEST_ASSERT(!mgr.isValid(stale)); // Old handle still invalid
    return true;
}

static bool test_create_destroy_reuse() {
    BodyManager mgr(16);
    std::vector<BodyHandle> handles;

    // Create 10, destroy all, create 10 more.
    for (int i = 0; i < 10; ++i) {
        handles.push_back(mgr.createBody(makeDynamicDef(
            Vec3(static_cast<float>(i), 0, 0))));
    }
    for (auto& h : handles) {
        mgr.destroyBody(h);
    }
    TEST_ASSERT(mgr.bodyCount() == 0);

    handles.clear();
    for (int i = 0; i < 10; ++i) {
        handles.push_back(mgr.createBody(makeDynamicDef(
            Vec3(static_cast<float>(i + 10), 0, 0))));
    }
    TEST_ASSERT(mgr.bodyCount() == 10);
    for (auto& h : handles) {
        TEST_ASSERT(mgr.isValid(h));
    }
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 3: SoA Access
// ═════════════════════════════════════════════════════════════════════════════

static bool test_get_set_position() {
    BodyManager mgr(64);
    BodyHandle h = mgr.createBody(makeDynamicDef(Vec3(1, 2, 3)));
    TEST_ASSERT(approxVec(mgr.getPosition(h), Vec3(1, 2, 3)));

    mgr.setPosition(h, Vec3(4, 5, 6));
    TEST_ASSERT(approxVec(mgr.getPosition(h), Vec3(4, 5, 6)));
    return true;
}

static bool test_get_set_rotation() {
    BodyManager mgr(64);
    BodyHandle h = mgr.createBody(makeDynamicDef(Vec3::zero()));
    Quat r = mgr.getRotation(h);
    // Default should be identity.
    TEST_ASSERT(approx(r.getW(), 1.0f));

    Quat newRot = Quat::fromAxisAngle(Vec3::unitY(), math::Pi * 0.5f);
    mgr.setRotation(h, newRot);
    Quat got = mgr.getRotation(h);
    TEST_ASSERT(approx(got.getW(), newRot.getW(), 0.01f));
    return true;
}

static bool test_get_set_velocity() {
    BodyManager mgr(64);
    BodyDef def = makeDynamicDef(Vec3::zero());
    def.linearVelocity = Vec3(1, 2, 3);
    def.angularVelocity = Vec3(0, 0, 5);
    BodyHandle h = mgr.createBody(def);

    TEST_ASSERT(approxVec(mgr.getLinearVelocity(h), Vec3(1, 2, 3)));
    TEST_ASSERT(approxVec(mgr.getAngularVelocity(h), Vec3(0, 0, 5)));

    mgr.setLinearVelocity(h, Vec3(10, 0, 0));
    TEST_ASSERT(approxVec(mgr.getLinearVelocity(h), Vec3(10, 0, 0)));
    return true;
}

static bool test_apply_force() {
    BodyManager mgr(64);
    BodyHandle h = mgr.createBody(makeDynamicDef(Vec3::zero()));
    mgr.applyForce(h, Vec3(10, 0, 0));
    mgr.applyForce(h, Vec3(0, 5, 0));

    uint32_t idx = mgr.getIndex(h);
    Vec3 f = mgr.store().force(idx);
    TEST_ASSERT(approxVec(f, Vec3(10, 5, 0)));

    mgr.clearForces();
    f = mgr.store().force(idx);
    TEST_ASSERT(approxVec(f, Vec3::zero()));
    return true;
}

static bool test_apply_linear_impulse() {
    BodyManager mgr(64);
    BodyHandle h = mgr.createBody(makeDynamicDef(Vec3::zero(), 2.0f));

    // invMass = 0.5, impulse = (4, 0, 0) → Δv = (2, 0, 0)
    mgr.applyLinearImpulse(h, Vec3(4, 0, 0));
    Vec3 v = mgr.getLinearVelocity(h);
    TEST_ASSERT(approxVec(v, Vec3(2, 0, 0)));
    return true;
}

static bool test_apply_angular_impulse() {
    BodyManager mgr(64);
    BodyHandle h = mgr.createBody(makeDynamicDef(Vec3::zero(), 1.0f));

    mgr.applyAngularImpulse(h, Vec3(0, 1, 0));
    Vec3 w = mgr.getAngularVelocity(h);
    // Angular velocity should be non-zero along Y.
    TEST_ASSERT(std::fabs(w.getY()) > 0.01f);
    return true;
}

static bool test_apply_force_at_point() {
    BodyManager mgr(64);
    BodyHandle h = mgr.createBody(makeDynamicDef(Vec3::zero(), 1.0f));

    // Force at offset creates both force and torque.
    mgr.applyForceAtPoint(h, Vec3(0, 10, 0), Vec3(1, 0, 0));
    uint32_t idx = mgr.getIndex(h);
    Vec3 f = mgr.store().force(idx);
    Vec3 t = mgr.store().torque(idx);
    TEST_ASSERT(approxVec(f, Vec3(0, 10, 0)));
    TEST_ASSERT(t.lengthSquared() > 0.01f); // Should have torque
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 4: Body Type Behaviour
// ═════════════════════════════════════════════════════════════════════════════

static bool test_static_body_inv_mass() {
    BodyManager mgr(64);
    BodyHandle h = mgr.createBody(makeStaticDef(Vec3(0, 0, 0)));
    TEST_ASSERT(approx(mgr.getInvMass(h), 0.0f));
    TEST_ASSERT(mgr.getBodyType(h) == BodyType::Static);
    return true;
}

static bool test_kinematic_body_inv_mass() {
    BodyManager mgr(64);
    BodyHandle h = mgr.createBody(makeKinematicDef(Vec3(0, 0, 0)));
    TEST_ASSERT(approx(mgr.getInvMass(h), 0.0f));
    TEST_ASSERT(mgr.getBodyType(h) == BodyType::Kinematic);
    return true;
}

static bool test_dynamic_body_inv_mass() {
    BodyManager mgr(64);
    BodyHandle h = mgr.createBody(makeDynamicDef(Vec3::zero(), 4.0f));
    TEST_ASSERT(approx(mgr.getInvMass(h), 0.25f));
    TEST_ASSERT(mgr.getBodyType(h) == BodyType::Dynamic);
    return true;
}

static bool test_fixed_rotation_zero_inertia() {
    BodyManager mgr(64);
    BodyDef def = makeDynamicDef(Vec3::zero(), 1.0f);
    def.fixedRotation = true;
    BodyHandle h = mgr.createBody(def);

    uint32_t idx = mgr.getIndex(h);
    const Mat3& invI = mgr.store().localInvInertia(idx);
    // All diagonal entries should be zero for fixed rotation.
    TEST_ASSERT(approx(invI.get(0, 0), 0.0f));
    TEST_ASSERT(approx(invI.get(1, 1), 0.0f));
    TEST_ASSERT(approx(invI.get(2, 2), 0.0f));
    return true;
}

static bool test_body_stats() {
    BodyManager mgr(64);
    mgr.createBody(makeDynamicDef(Vec3(0, 0, 0)));
    mgr.createBody(makeDynamicDef(Vec3(1, 0, 0)));
    mgr.createBody(makeStaticDef(Vec3(0, -1, 0)));
    mgr.createBody(makeKinematicDef(Vec3(0, 5, 0)));

    BodyStats stats = mgr.getStats();
    TEST_ASSERT(stats.totalBodies == 4);
    TEST_ASSERT(stats.activeBodies == 2);
    TEST_ASSERT(stats.staticBodies == 1);
    TEST_ASSERT(stats.kinematicBodies == 1);
    TEST_ASSERT(stats.sleepingBodies == 0);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 5: World Inertia Update
// ═════════════════════════════════════════════════════════════════════════════

static bool test_world_inertia_identity_rotation() {
    BodyManager mgr(64);
    BodyDef def = makeDynamicDef(Vec3::zero(), 1.0f);
    BodyHandle h = mgr.createBody(def);
    uint32_t idx = mgr.getIndex(h);

    mgr.updateAllWorldInertias();

    // With identity rotation, world inertia == local inertia.
    const Mat3& local = mgr.store().localInvInertia(idx);
    const Mat3& world = mgr.store().worldInvInertia(idx);

    TEST_ASSERT(approx(local.get(0, 0), world.get(0, 0)));
    TEST_ASSERT(approx(local.get(1, 1), world.get(1, 1)));
    TEST_ASSERT(approx(local.get(2, 2), world.get(2, 2)));
    return true;
}

static bool test_world_inertia_rotated() {
    BodyManager mgr(64);
    BodyDef def;
    def.type = BodyType::Dynamic;
    def.initialTransform = Transform(Vec3::zero(),
        Quat::fromAxisAngle(Vec3::unitZ(), math::Pi * 0.5f)); // 90° around Z
    def.mass = 1.0f;
    // Non-uniform inertia to make rotation visible.
    def.localInertia = Mat3(1.0f, 0, 0,
                            0, 2.0f, 0,
                            0, 0, 3.0f);
    BodyHandle h = mgr.createBody(def);
    uint32_t idx = mgr.getIndex(h);

    mgr.updateAllWorldInertias();

    const Mat3& world = mgr.store().worldInvInertia(idx);
    // After 90° rotation around Z, X↔Y should swap.
    // Local inverse diag: (1, 0.5, 1/3). After rotation: (~0.5, ~1.0, ~1/3)
    TEST_ASSERT(std::fabs(world.get(0, 0) - 0.5f) < 0.1f);
    TEST_ASSERT(std::fabs(world.get(1, 1) - 1.0f) < 0.1f);
    return true;
}

static bool test_world_inertia_static_unchanged() {
    BodyManager mgr(64);
    BodyHandle h = mgr.createBody(makeStaticDef(Vec3::zero()));
    uint32_t idx = mgr.getIndex(h);

    mgr.updateAllWorldInertias();

    // Static body should have zero inertia.
    const Mat3& world = mgr.store().worldInvInertia(idx);
    TEST_ASSERT(approx(world.get(0, 0), 0.0f));
    TEST_ASSERT(approx(world.get(1, 1), 0.0f));
    TEST_ASSERT(approx(world.get(2, 2), 0.0f));
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 6: SolverBody Conversion
// ═════════════════════════════════════════════════════════════════════════════

static bool test_to_solver_body() {
    BodyManager mgr(64);
    BodyDef def = makeDynamicDef(Vec3(1, 2, 3), 2.0f);
    def.linearVelocity = Vec3(4, 5, 6);
    def.restitution = 0.5f;
    def.friction = 0.6f;
    BodyHandle h = mgr.createBody(def);
    mgr.updateAllWorldInertias();

    SolverBody sb = mgr.toSolverBody(h);
    TEST_ASSERT(approxVec(sb.position, Vec3(1, 2, 3)));
    TEST_ASSERT(approxVec(sb.linearVelocity, Vec3(4, 5, 6)));
    TEST_ASSERT(approx(sb.invMass, 0.5f));
    TEST_ASSERT(approx(sb.restitution, 0.5f));
    TEST_ASSERT(approx(sb.friction, 0.6f));
    TEST_ASSERT(sb.invInertia.getX() > 0.0f); // Should have non-zero inertia
    return true;
}

static bool test_from_solver_body() {
    BodyManager mgr(64);
    BodyHandle h = mgr.createBody(makeDynamicDef(Vec3(0, 0, 0)));

    SolverBody sb = mgr.toSolverBody(h);
    sb.position = Vec3(10, 20, 30);
    sb.linearVelocity = Vec3(1, 2, 3);
    sb.angularVelocity = Vec3(4, 5, 6);

    mgr.fromSolverBody(h, sb);
    TEST_ASSERT(approxVec(mgr.getPosition(h), Vec3(10, 20, 30)));
    TEST_ASSERT(approxVec(mgr.getLinearVelocity(h), Vec3(1, 2, 3)));
    TEST_ASSERT(approxVec(mgr.getAngularVelocity(h), Vec3(4, 5, 6)));
    return true;
}

static bool test_solver_body_static() {
    BodyManager mgr(64);
    BodyHandle h = mgr.createBody(makeStaticDef(Vec3(5, 0, 0)));
    mgr.updateAllWorldInertias();

    SolverBody sb = mgr.toSolverBody(h);
    TEST_ASSERT(approx(sb.invMass, 0.0f));
    TEST_ASSERT(sb.isStatic());
    return true;
}

static bool test_solver_body_by_index() {
    BodyManager mgr(64);
    BodyHandle h = mgr.createBody(makeDynamicDef(Vec3(7, 8, 9)));
    uint32_t idx = mgr.getIndex(h);

    SolverBody sb = mgr.toSolverBodyByIndex(idx);
    TEST_ASSERT(approxVec(sb.position, Vec3(7, 8, 9)));
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 7: Island Detection
// ═════════════════════════════════════════════════════════════════════════════

static bool test_island_single_body() {
    IslandManager islands(16);
    islands.reset(1);
    islands.buildIslands();
    TEST_ASSERT(islands.getIslandCount() == 1);
    TEST_ASSERT(islands.getIsland(0).bodyCount == 1);
    return true;
}

static bool test_island_two_connected() {
    IslandManager islands(16);
    islands.reset(2);
    islands.unite(0, 1);
    islands.buildIslands();
    TEST_ASSERT(islands.getIslandCount() == 1);
    TEST_ASSERT(islands.getIsland(0).bodyCount == 2);
    return true;
}

static bool test_island_two_separate() {
    IslandManager islands(16);
    islands.reset(2);
    // No unite — two separate islands.
    islands.buildIslands();
    TEST_ASSERT(islands.getIslandCount() == 2);
    return true;
}

static bool test_island_chain() {
    IslandManager islands(16);
    islands.reset(5);
    islands.unite(0, 1);
    islands.unite(1, 2);
    islands.unite(3, 4);
    // Island 1: {0,1,2}  Island 2: {3,4}
    islands.buildIslands();
    TEST_ASSERT(islands.getIslandCount() == 2);

    // Find which island has 3 bodies and which has 2.
    bool found3 = false, found2 = false;
    for (uint32_t i = 0; i < islands.getIslandCount(); ++i) {
        if (islands.getIsland(i).bodyCount == 3) found3 = true;
        if (islands.getIsland(i).bodyCount == 2) found2 = true;
    }
    TEST_ASSERT(found3);
    TEST_ASSERT(found2);
    return true;
}

static bool test_island_transitive() {
    IslandManager islands(16);
    islands.reset(4);
    islands.unite(0, 1);
    islands.unite(2, 3);
    islands.unite(1, 2); // Merges all into one island.

    TEST_ASSERT(islands.connected(0, 3));
    islands.buildIslands();
    TEST_ASSERT(islands.getIslandCount() == 1);
    TEST_ASSERT(islands.getIsland(0).bodyCount == 4);
    return true;
}

static bool test_island_skip_static() {
    IslandManager islands(16);
    islands.reset(3);
    islands.unite(0, 1);
    islands.unite(1, 2);

    // Mark body 0 as static — should be excluded from islands.
    bool isStatic[3] = {true, false, false};
    islands.buildIslands(isStatic);

    // Bodies 1 and 2 are connected. Body 0 is excluded.
    TEST_ASSERT(islands.getIslandCount() == 1);
    TEST_ASSERT(islands.getIsland(0).bodyCount == 2);
    return true;
}

static bool test_island_many_bodies() {
    IslandManager islands(1024);
    islands.reset(100);

    // Connect into 10 chains of 10.
    for (uint32_t chain = 0; chain < 10; ++chain) {
        for (uint32_t i = 1; i < 10; ++i) {
            islands.unite(chain * 10, chain * 10 + i);
        }
    }

    islands.buildIslands();
    TEST_ASSERT(islands.getIslandCount() == 10);
    for (uint32_t i = 0; i < islands.getIslandCount(); ++i) {
        TEST_ASSERT(islands.getIsland(i).bodyCount == 10);
    }
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 8: Sleep Management
// ═════════════════════════════════════════════════════════════════════════════

static bool test_sleep_awake_by_default() {
    BodyManager mgr(64);
    BodyHandle h = mgr.createBody(makeDynamicDef(Vec3::zero()));
    TEST_ASSERT(!mgr.isSleeping(h));
    return true;
}

static bool test_sleep_after_timeout() {
    RigidBodyStore store(64);
    BodyDef def = makeDynamicDef(Vec3::zero());
    def.linearVelocity = Vec3::zero();
    def.angularVelocity = Vec3::zero();
    std::size_t idx = store.add(def);

    SleepConfig cfg;
    cfg.timeToSleep = 0.5f;
    SleepManager sleepMgr(cfg);

    // Simulate 1 second at 60fps — should sleep after 0.5s.
    for (int frame = 0; frame < 60; ++frame) {
        sleepMgr.updateSleep(store, 1.0f / 60.0f);
    }

    TEST_ASSERT(store.isSleeping(idx));
    return true;
}

static bool test_sleep_prevented_by_velocity() {
    RigidBodyStore store(64);
    BodyDef def = makeDynamicDef(Vec3::zero());
    def.linearVelocity = Vec3(5, 0, 0); // Well above threshold
    std::size_t idx = store.add(def);

    SleepManager sleepMgr;
    for (int frame = 0; frame < 120; ++frame) {
        sleepMgr.updateSleep(store, 1.0f / 60.0f);
    }

    TEST_ASSERT(!store.isSleeping(idx));
    return true;
}

static bool test_wake_body() {
    RigidBodyStore store(64);
    BodyDef def = makeDynamicDef(Vec3::zero());
    std::size_t idx = store.add(def);

    SleepManager sleepMgr;
    sleepMgr.sleepBody(store, idx);
    TEST_ASSERT(store.isSleeping(idx));

    sleepMgr.wakeBody(store, idx);
    TEST_ASSERT(!store.isSleeping(idx));
    TEST_ASSERT(approx(store.sleepTimer(idx), 0.0f));
    return true;
}

static bool test_sleep_zeros_velocity() {
    RigidBodyStore store(64);
    BodyDef def = makeDynamicDef(Vec3::zero());
    def.linearVelocity = Vec3(0.01f, 0, 0); // Below threshold
    def.angularVelocity = Vec3(0, 0.01f, 0);
    std::size_t idx = store.add(def);

    SleepConfig cfg;
    cfg.timeToSleep = 0.1f; // Short sleep time
    SleepManager sleepMgr(cfg);

    for (int frame = 0; frame < 60; ++frame) {
        sleepMgr.updateSleep(store, 1.0f / 60.0f);
    }

    TEST_ASSERT(store.isSleeping(idx));
    // Velocity should be zeroed when asleep.
    TEST_ASSERT(approxVec(store.linearVelocity(idx), Vec3::zero()));
    TEST_ASSERT(approxVec(store.angularVelocity(idx), Vec3::zero()));
    return true;
}

static bool test_island_sleep_check() {
    RigidBodyStore store(64);
    BodyDef def = makeDynamicDef(Vec3::zero());
    std::size_t i0 = store.add(def);
    std::size_t i1 = store.add(def);

    SleepConfig cfg;
    cfg.timeToSleep = 0.1f;
    SleepManager sleepMgr(cfg);

    // Let both bodies settle to sleep.
    for (int frame = 0; frame < 60; ++frame) {
        sleepMgr.updateSleep(store, 1.0f / 60.0f);
    }

    uint32_t indices[] = {static_cast<uint32_t>(i0), static_cast<uint32_t>(i1)};
    TEST_ASSERT(sleepMgr.canIslandSleep(store, indices, 2));
    return true;
}

static bool test_island_wake() {
    RigidBodyStore store(64);
    BodyDef def = makeDynamicDef(Vec3::zero());
    std::size_t i0 = store.add(def);
    std::size_t i1 = store.add(def);

    SleepManager sleepMgr;
    uint32_t indices[] = {static_cast<uint32_t>(i0), static_cast<uint32_t>(i1)};

    sleepMgr.sleepIsland(store, indices, 2);
    TEST_ASSERT(store.isSleeping(i0));
    TEST_ASSERT(store.isSleeping(i1));

    sleepMgr.wakeIsland(store, indices, 2);
    TEST_ASSERT(!store.isSleeping(i0));
    TEST_ASSERT(!store.isSleeping(i1));
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 9: Edge Cases
// ═════════════════════════════════════════════════════════════════════════════

static bool test_rigid_body_store_empty() {
    RigidBodyStore store(64);
    TEST_ASSERT(store.size() == 0);
    TEST_ASSERT(store.empty());
    TEST_ASSERT(store.capacity() >= 64);
    return true;
}

static bool test_store_grow() {
    RigidBodyStore store(4);
    for (int i = 0; i < 100; ++i) {
        store.add(makeDynamicDef(Vec3(static_cast<float>(i), 0, 0)));
    }
    TEST_ASSERT(store.size() == 100);
    TEST_ASSERT(store.capacity() >= 100);

    // Verify first and last positions are correct.
    TEST_ASSERT(approxVec(store.position(0), Vec3(0, 0, 0)));
    TEST_ASSERT(approxVec(store.position(99), Vec3(99, 0, 0)));
    return true;
}

static bool test_store_remove_last() {
    RigidBodyStore store(16);
    store.add(makeDynamicDef(Vec3(0, 0, 0)));
    store.add(makeDynamicDef(Vec3(1, 0, 0)));
    store.add(makeDynamicDef(Vec3(2, 0, 0)));

    store.remove(2); // Remove last — no swap needed.
    TEST_ASSERT(store.size() == 2);
    TEST_ASSERT(approxVec(store.position(0), Vec3(0, 0, 0)));
    TEST_ASSERT(approxVec(store.position(1), Vec3(1, 0, 0)));
    return true;
}

static bool test_iteration_for_each() {
    BodyManager mgr(64);
    mgr.createBody(makeDynamicDef(Vec3(1, 0, 0)));
    mgr.createBody(makeStaticDef(Vec3(0, -1, 0)));
    mgr.createBody(makeDynamicDef(Vec3(2, 0, 0)));

    uint32_t total = 0;
    mgr.forEach([&total](uint32_t) { total++; });
    TEST_ASSERT(total == 3);

    uint32_t active = 0;
    mgr.forEachActive([&active](uint32_t) { active++; });
    TEST_ASSERT(active == 2);
    return true;
}

static bool test_clear_island_ids() {
    RigidBodyStore store(16);
    store.add(makeDynamicDef(Vec3::zero()));
    store.add(makeDynamicDef(Vec3(1, 0, 0)));
    store.setIslandId(0, 5);
    store.setIslandId(1, 7);
    TEST_ASSERT(store.islandId(0) == 5);
    TEST_ASSERT(store.islandId(1) == 7);

    store.clearIslandIds();
    TEST_ASSERT(store.islandId(0) == 0xFFFF);
    TEST_ASSERT(store.islandId(1) == 0xFFFF);
    return true;
}

static bool test_collision_layer_mask() {
    RigidBodyStore store(16);
    BodyDef def = makeDynamicDef(Vec3::zero());
    def.collisionLayer = 0x0004;
    def.collisionMask = 0x00FF;
    std::size_t idx = store.add(def);

    TEST_ASSERT(store.collisionLayer(idx) == 0x0004);
    TEST_ASSERT(store.collisionMask(idx) == 0x00FF);
    return true;
}

static bool test_sensor_flag() {
    RigidBodyStore store(16);
    BodyDef def = makeDynamicDef(Vec3::zero());
    def.isSensor = true;
    std::size_t idx = store.add(def);

    TEST_ASSERT(hasFlag(store.flags(idx), BodyFlags::Sensor));
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("=== Pulse RigidBody Module Tests ===\n\n");

    // Group 1: BodyDef & Common Types
    std::printf("--- BodyDef & Common Types ---\n");
    RUN_TEST(test_body_def_defaults);
    RUN_TEST(test_body_handle_null);
    RUN_TEST(test_body_flags_operations);
    RUN_TEST(test_body_type_enum);
    RUN_TEST(test_body_config_defaults);
    RUN_TEST(test_body_stats_default);

    // Group 2: BodyManager Creation & Destruction
    std::printf("--- BodyManager Creation & Destruction ---\n");
    RUN_TEST(test_create_single_body);
    RUN_TEST(test_create_multiple_bodies);
    RUN_TEST(test_destroy_body);
    RUN_TEST(test_destroy_middle_body);
    RUN_TEST(test_generation_invalidation);
    RUN_TEST(test_create_destroy_reuse);

    // Group 3: SoA Access
    std::printf("--- SoA Access ---\n");
    RUN_TEST(test_get_set_position);
    RUN_TEST(test_get_set_rotation);
    RUN_TEST(test_get_set_velocity);
    RUN_TEST(test_apply_force);
    RUN_TEST(test_apply_linear_impulse);
    RUN_TEST(test_apply_angular_impulse);
    RUN_TEST(test_apply_force_at_point);

    // Group 4: Body Type Behaviour
    std::printf("--- Body Type Behaviour ---\n");
    RUN_TEST(test_static_body_inv_mass);
    RUN_TEST(test_kinematic_body_inv_mass);
    RUN_TEST(test_dynamic_body_inv_mass);
    RUN_TEST(test_fixed_rotation_zero_inertia);
    RUN_TEST(test_body_stats);

    // Group 5: World Inertia Update
    std::printf("--- World Inertia Update ---\n");
    RUN_TEST(test_world_inertia_identity_rotation);
    RUN_TEST(test_world_inertia_rotated);
    RUN_TEST(test_world_inertia_static_unchanged);

    // Group 6: SolverBody Conversion
    std::printf("--- SolverBody Conversion ---\n");
    RUN_TEST(test_to_solver_body);
    RUN_TEST(test_from_solver_body);
    RUN_TEST(test_solver_body_static);
    RUN_TEST(test_solver_body_by_index);

    // Group 7: Island Detection
    std::printf("--- Island Detection ---\n");
    RUN_TEST(test_island_single_body);
    RUN_TEST(test_island_two_connected);
    RUN_TEST(test_island_two_separate);
    RUN_TEST(test_island_chain);
    RUN_TEST(test_island_transitive);
    RUN_TEST(test_island_skip_static);
    RUN_TEST(test_island_many_bodies);

    // Group 8: Sleep Management
    std::printf("--- Sleep Management ---\n");
    RUN_TEST(test_sleep_awake_by_default);
    RUN_TEST(test_sleep_after_timeout);
    RUN_TEST(test_sleep_prevented_by_velocity);
    RUN_TEST(test_wake_body);
    RUN_TEST(test_sleep_zeros_velocity);
    RUN_TEST(test_island_sleep_check);
    RUN_TEST(test_island_wake);

    // Group 9: Edge Cases
    std::printf("--- Edge Cases ---\n");
    RUN_TEST(test_rigid_body_store_empty);
    RUN_TEST(test_store_grow);
    RUN_TEST(test_store_remove_last);
    RUN_TEST(test_iteration_for_each);
    RUN_TEST(test_clear_island_ids);
    RUN_TEST(test_collision_layer_mask);
    RUN_TEST(test_sensor_flag);

    // Summary
    std::printf("\n=== Results: %d/%d passed", g_passed, g_total);
    if (g_failed > 0) std::printf(", %d FAILED", g_failed);
    std::printf(" ===\n");

    return (g_failed > 0) ? 1 : 0;
}
