/**
 * @file rigidbody_stub.cpp
 * @brief Compilation stub for the Pulse rigid body module (Module 11).
 *
 * Ensures all rigid body headers compile cleanly as part of the static library.
 */

#include <pulse/rigidbody/rigid_body_common.h>
#include <pulse/rigidbody/rigid_body.h>
#include <pulse/rigidbody/body_manager.h>
#include <pulse/rigidbody/island_manager.h>
#include <pulse/rigidbody/sleep_manager.h>

namespace pulse {
namespace {
    [[maybe_unused]] void rigidbody_compile_check() {
        // rigid_body_common.h
        BodyHandle handle;
        (void)handle.isNull();
        (void)handle.isValid();

        BodyDef def;
        def.type = BodyType::Dynamic;
        def.mass = 1.0f;
        def.restitution = 0.3f;
        def.friction = 0.4f;
        (void)def.enableCCD;
        (void)def.fixedRotation;

        BodyFlags flags = BodyFlags::Active | BodyFlags::EnableGravity;
        (void)hasFlag(flags, BodyFlags::Active);
        flags |= BodyFlags::Sleeping;
        flags &= ~BodyFlags::Sleeping;

        BodyConfig config;
        (void)config.maxBodies;
        (void)config.gravity;

        BodyStats stats;
        (void)stats.totalBodies;

        // rigid_body.h — RigidBodyStore
        RigidBodyStore store(64);
        std::size_t idx = store.add(def);
        (void)store.position(idx);
        (void)store.rotation(idx);
        (void)store.linearVelocity(idx);
        (void)store.angularVelocity(idx);
        (void)store.force(idx);
        (void)store.torque(idx);
        (void)store.invMass(idx);
        (void)store.localInvInertia(idx);
        (void)store.worldInvInertia(idx);
        (void)store.bodyType(idx);
        (void)store.flags(idx);
        (void)store.isDynamic(idx);
        (void)store.isStatic(idx);
        (void)store.isKinematic(idx);
        (void)store.isSleeping(idx);
        (void)store.isActive(idx);

        store.updateWorldInertia(idx);
        store.updateAllWorldInertias();
        store.clearForces();
        store.clearIslandIds();

        SolverBody sb = store.toSolverBody(idx);
        store.fromSolverBody(idx, sb);

        (void)store.size();
        (void)store.capacity();
        (void)store.empty();

        store.remove(idx);

        // body_manager.h — BodyManager
        BodyManager mgr(64);
        BodyHandle h1 = mgr.createBody(def);
        (void)mgr.isValid(h1);
        (void)mgr.getIndex(h1);
        (void)mgr.getPosition(h1);
        mgr.setPosition(h1, Vec3(1, 2, 3));
        (void)mgr.getRotation(h1);
        mgr.setRotation(h1, Quat::identity());
        (void)mgr.getLinearVelocity(h1);
        mgr.setLinearVelocity(h1, Vec3::zero());
        (void)mgr.getAngularVelocity(h1);
        mgr.setAngularVelocity(h1, Vec3::zero());
        mgr.applyForce(h1, Vec3(0, -9.81f, 0));
        mgr.applyTorque(h1, Vec3::zero());
        mgr.applyLinearImpulse(h1, Vec3(1, 0, 0));
        mgr.applyAngularImpulse(h1, Vec3(0, 1, 0));
        mgr.applyForceAtPoint(h1, Vec3(0, -9.81f, 0), Vec3(0, 1, 0));
        (void)mgr.getBodyType(h1);
        (void)mgr.isSleeping(h1);
        (void)mgr.getInvMass(h1);

        SolverBody sb2 = mgr.toSolverBody(h1);
        mgr.fromSolverBody(h1, sb2);

        mgr.clearForces();
        mgr.updateAllWorldInertias();
        mgr.clearIslandIds();

        (void)mgr.bodyCount();
        (void)mgr.capacity();

        BodyStats bstats = mgr.getStats();
        (void)bstats.totalBodies;

        mgr.forEach([](uint32_t) {});
        mgr.forEachActive([](uint32_t) {});

        mgr.destroyBody(h1);

        // island_manager.h — IslandManager
        IslandManager islands(64);
        islands.reset(10);
        islands.unite(0, 1);
        islands.unite(1, 2);
        (void)islands.find(0);
        (void)islands.connected(0, 2);
        islands.buildIslands();
        (void)islands.getIslandCount();
        if (islands.getIslandCount() > 0) {
            const IslandInfo& info = islands.getIsland(0);
            (void)info.bodyCount;
            (void)info.allSleeping;
        }

        // sleep_manager.h — SleepManager
        SleepConfig sleepCfg;
        (void)sleepCfg.linearSleepThreshold;
        (void)sleepCfg.angularSleepThreshold;
        (void)sleepCfg.timeToSleep;

        SleepManager sleepMgr;
        RigidBodyStore testStore(4);
        std::size_t ti = testStore.add(def);
        sleepMgr.updateSleep(testStore, 1.0f / 60.0f);
        sleepMgr.wakeBody(testStore, ti);
        (void)sleepMgr.isAwake(testStore, ti);
        sleepMgr.sleepBody(testStore, ti);

        uint32_t indices[] = {0};
        sleepMgr.wakeIsland(testStore, indices, 1);
        sleepMgr.sleepIsland(testStore, indices, 1);
        (void)sleepMgr.canIslandSleep(testStore, indices, 1);
    }
}
} // namespace pulse
