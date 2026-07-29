/**
 * @file world.h
 * @brief PhysicsWorld — the unified simulation pipeline (Module 13).
 *
 * Orchestrates all 12 Pulse subsystems into a single step() call:
 *
 *   1. Apply gravity to forces
 *   2. Integrate velocities (pre-solver)
 *   3. Update world inertias
 *   4. Update broadphase proxies
 *   5. Compute broadphase pairs
 *   6. Run narrowphase on each pair
 *   7. Feed manifolds into ContactCache (match/prune/warm-start)
 *   8. Build islands
 *   9. Solve constraints (per-island)
 *  10. Integrate positions (post-solver)
 *  11. Update sleep state
 *  12. Clear forces
 *  13. Fire contact callbacks
 *
 * Thread safety: NOT thread-safe. External synchronization required.
 * The step() call is designed to be called from a single thread; internal
 * island solving may be parallelised via JobSystem in a future update.
 */

#pragma once

#include "world_common.h"

#include <pulse/math/math_common.h>
#include <pulse/math/vec3.h>
#include <pulse/math/quat.h>
#include <pulse/math/mat3.h>
#include <pulse/math/transform.h>
#include <pulse/math/aabb.h>

#include <pulse/rigidbody/rigid_body_common.h>
#include <pulse/rigidbody/rigid_body.h>
#include <pulse/rigidbody/body_manager.h>
#include <pulse/rigidbody/island_manager.h>
#include <pulse/rigidbody/sleep_manager.h>

#include <pulse/broadphase/broadphase_common.h>
#include <pulse/broadphase/dynamic_aabb_tree.h>

#include <pulse/narrowphase/narrowphase_common.h>
#include <pulse/narrowphase/collision_dispatch.h>

#include <pulse/contact/contact_common.h>
#include <pulse/contact/contact_cache.h>
#include <pulse/contact/contact_manifold_persistent.h>

#include <pulse/solver/solver_common.h>
#include <pulse/solver/solver.h>

#include <pulse/integration/integration_common.h>
#include <pulse/integration/integrator.h>

#include <pulse/shapes/shape_common.h>
#include <pulse/shapes/sphere.h>
#include <pulse/shapes/box.h>
#include <pulse/shapes/capsule.h>
#include <pulse/shapes/cylinder.h>
#include <pulse/shapes/convex_hull.h>

#include <pulse/utilities/assert.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace pulse {

// ── Shape storage entry ──────────────────────────────────────────────────────

namespace world_detail {

/**
 * @struct ShapeEntry
 * @brief Stores a shape pointer + type tag for narrowphase dispatch.
 *
 * PhysicsWorld holds an array of these, one per body (indexed by dense index).
 * The shape pointer is non-owning — the user is responsible for shape lifetime.
 */
struct ShapeEntry {
    const void* shape;    ///< Pointer to shape data (Sphere, Box, etc.).
    ShapeType   type;     ///< Type tag for dispatch.

    ShapeEntry() noexcept : shape(nullptr), type(ShapeType::Sphere) {}
    ShapeEntry(const void* s, ShapeType t) noexcept : shape(s), type(t) {}
};

} // namespace world_detail

// ── PhysicsWorld ─────────────────────────────────────────────────────────────

/**
 * @class PhysicsWorld
 * @brief Unified physics simulation pipeline.
 *
 * Owns all subsystem instances and provides a single step(dt) entry point.
 * Bodies are referenced by stable BodyHandle values that survive swap-and-pop
 * removal in the underlying SoA store.
 */
class PhysicsWorld {
public:
    // ── Construction / Destruction ───────────────────────────────────────

    explicit PhysicsWorld(const WorldConfig& config = WorldConfig()) noexcept
        : config_(config),
          bodyManager_(config.maxBodies),
          broadphase_(config.broadPhaseConfig.maxProxies * 2),
          contactCache_(config.contactConfig.maxCachedPairs),
          islandManager_(config.maxBodies),
          sleepManager_(config.sleepConfig),
          contactCallback_(nullptr),
          callbackUserData_(nullptr),
          timeAccumulator_(0.0f)
    {
        // Build the integration config from world config.
        integrationConfig_.type = config.integratorType;
        integrationConfig_.gravity = config.gravity;
        integrationConfig_.maxLinearSpeed = config.maxLinearSpeed;
        integrationConfig_.maxAngularSpeed = config.maxAngularSpeed;

        // Allocate shape entry array.
        shapeCapacity_ = config.maxBodies;
        shapes_ = static_cast<world_detail::ShapeEntry*>(
            std::malloc(shapeCapacity_ * sizeof(world_detail::ShapeEntry)));
        PULSE_ASSERT_MSG(shapes_ != nullptr, "PhysicsWorld: shape array alloc failed");

        // Allocate proxy handle array (maps dense body index → broadphase proxy).
        proxyHandles_ = static_cast<ProxyHandle*>(
            std::malloc(shapeCapacity_ * sizeof(ProxyHandle)));
        PULSE_ASSERT_MSG(proxyHandles_ != nullptr, "PhysicsWorld: proxy array alloc failed");

        // Allocate overlap pair scratch buffer.
        maxPairs_ = config.maxPairs;
        overlapPairs_ = static_cast<OverlapPair*>(
            std::malloc(maxPairs_ * sizeof(OverlapPair)));
        PULSE_ASSERT_MSG(overlapPairs_ != nullptr, "PhysicsWorld: overlap pair alloc failed");

        // Allocate solver body scratch buffer.
        maxSolverBodies_ = config.maxBodies;
        solverBodies_ = static_cast<SolverBody*>(
            std::malloc(maxSolverBodies_ * sizeof(SolverBody)));
        PULSE_ASSERT_MSG(solverBodies_ != nullptr, "PhysicsWorld: solver body alloc failed");

        // Allocate persistent manifold scratch buffer.
        maxManifolds_ = config.maxPairs;
        manifolds_ = static_cast<PersistentManifold*>(
            std::malloc(maxManifolds_ * sizeof(PersistentManifold)));
        PULSE_ASSERT_MSG(manifolds_ != nullptr, "PhysicsWorld: manifold array alloc failed");

        // Initialize arrays.
        for (uint32_t i = 0; i < shapeCapacity_; ++i) {
            shapes_[i] = world_detail::ShapeEntry();
            proxyHandles_[i] = ProxyHandle::invalid();
        }
    }

    ~PhysicsWorld() noexcept {
        std::free(shapes_);
        std::free(proxyHandles_);
        std::free(overlapPairs_);
        std::free(solverBodies_);
        std::free(manifolds_);
    }

    // Non-copyable.
    PhysicsWorld(const PhysicsWorld&) = delete;
    PhysicsWorld& operator=(const PhysicsWorld&) = delete;

    // ── Body lifecycle ──────────────────────────────────────────────────

    /**
     * @brief Create a rigid body in the world.
     *
     * @param def    Body definition (position, mass, type, etc.).
     * @param shape  Pointer to the collision shape (non-owning).
     *               Must remain valid for the body's lifetime.
     * @return Stable BodyHandle for future reference.
     */
    [[nodiscard]] BodyHandle createBody(const BodyDef& def, const void* shape) noexcept {
        BodyHandle handle = bodyManager_.createBody(def);
        uint32_t denseIdx = bodyManager_.getIndex(handle);

        // Grow shape/proxy arrays if needed.
        if (denseIdx >= shapeCapacity_) {
            growShapeArrays(denseIdx + 1);
        }

        // Store shape reference.
        shapes_[denseIdx] = world_detail::ShapeEntry(shape, def.shapeType);

        // Compute initial AABB and insert into broadphase.
        AABB aabb = computeBodyAABB(denseIdx);
        bodyManager_.store().aabb(denseIdx) = aabb;

        AABB fatAABB = makeFatAABB(aabb, config_.broadPhaseConfig.fatMargin);
        ProxyHandle ph = broadphase_.createProxy(fatAABB, denseIdx);
        proxyHandles_[denseIdx] = ph;

        return handle;
    }

    /**
     * @brief Destroy a rigid body and remove it from the world.
     */
    void destroyBody(BodyHandle handle) noexcept {
        PULSE_ASSERT(bodyManager_.isValid(handle));
        uint32_t denseIdx = bodyManager_.getIndex(handle);
        uint32_t lastDense = static_cast<uint32_t>(bodyManager_.bodyCount()) - 1;

        // Remove broadphase proxy.
        ProxyHandle ph = proxyHandles_[denseIdx];
        if (ph.isValid()) {
            broadphase_.destroyProxy(ph);
        }

        // If not the last element, the last element will be swapped into denseIdx
        // by BodyManager::destroyBody(). We need to update shape/proxy mappings.
        if (denseIdx != lastDense) {
            shapes_[denseIdx] = shapes_[lastDense];
            proxyHandles_[denseIdx] = proxyHandles_[lastDense];
        }

        // Clear the last slot.
        shapes_[lastDense] = world_detail::ShapeEntry();
        proxyHandles_[lastDense] = ProxyHandle::invalid();

        // Destroy in body manager (does swap-and-pop internally).
        bodyManager_.destroyBody(handle);
    }

    // ── Simulation stepping ─────────────────────────────────────────────

    /**
     * @brief Advance the simulation by dt seconds using fixed-timestep accumulation.
     *
     * Internally accumulates time and runs singleStep(fixedDt) up to
     * maxSubSteps times per call.
     *
     * @param dt  Elapsed wall time since last call (seconds).
     * @return Per-frame diagnostics.
     */
    WorldStats step(float dt) noexcept {
        WorldStats stats;

        timeAccumulator_ += dt;
        float fixedDt = config_.fixedTimeStep;
        uint32_t steps = 0;

        while (timeAccumulator_ >= fixedDt && steps < config_.maxSubSteps) {
            singleStep(fixedDt);
            timeAccumulator_ -= fixedDt;
            steps++;
        }

        // Cap accumulator to prevent spiral of death.
        if (timeAccumulator_ > fixedDt * 2.0f) {
            timeAccumulator_ = 0.0f;
        }

        stats.subStepsTaken = steps;

        // Fill in body stats.
        BodyStats bs = bodyManager_.getStats();
        stats.totalBodies = bs.totalBodies;
        stats.activeBodies = bs.activeBodies;
        stats.sleepingBodies = bs.sleepingBodies;
        stats.staticBodies = bs.staticBodies;
        stats.kinematicBodies = bs.kinematicBodies;

        // Fill in last step's collision stats.
        stats.broadPhasePairs = lastBroadPhasePairs_;
        stats.narrowPhaseContacts = lastManifoldCount_;
        stats.islandCount = islandManager_.getIslandCount();
        stats.solverVelIters = lastSolverStats_.velocityIterationsUsed;
        stats.solverPosIters = lastSolverStats_.positionIterationsUsed;
        stats.maxPenetration = lastSolverStats_.maxPenetrationAfter;
        stats.positionSolved = lastSolverStats_.positionSolved;

        return stats;
    }

    /**
     * @brief Run one complete simulation tick at the given time step.
     *
     * This is the core pipeline — exposed publicly for users who want
     * manual control over sub-stepping.
     *
     * @param dt  Fixed time step (seconds).
     */
    void singleStep(float dt) noexcept {
        if (bodyManager_.bodyCount() == 0) return;

        RigidBodyStore& store = bodyManager_.store();
        const std::size_t bodyCount = store.size();

        // ── 1. Apply gravity to forces ──────────────────────────────────
        for (std::size_t i = 0; i < bodyCount; ++i) {
            if (!store.isActive(i)) continue;
            float invMass = store.invMass(i);
            if (invMass <= 0.0f) continue;
            float mass = 1.0f / invMass;
            float gs = store.gravityScale(i);
            store.force(i) += config_.gravity * mass * gs;
        }

        // ── 2. Integrate velocities (pre-solver) ────────────────────────
        integrateVelocities(store, integrationConfig_, dt);

        // ── 3. Update world-space inertia tensors ───────────────────────
        store.updateAllWorldInertias();

        // ── 4+5. Update broadphase proxies & compute pairs ──────────────
        updateBroadphase(store);
        uint32_t pairCount = broadphase_.computePairs(overlapPairs_, maxPairs_);
        lastBroadPhasePairs_ = pairCount;

        // ── 6+7. Narrowphase + ContactCache pipeline ────────────────────
        contactCache_.beginFrame();

        uint32_t manifoldCount = 0;
        for (uint32_t p = 0; p < pairCount; ++p) {
            uint32_t idxA = broadphase_.getUserData(overlapPairs_[p].a);
            uint32_t idxB = broadphase_.getUserData(overlapPairs_[p].b);

            // Validate indices.
            if (idxA >= bodyCount || idxB >= bodyCount) continue;

            // Skip if both bodies are sleeping.
            if (store.isSleeping(idxA) && store.isSleeping(idxB)) continue;

            // Skip static-static pairs.
            if (store.isStatic(idxA) && store.isStatic(idxB)) continue;

            // Collision layer/mask filter.
            uint16_t layerA = store.collisionLayer(idxA);
            uint16_t maskA = store.collisionMask(idxA);
            uint16_t layerB = store.collisionLayer(idxB);
            uint16_t maskB = store.collisionMask(idxB);
            if ((layerA & maskB) == 0 || (layerB & maskA) == 0) continue;

            // Run narrowphase.
            ContactManifold narrowManifold;
            bool hit = runNarrowphase(idxA, idxB, store, narrowManifold);

            if (hit && narrowManifold.numContacts > 0) {
                // Feed into contact cache for persistence + warm starting.
                contactCache_.processManifold(idxA, idxB, narrowManifold, config_.contactConfig);

                // Also build a solver-ready manifold.
                if (manifoldCount < maxManifolds_) {
                    PersistentManifold& pm = manifolds_[manifoldCount];
                    pm = PersistentManifold(idxA, idxB);

                    for (uint32_t c = 0; c < narrowManifold.numContacts && c < 4; ++c) {
                        pm.contacts[c].positionOnA = narrowManifold.points[c].positionOnA;
                        pm.contacts[c].positionOnB = narrowManifold.points[c].positionOnB;
                        pm.contacts[c].normal = narrowManifold.points[c].normal;
                        pm.contacts[c].penetration = narrowManifold.points[c].penetration;
                        pm.contacts[c].normalImpulse = 0.0f;
                        pm.contacts[c].tangentImpulse0 = 0.0f;
                        pm.contacts[c].tangentImpulse1 = 0.0f;
                    }
                    pm.contactCount = narrowManifold.numContacts;
                    if (pm.contactCount > 4) pm.contactCount = 4;
                    pm.refreshedThisFrame = true;

                    manifoldCount++;
                }
            }
        }

        // End-of-frame cleanup for contact cache.
        contactCache_.endFrame();

        // Warm-start impulses from the cache.
        contactCache_.warmStartAll(config_.contactConfig.warmStartFactor,
                                   config_.contactConfig.newContactDamping);

        lastManifoldCount_ = manifoldCount;

        // ── 8. Build islands ────────────────────────────────────────────
        islandManager_.reset(bodyCount);
        for (uint32_t m = 0; m < manifoldCount; ++m) {
            uint32_t a = manifolds_[m].bodyIdA;
            uint32_t b = manifolds_[m].bodyIdB;
            if (!store.isStatic(a) && !store.isStatic(b)) {
                islandManager_.unite(a, b);
            }
        }

        // Build static/kinematic mask for island building.
        bool* staticMask = static_cast<bool*>(std::malloc(bodyCount * sizeof(bool)));
        if (staticMask) {
            for (std::size_t i = 0; i < bodyCount; ++i) {
                staticMask[i] = store.isStatic(i) || store.isKinematic(i);
            }
            islandManager_.buildIslands(staticMask);
            std::free(staticMask);
        }

        // ── 9. Solve constraints ────────────────────────────────────────
        // Build solver bodies for ALL bodies.
        for (std::size_t i = 0; i < bodyCount && i < maxSolverBodies_; ++i) {
            solverBodies_[i] = store.toSolverBody(i);
        }

        // Run the solver over all manifolds.
        if (manifoldCount > 0) {
            lastSolverStats_ = solve(
                solverBodies_, static_cast<uint32_t>(bodyCount),
                manifolds_, manifoldCount,
                config_.solverConfig, dt);
        } else {
            lastSolverStats_ = SolverStats();
            lastSolverStats_.positionSolved = true;
        }

        // Write solver results back to body store.
        for (std::size_t i = 0; i < bodyCount; ++i) {
            if (!store.isStatic(i) && !store.isKinematic(i)) {
                store.fromSolverBody(i, solverBodies_[i]);
            }
        }

        // ── 10. Integrate positions (post-solver) ───────────────────────
        integratePositions(store, integrationConfig_, dt);

        // ── 11. Update sleep state ──────────────────────────────────────
        sleepManager_.updateSleep(store, dt);

        // Wake sleeping bodies that gained new contacts with awake bodies.
        for (uint32_t m = 0; m < manifoldCount; ++m) {
            uint32_t a = manifolds_[m].bodyIdA;
            uint32_t b = manifolds_[m].bodyIdB;
            if (store.isSleeping(a) && !store.isStatic(b) && !store.isSleeping(b)) {
                sleepManager_.wakeBody(store, a);
            }
            if (store.isSleeping(b) && !store.isStatic(a) && !store.isSleeping(a)) {
                sleepManager_.wakeBody(store, b);
            }
        }

        // ── 12. Clear accumulated forces ────────────────────────────────
        store.clearForces();

        // ── 13. Fire contact callbacks ──────────────────────────────────
        if (contactCallback_ != nullptr) {
            fireContactCallbacks(manifoldCount);
        }
    }

    // ── Body accessors (delegate to BodyManager) ────────────────────────

    [[nodiscard]] bool isValid(BodyHandle h) const noexcept { return bodyManager_.isValid(h); }

    [[nodiscard]] Vec3 getPosition(BodyHandle h) const noexcept { return bodyManager_.getPosition(h); }
    void setPosition(BodyHandle h, Vec3 pos) noexcept { bodyManager_.setPosition(h, pos); }

    [[nodiscard]] Quat getRotation(BodyHandle h) const noexcept { return bodyManager_.getRotation(h); }
    void setRotation(BodyHandle h, Quat rot) noexcept { bodyManager_.setRotation(h, rot); }

    [[nodiscard]] Vec3 getLinearVelocity(BodyHandle h) const noexcept { return bodyManager_.getLinearVelocity(h); }
    void setLinearVelocity(BodyHandle h, Vec3 vel) noexcept { bodyManager_.setLinearVelocity(h, vel); }

    [[nodiscard]] Vec3 getAngularVelocity(BodyHandle h) const noexcept { return bodyManager_.getAngularVelocity(h); }
    void setAngularVelocity(BodyHandle h, Vec3 vel) noexcept { bodyManager_.setAngularVelocity(h, vel); }

    void applyForce(BodyHandle h, Vec3 f) noexcept { bodyManager_.applyForce(h, f); }
    void applyTorque(BodyHandle h, Vec3 t) noexcept { bodyManager_.applyTorque(h, t); }
    void applyLinearImpulse(BodyHandle h, Vec3 impulse) noexcept { bodyManager_.applyLinearImpulse(h, impulse); }
    void applyAngularImpulse(BodyHandle h, Vec3 impulse) noexcept { bodyManager_.applyAngularImpulse(h, impulse); }
    void applyForceAtPoint(BodyHandle h, Vec3 f, Vec3 worldPoint) noexcept { bodyManager_.applyForceAtPoint(h, f, worldPoint); }

    [[nodiscard]] BodyType getBodyType(BodyHandle h) const noexcept { return bodyManager_.getBodyType(h); }
    [[nodiscard]] bool isSleeping(BodyHandle h) const noexcept { return bodyManager_.isSleeping(h); }
    [[nodiscard]] float getInvMass(BodyHandle h) const noexcept { return bodyManager_.getInvMass(h); }

    void wakeBody(BodyHandle h) noexcept {
        sleepManager_.wakeBody(bodyManager_.store(), bodyManager_.getIndex(h));
    }

    // ── World-level settings ────────────────────────────────────────────

    [[nodiscard]] Vec3 getGravity() const noexcept { return config_.gravity; }
    void setGravity(Vec3 g) noexcept {
        config_.gravity = g;
        integrationConfig_.gravity = g;
    }

    [[nodiscard]] const WorldConfig& getConfig() const noexcept { return config_; }

    // ── Callback registration ───────────────────────────────────────────

    /**
     * @brief Register a contact event callback.
     *
     * @param callback  Function pointer (or nullptr to disable).
     * @param userData  Opaque user data passed to every callback invocation.
     */
    void setContactCallback(ContactCallback callback, void* userData = nullptr) noexcept {
        contactCallback_ = callback;
        callbackUserData_ = userData;
    }

    // ── Queries ─────────────────────────────────────────────────────────

    [[nodiscard]] std::size_t bodyCount() const noexcept { return bodyManager_.bodyCount(); }

    [[nodiscard]] BodyStats getBodyStats() const noexcept { return bodyManager_.getStats(); }

    // ── Direct access (for advanced users / modules) ────────────────────

    [[nodiscard]] const BodyManager& bodyManager() const noexcept { return bodyManager_; }
    [[nodiscard]] BodyManager& bodyManager() noexcept { return bodyManager_; }
    [[nodiscard]] const RigidBodyStore& store() const noexcept { return bodyManager_.store(); }
    [[nodiscard]] RigidBodyStore& store() noexcept { return bodyManager_.store(); }

private:
    WorldConfig         config_;
    IntegrationConfig   integrationConfig_;

    BodyManager         bodyManager_;
    DynamicAABBTree     broadphase_;
    ContactCache        contactCache_;
    IslandManager       islandManager_;
    SleepManager        sleepManager_;

    // Shape storage (parallel to body dense array).
    world_detail::ShapeEntry* shapes_      = nullptr;
    uint32_t                  shapeCapacity_ = 0;

    // Broadphase proxy handles (parallel to body dense array).
    ProxyHandle*              proxyHandles_ = nullptr;

    // Scratch buffers (re-used each frame).
    OverlapPair*              overlapPairs_    = nullptr;
    uint32_t                  maxPairs_        = 0;
    PersistentManifold*       manifolds_       = nullptr;
    uint32_t                  maxManifolds_    = 0;
    SolverBody*               solverBodies_    = nullptr;
    uint32_t                  maxSolverBodies_ = 0;

    // Callback.
    ContactCallback           contactCallback_  = nullptr;
    void*                     callbackUserData_ = nullptr;

    // Fixed-timestep accumulator.
    float                     timeAccumulator_  = 0.0f;

    // Last-frame stats for reporting.
    uint32_t                  lastBroadPhasePairs_ = 0;
    uint32_t                  lastManifoldCount_   = 0;
    SolverStats               lastSolverStats_;

    // ── Internal helpers ─────────────────────────────────────────────────

    /// Grow shape and proxy arrays to accommodate at least minCapacity elements.
    void growShapeArrays(uint32_t minCapacity) noexcept {
        uint32_t newCap = shapeCapacity_;
        while (newCap < minCapacity) newCap *= 2;

        auto* newShapes = static_cast<world_detail::ShapeEntry*>(
            std::realloc(shapes_, newCap * sizeof(world_detail::ShapeEntry)));
        auto* newProxies = static_cast<ProxyHandle*>(
            std::realloc(proxyHandles_, newCap * sizeof(ProxyHandle)));
        PULSE_ASSERT_MSG(newShapes != nullptr, "PhysicsWorld: shape realloc failed");
        PULSE_ASSERT_MSG(newProxies != nullptr, "PhysicsWorld: proxy realloc failed");

        // Initialize new entries.
        for (uint32_t i = shapeCapacity_; i < newCap; ++i) {
            newShapes[i] = world_detail::ShapeEntry();
            newProxies[i] = ProxyHandle::invalid();
        }

        shapes_ = newShapes;
        proxyHandles_ = newProxies;
        shapeCapacity_ = newCap;
    }

    /// Compute AABB for a body from its shape and current transform.
    [[nodiscard]] AABB computeBodyAABB(uint32_t denseIdx) const noexcept {
        const RigidBodyStore& s = bodyManager_.store();
        const Transform& tx = s.transform(denseIdx);
        const world_detail::ShapeEntry& se = shapes_[denseIdx];

        if (!se.shape) {
            // Fallback: unit AABB around position.
            return AABB(tx.position - Vec3(0.5f), tx.position + Vec3(0.5f));
        }

        switch (se.type) {
        case ShapeType::Sphere: {
            const auto* sphere = static_cast<const Sphere*>(se.shape);
            return sphere->computeAABB(tx);
        }
        case ShapeType::Box: {
            const auto* box = static_cast<const Box*>(se.shape);
            return box->computeAABB(tx);
        }
        case ShapeType::Capsule: {
            const auto* cap = static_cast<const Capsule*>(se.shape);
            return cap->computeAABB(tx);
        }
        case ShapeType::Cylinder: {
            const auto* cyl = static_cast<const Cylinder*>(se.shape);
            return cyl->computeAABB(tx);
        }
        case ShapeType::ConvexHull: {
            const auto* hull = static_cast<const ConvexHull*>(se.shape);
            return hull->computeAABB(tx);
        }
        default: {
            return AABB(tx.position - Vec3(0.5f), tx.position + Vec3(0.5f));
        }
        }
    }

    /// Update all broadphase proxies from current body AABBs.
    void updateBroadphase(RigidBodyStore& store) noexcept {
        const std::size_t n = store.size();
        for (std::size_t i = 0; i < n; ++i) {
            ProxyHandle ph = proxyHandles_[i];
            if (!ph.isValid()) continue;

            // Skip sleeping bodies for move (they haven't changed position).
            if (store.isSleeping(i)) continue;

            AABB newAABB = computeBodyAABB(static_cast<uint32_t>(i));
            store.aabb(i) = newAABB;

            // Compute displacement for predictive fat AABB.
            Vec3 displacement = store.linearVelocity(i) * config_.fixedTimeStep;
            broadphase_.moveProxy(ph, newAABB, displacement,
                                  config_.broadPhaseConfig.fatMargin);
        }
    }

    /// Run narrowphase collision between two bodies.
    [[nodiscard]] bool runNarrowphase(uint32_t idxA, uint32_t idxB,
                                       const RigidBodyStore& store,
                                       ContactManifold& manifold) const noexcept
    {
        const world_detail::ShapeEntry& seA = shapes_[idxA];
        const world_detail::ShapeEntry& seB = shapes_[idxB];

        if (!seA.shape || !seB.shape) return false;

        const Transform& txA = store.transform(idxA);
        const Transform& txB = store.transform(idxB);

        // Dispatch based on shape type pair.
        // Order: ensure A's type <= B's type for canonical dispatch.
        uint8_t tA = static_cast<uint8_t>(seA.type);
        uint8_t tB = static_cast<uint8_t>(seB.type);

        if (tA <= tB) {
            return dispatchCollision(seA.shape, seA.type, txA,
                                     seB.shape, seB.type, txB, manifold);
        } else {
            // Swap order, then flip the manifold.
            bool hit = dispatchCollision(seB.shape, seB.type, txB,
                                          seA.shape, seA.type, txA, manifold);
            if (hit) {
                // Flip normals for swapped pair.
                for (uint32_t c = 0; c < manifold.numContacts; ++c) {
                    manifold.points[c].normal = manifold.points[c].normal * -1.0f;
                    Vec3 tmp = manifold.points[c].positionOnA;
                    manifold.points[c].positionOnA = manifold.points[c].positionOnB;
                    manifold.points[c].positionOnB = tmp;
                }
            }
            return hit;
        }
    }

    /// Dispatch collision based on shape types.
    [[nodiscard]] static bool dispatchCollision(
        const void* shapeA, ShapeType typeA, const Transform& txA,
        const void* shapeB, ShapeType typeB, const Transform& txB,
        ContactManifold& manifold) noexcept
    {
        // Sphere vs. *
        if (typeA == ShapeType::Sphere) {
            const auto& sA = *static_cast<const Sphere*>(shapeA);
            if (typeB == ShapeType::Sphere)
                return collide(sA, txA, *static_cast<const Sphere*>(shapeB), txB, manifold);
            if (typeB == ShapeType::Box)
                return collide(sA, txA, *static_cast<const Box*>(shapeB), txB, manifold);
            if (typeB == ShapeType::Capsule)
                return collide(sA, txA, *static_cast<const Capsule*>(shapeB), txB, manifold);
            if (typeB == ShapeType::Cylinder)
                return collide(sA, txA, *static_cast<const Cylinder*>(shapeB), txB, manifold);
            if (typeB == ShapeType::ConvexHull)
                return collide(sA, txA, *static_cast<const ConvexHull*>(shapeB), txB, manifold);
        }

        // Box vs. *
        if (typeA == ShapeType::Box) {
            const auto& bA = *static_cast<const Box*>(shapeA);
            if (typeB == ShapeType::Box)
                return collide(bA, txA, *static_cast<const Box*>(shapeB), txB, manifold);
            if (typeB == ShapeType::Capsule)
                return collide(bA, txA, *static_cast<const Capsule*>(shapeB), txB, manifold);
            if (typeB == ShapeType::Cylinder)
                return collide(bA, txA, *static_cast<const Cylinder*>(shapeB), txB, manifold);
            if (typeB == ShapeType::ConvexHull)
                return collide(bA, txA, *static_cast<const ConvexHull*>(shapeB), txB, manifold);
        }

        // Capsule vs. *
        if (typeA == ShapeType::Capsule) {
            const auto& cA = *static_cast<const Capsule*>(shapeA);
            if (typeB == ShapeType::Capsule)
                return collide(cA, txA, *static_cast<const Capsule*>(shapeB), txB, manifold);
            if (typeB == ShapeType::Cylinder)
                return collide(cA, txA, *static_cast<const Cylinder*>(shapeB), txB, manifold);
            if (typeB == ShapeType::ConvexHull)
                return collide(cA, txA, *static_cast<const ConvexHull*>(shapeB), txB, manifold);
        }

        // Cylinder vs. *
        if (typeA == ShapeType::Cylinder) {
            const auto& cylA = *static_cast<const Cylinder*>(shapeA);
            if (typeB == ShapeType::Cylinder)
                return collide(cylA, txA, *static_cast<const Cylinder*>(shapeB), txB, manifold);
            if (typeB == ShapeType::ConvexHull)
                return collide(cylA, txA, *static_cast<const ConvexHull*>(shapeB), txB, manifold);
        }

        // ConvexHull vs. ConvexHull
        if (typeA == ShapeType::ConvexHull && typeB == ShapeType::ConvexHull) {
            return collide(*static_cast<const ConvexHull*>(shapeA), txA,
                           *static_cast<const ConvexHull*>(shapeB), txB, manifold);
        }

        return false; // Unsupported combination (e.g., TriMesh).
    }

    /// Fire contact event callbacks for the current frame's manifolds.
    void fireContactCallbacks(uint32_t manifoldCount) noexcept {
        for (uint32_t m = 0; m < manifoldCount; ++m) {
            const PersistentManifold& pm = manifolds_[m];

            ContactEvent evt;
            evt.bodyA = BodyHandle(pm.bodyIdA, 0);
            evt.bodyB = BodyHandle(pm.bodyIdB, 0);
            evt.type = ContactEventType::Persist;
            evt.contactCount = pm.contactCount;

            if (pm.contactCount > 0) {
                evt.normal = pm.contacts[0].normal;
                evt.penetration = pm.contacts[0].penetration;

                // Average contact point.
                Vec3 avgPoint = Vec3::zero();
                for (uint32_t c = 0; c < pm.contactCount; ++c) {
                    avgPoint += (pm.contacts[c].positionOnA + pm.contacts[c].positionOnB) * 0.5f;
                }
                evt.point = avgPoint * (1.0f / static_cast<float>(pm.contactCount));

                // Find max penetration.
                for (uint32_t c = 1; c < pm.contactCount; ++c) {
                    if (pm.contacts[c].penetration > evt.penetration) {
                        evt.penetration = pm.contacts[c].penetration;
                    }
                }
            }

            contactCallback_(evt, callbackUserData_);
        }
    }
};

} // namespace pulse
