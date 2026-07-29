/**
 * @file world_common.h
 * @brief Shared types, configuration, and callback definitions for the World module (Module 13).
 *
 * Aggregates all sub-system configurations into a single WorldConfig, defines
 * per-frame diagnostic counters (WorldStats), and declares the contact event
 * callback system using raw function pointers (zero heap allocation).
 *
 * Design: WorldConfig is a POD-like aggregate — users fill in the fields
 * they care about and leave the rest at production-tuned defaults.
 */

#pragma once

#include <pulse/math/math_common.h>
#include <pulse/math/vec3.h>
#include <pulse/rigidbody/rigid_body_common.h>
#include <pulse/broadphase/broadphase_common.h>
#include <pulse/contact/contact_common.h>
#include <pulse/solver/solver_common.h>
#include <pulse/integration/integration_common.h>
#include <pulse/rigidbody/sleep_manager.h>

#include <cstdint>

namespace pulse {

// ── Broadphase algorithm selector ────────────────────────────────────────────

/**
 * @enum BroadPhaseType
 * @brief Selects the broadphase algorithm used by PhysicsWorld.
 */
enum class BroadPhaseType : uint8_t {
    DynamicAABBTree = 0,   ///< Default — balanced insert/query/update.
    SAP             = 1,   ///< Sweep-and-Prune — good for mostly-static scenes.
    UniformGrid     = 2    ///< Spatial hash grid — good for dense, uniform distributions.
};

// ── Contact event system ─────────────────────────────────────────────────────

/**
 * @enum ContactEventType
 * @brief Classification of contact lifecycle events.
 */
enum class ContactEventType : uint8_t {
    Begin   = 0,   ///< Two bodies just started touching this frame.
    Persist = 1,   ///< Contact persisted from the previous frame.
    End     = 2    ///< Two bodies separated this frame.
};

/**
 * @struct ContactEvent
 * @brief Data delivered to the user's contact callback.
 */
struct ContactEvent {
    BodyHandle      bodyA;          ///< First body in the contact.
    BodyHandle      bodyB;          ///< Second body in the contact.
    ContactEventType type;          ///< Begin, Persist, or End.
    Vec3            normal;         ///< Contact normal (from B toward A).
    Vec3            point;          ///< World-space contact point (average of manifold).
    float           penetration;    ///< Maximum penetration depth.
    uint32_t        contactCount;   ///< Number of contact points in the manifold.

    PULSE_FORCE_INLINE ContactEvent() noexcept
        : bodyA(), bodyB(),
          type(ContactEventType::Begin),
          normal(Vec3::zero()),
          point(Vec3::zero()),
          penetration(0.0f),
          contactCount(0)
    {}
};

/**
 * @typedef ContactCallback
 * @brief Raw function pointer for contact event notifications.
 *
 * Signature: void callback(const ContactEvent& event, void* userData)
 *
 * Raw function pointer (not std::function) to ensure zero heap allocation
 * in the hot path, consistent with the engine's no-alloc-in-inner-loop design.
 */
using ContactCallback = void (*)(const ContactEvent& event, void* userData);

// ── World configuration ──────────────────────────────────────────────────────

/**
 * @struct WorldConfig
 * @brief Master configuration aggregating all sub-system parameters.
 *
 * Users create a WorldConfig, tweak the fields they care about, and pass
 * it to PhysicsWorld's constructor.  All sub-configs default to their
 * own production-tuned values.
 */
struct WorldConfig {
    // ── Simulation parameters ────────────────────────────────────────────
    Vec3  gravity;              ///< World gravity (m/s²).
    float fixedTimeStep;        ///< Fixed simulation time step (seconds).
    uint32_t maxSubSteps;       ///< Maximum sub-steps per step() call.

    // ── Capacity hints ───────────────────────────────────────────────────
    uint32_t maxBodies;         ///< Maximum rigid bodies.
    uint32_t maxPairs;          ///< Maximum broadphase overlap pairs.

    // ── Algorithm selection ──────────────────────────────────────────────
    BroadPhaseType   broadPhaseType;    ///< Broadphase algorithm.
    IntegratorType   integratorType;    ///< Integration method.

    // ── Sub-system configurations ────────────────────────────────────────
    SolverConfig      solverConfig;
    SleepConfig       sleepConfig;
    ContactConfig     contactConfig;
    BroadPhaseConfig  broadPhaseConfig;

    // ── Speed limits ─────────────────────────────────────────────────────
    float maxLinearSpeed;       ///< Clamp linear speed (m/s).  0 = no clamp.
    float maxAngularSpeed;      ///< Clamp angular speed (rad/s). 0 = no clamp.

    /// Default: production-tuned values.
    PULSE_FORCE_INLINE WorldConfig() noexcept
        : gravity(Vec3(0.0f, -9.81f, 0.0f)),
          fixedTimeStep(1.0f / 60.0f),
          maxSubSteps(8),
          maxBodies(16384),
          maxPairs(65536),
          broadPhaseType(BroadPhaseType::DynamicAABBTree),
          integratorType(IntegratorType::SemiImplicitEuler),
          solverConfig(),
          sleepConfig(),
          contactConfig(),
          broadPhaseConfig(),
          maxLinearSpeed(500.0f),
          maxAngularSpeed(100.0f)
    {}
};

// ── World statistics ─────────────────────────────────────────────────────────

/**
 * @struct WorldStats
 * @brief Per-frame diagnostic counters from PhysicsWorld::step().
 */
struct WorldStats {
    // ── Body counts ──────────────────────────────────────────────────────
    uint32_t totalBodies;       ///< Total bodies in the world.
    uint32_t activeBodies;      ///< Awake dynamic bodies.
    uint32_t sleepingBodies;    ///< Sleeping bodies.
    uint32_t staticBodies;      ///< Static bodies.
    uint32_t kinematicBodies;   ///< Kinematic bodies.

    // ── Collision counts ─────────────────────────────────────────────────
    uint32_t broadPhasePairs;   ///< Overlap pairs from broadphase.
    uint32_t narrowPhaseContacts; ///< Contact manifolds generated.
    uint32_t islandCount;       ///< Number of simulation islands.

    // ── Solver ───────────────────────────────────────────────────────────
    uint32_t solverVelIters;    ///< Velocity iterations used.
    uint32_t solverPosIters;    ///< Position iterations used.
    float    maxPenetration;    ///< Worst penetration after solving.
    bool     positionSolved;    ///< Position solver converged.

    // ── Timing ───────────────────────────────────────────────────────────
    uint32_t subStepsTaken;     ///< Sub-steps taken in this step() call.

    PULSE_FORCE_INLINE WorldStats() noexcept
        : totalBodies(0), activeBodies(0), sleepingBodies(0),
          staticBodies(0), kinematicBodies(0),
          broadPhasePairs(0), narrowPhaseContacts(0), islandCount(0),
          solverVelIters(0), solverPosIters(0),
          maxPenetration(0.0f), positionSolved(false),
          subStepsTaken(0)
    {}
};

} // namespace pulse
