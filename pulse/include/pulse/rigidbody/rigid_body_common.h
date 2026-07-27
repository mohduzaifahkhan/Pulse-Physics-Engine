/**
 * @file rigid_body_common.h
 * @brief Shared types, enums, and configuration for the rigid body module (Module 11).
 *
 * Defines the fundamental building blocks used by BodyManager, IslandManager,
 * and SleepManager:
 *  - BodyTag / BodyHandle — type-safe generational handle for body references.
 *  - BodyType — Static / Dynamic / Kinematic classification.
 *  - BodyFlags — per-body status bitmask.
 *  - BodyDef — user-facing body creation descriptor.
 *  - BodyConfig — global configuration parameters.
 *  - BodyStats — diagnostic counters.
 *
 * Design: No virtual dispatch. Body type is a tag enum used for static dispatch.
 * All body state lives in SoA arrays managed by BodyManager.
 */

#pragma once

#include <pulse/math/math_common.h>
#include <pulse/math/vec3.h>
#include <pulse/math/quat.h>
#include <pulse/math/mat3.h>
#include <pulse/math/transform.h>
#include <pulse/shapes/shape_common.h>
#include <pulse/utilities/handle.h>

#include <cstdint>

namespace pulse {

// ── Handle types ─────────────────────────────────────────────────────────────

/// Tag type for rigid body handles.
struct BodyTag {};

/// A type-safe generational handle identifying a rigid body.
using BodyHandle = util::Handle<BodyTag>;

// ── Body type ────────────────────────────────────────────────────────────────

/**
 * @enum BodyType
 * @brief Classification of a rigid body's motion behaviour.
 */
enum class BodyType : uint8_t {
    /// Static bodies have infinite mass, never move, and are not affected by
    /// forces or collisions.  They participate in collision detection as
    /// immovable obstacles.
    Static = 0,

    /// Dynamic bodies are fully simulated — affected by gravity, forces,
    /// impulses, and collision response.
    Dynamic = 1,

    /// Kinematic bodies have infinite mass but can be moved programmatically
    /// via velocity.  They affect dynamic bodies but are not affected by them.
    Kinematic = 2
};

// ── Body flags ───────────────────────────────────────────────────────────────

/**
 * @enum BodyFlags
 * @brief Per-body status bitmask.
 */
enum class BodyFlags : uint16_t {
    None          = 0,
    Active        = 1 << 0,   ///< Body is active (not destroyed).
    Sleeping      = 1 << 1,   ///< Body is sleeping (excluded from solving/integration).
    InIsland      = 1 << 2,   ///< Body has been assigned to an island this frame.
    EnableCCD     = 1 << 3,   ///< Continuous collision detection enabled.
    EnableGravity = 1 << 4,   ///< Gravity is applied to this body.
    FixedRotation = 1 << 5,   ///< Angular velocity is always zero (infinite inertia).
    Bullet        = 1 << 6,   ///< High-velocity object — always uses CCD.
    Sensor        = 1 << 7,   ///< Trigger volume — detects overlaps but no physics response.
};

/// Bitwise OR for BodyFlags.
[[nodiscard]] PULSE_FORCE_INLINE BodyFlags operator|(BodyFlags a, BodyFlags b) noexcept {
    return static_cast<BodyFlags>(static_cast<uint16_t>(a) | static_cast<uint16_t>(b));
}

/// Bitwise AND for BodyFlags.
[[nodiscard]] PULSE_FORCE_INLINE BodyFlags operator&(BodyFlags a, BodyFlags b) noexcept {
    return static_cast<BodyFlags>(static_cast<uint16_t>(a) & static_cast<uint16_t>(b));
}

/// Bitwise NOT for BodyFlags.
[[nodiscard]] PULSE_FORCE_INLINE BodyFlags operator~(BodyFlags a) noexcept {
    return static_cast<BodyFlags>(~static_cast<uint16_t>(a));
}

/// In-place OR for BodyFlags.
PULSE_FORCE_INLINE BodyFlags& operator|=(BodyFlags& a, BodyFlags b) noexcept {
    a = a | b;
    return a;
}

/// In-place AND for BodyFlags.
PULSE_FORCE_INLINE BodyFlags& operator&=(BodyFlags& a, BodyFlags b) noexcept {
    a = a & b;
    return a;
}

/// Test if a flag is set.
[[nodiscard]] PULSE_FORCE_INLINE bool hasFlag(BodyFlags flags, BodyFlags test) noexcept {
    return (static_cast<uint16_t>(flags) & static_cast<uint16_t>(test)) != 0;
}

// ── Body definition ──────────────────────────────────────────────────────────

/**
 * @struct BodyDef
 * @brief User-facing descriptor for creating a rigid body.
 *
 * Fill in the fields and pass to BodyManager::createBody().
 */
struct BodyDef {
    BodyType  type;             ///< Static, Dynamic, or Kinematic.

    Transform initialTransform; ///< Initial world-space pose.
    Vec3      linearVelocity;   ///< Initial linear velocity (m/s).
    Vec3      angularVelocity;  ///< Initial angular velocity (rad/s).

    float     mass;             ///< Total mass (kg). Ignored for Static/Kinematic.
    Mat3      localInertia;     ///< Inertia tensor in local frame. Zero = compute from mass.
    ShapeType shapeType;        ///< Associated collision shape type.

    float     restitution;      ///< Coefficient of restitution [0, 1].
    float     friction;         ///< Coefficient of friction [0, ∞).
    float     linearDamping;    ///< Linear velocity damping [0, 1). 0 = no damping.
    float     angularDamping;   ///< Angular velocity damping [0, 1). 0 = no damping.
    float     gravityScale;     ///< Gravity multiplier (1.0 = normal, 0.0 = no gravity).

    uint16_t  collisionLayer;   ///< Collision layer bitmask (which layer this body is on).
    uint16_t  collisionMask;    ///< Collision mask (which layers this body collides with).

    bool      enableCCD;        ///< Enable continuous collision detection.
    bool      fixedRotation;    ///< Lock rotation (infinite angular inertia).
    bool      isBullet;         ///< Mark as bullet (always CCD for fast-moving).
    bool      isSensor;         ///< Trigger only — no physics response.
    bool      startAwake;       ///< Start active (not sleeping).

    /// Default: dynamic body at origin with 1kg mass.
    PULSE_FORCE_INLINE BodyDef() noexcept
        : type(BodyType::Dynamic),
          initialTransform(Transform::identity()),
          linearVelocity(Vec3::zero()),
          angularVelocity(Vec3::zero()),
          mass(1.0f),
          localInertia(Mat3()),  // Identity — will be overridden
          shapeType(ShapeType::Sphere),
          restitution(0.3f),
          friction(0.4f),
          linearDamping(0.01f),
          angularDamping(0.01f),
          gravityScale(1.0f),
          collisionLayer(0x0001),
          collisionMask(0xFFFF),
          enableCCD(false),
          fixedRotation(false),
          isBullet(false),
          isSensor(false),
          startAwake(true)
    {}
};

// ── Body configuration ───────────────────────────────────────────────────────

/**
 * @struct BodyConfig
 * @brief Global configuration for the rigid body system.
 */
struct BodyConfig {
    uint32_t maxBodies;            ///< Maximum number of bodies.
    Vec3     gravity;              ///< Global gravity vector (m/s²).
    float    defaultLinearDamping;  ///< Default linear damping for new bodies.
    float    defaultAngularDamping; ///< Default angular damping for new bodies.

    /// Default configuration.
    PULSE_FORCE_INLINE BodyConfig() noexcept
        : maxBodies(16384),
          gravity(Vec3(0.0f, -9.81f, 0.0f)),
          defaultLinearDamping(0.01f),
          defaultAngularDamping(0.01f)
    {}

    /// Custom configuration.
    PULSE_FORCE_INLINE BodyConfig(uint32_t max, Vec3 grav,
                                   float linDamp, float angDamp) noexcept
        : maxBodies(max),
          gravity(grav),
          defaultLinearDamping(linDamp),
          defaultAngularDamping(angDamp)
    {}
};

// ── Body statistics ──────────────────────────────────────────────────────────

/**
 * @struct BodyStats
 * @brief Diagnostic counters for the body system.
 */
struct BodyStats {
    uint32_t totalBodies;      ///< Total bodies currently allocated.
    uint32_t activeBodies;     ///< Bodies that are awake and dynamic.
    uint32_t sleepingBodies;   ///< Bodies that are sleeping.
    uint32_t staticBodies;     ///< Static bodies.
    uint32_t kinematicBodies;  ///< Kinematic bodies.
    uint32_t islandCount;      ///< Number of islands detected.

    PULSE_FORCE_INLINE BodyStats() noexcept
        : totalBodies(0), activeBodies(0), sleepingBodies(0),
          staticBodies(0), kinematicBodies(0), islandCount(0)
    {}
};

} // namespace pulse
