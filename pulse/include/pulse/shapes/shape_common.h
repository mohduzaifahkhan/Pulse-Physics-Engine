/**
 * @file shape_common.h
 * @brief Common shape types — ShapeType enum, MassProperties, support structures.
 *
 * Shared definitions used by all collision shape types in the engine.
 * No virtual dispatch — shapes use ShapeType tags for static dispatch
 * in the collision pipeline.
 */

#pragma once

#include <pulse/math/math_common.h>
#include <pulse/math/vec3.h>
#include <pulse/math/mat3.h>

namespace pulse {

// ── Shape type enumeration ───────────────────────────────────────────────────

/**
 * @enum ShapeType
 * @brief Tag identifying the collision shape type for dispatch.
 *
 * Used by the collision dispatcher to select the correct narrow-phase
 * algorithm without virtual function overhead.
 */
enum class ShapeType : uint8_t {
    Sphere     = 0,
    Box        = 1,
    Capsule    = 2,
    Cylinder   = 3,
    ConvexHull = 4,
    TriMesh    = 5,

    Count      = 6  ///< Sentinel — total number of shape types.
};

// ── Mass properties ──────────────────────────────────────────────────────────

/**
 * @struct MassProperties
 * @brief Mass, center of mass, and inertia tensor computed from a shape and density.
 *
 * All shapes produce this struct via their `computeMass(density)` method.
 * The inertia tensor is in the shape's local frame, about the center of mass.
 */
struct MassProperties {
    float mass;           ///< Total mass (kg).
    Vec3  centerOfMass;   ///< Center of mass in local space.
    Mat3  inertiaTensor;  ///< Inertia tensor about the center of mass, local frame.

    /// Default: zero mass, origin CoM, zero inertia.
    PULSE_FORCE_INLINE MassProperties() noexcept
        : mass(0.0f),
          centerOfMass(Vec3::zero()),
          inertiaTensor(Mat3(0.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 0.0f))
    {}

    /// Construct with explicit values.
    PULSE_FORCE_INLINE MassProperties(float m, Vec3 com, Mat3 inertia) noexcept
        : mass(m), centerOfMass(com), inertiaTensor(inertia)
    {}
};

// ── Ray hit result ───────────────────────────────────────────────────────────

/**
 * @struct ShapeRayResult
 * @brief Result of a ray intersection test against a shape (local space).
 */
struct ShapeRayResult {
    float t;        ///< Parametric distance along the ray. Negative if no hit.
    Vec3  normal;   ///< Surface normal at the hit point.
    bool  hit;      ///< True if the ray intersected the shape.

    /// Default: no hit.
    PULSE_FORCE_INLINE ShapeRayResult() noexcept
        : t(-1.0f), normal(Vec3::zero()), hit(false)
    {}

    /// Construct a hit result.
    PULSE_FORCE_INLINE ShapeRayResult(float t_, Vec3 n) noexcept
        : t(t_), normal(n), hit(true)
    {}

    /// Named factory for a miss.
    [[nodiscard]] static PULSE_FORCE_INLINE ShapeRayResult miss() noexcept {
        return ShapeRayResult();
    }
};

} // namespace pulse
