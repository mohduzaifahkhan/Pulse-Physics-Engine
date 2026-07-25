/**
 * @file constraint_base.h
 * @brief Extensibility hook for generic constraints (joints, motors, etc.).
 *
 * Defines ConstraintType and a lightweight ConstraintHeader that future
 * Module 10 (Constraints) will use to register custom constraint types
 * with the solver.  The solver hot-path works directly with
 * VelocityConstraint / PositionConstraint arrays — this header exists
 * purely for future extensibility and type tagging.
 *
 * Design: No virtual dispatch.  Constraints are identified by a type
 * enum and processed via template specialisation or switch-based dispatch.
 * This keeps the constraint data in contiguous arrays suitable for SIMD
 * batch processing.
 */

#pragma once

#include <pulse/math/math_common.h>
#include <cstdint>

namespace pulse {

// ── Constraint type enumeration ──────────────────────────────────────────────

/**
 * @enum ConstraintType
 * @brief Identifies the kind of constraint for dispatch.
 *
 * Contact is handled in this module (Module 9).  All other types
 * are defined here for future use by Module 10 (Constraints).
 */
enum class ConstraintType : uint8_t {
    Contact    = 0,   ///< Normal + friction contact constraint.
    Distance   = 1,   ///< Distance (rod/spring) constraint.
    Hinge      = 2,   ///< Revolute joint (one rotational DoF).
    Slider     = 3,   ///< Prismatic joint (one translational DoF).
    ConeTwist  = 4,   ///< Cone-twist (shoulder-like) joint.
    SixDof     = 5,   ///< Fully configurable 6-DoF joint.
    Spring     = 6,   ///< Hooke's law spring.
    Motor      = 7,   ///< Velocity/position motor drive.
    Count      = 8    ///< Sentinel — number of constraint types.
};

// ── Constraint header ────────────────────────────────────────────────────────

/**
 * @struct ConstraintHeader
 * @brief Lightweight header tagging a constraint for dispatch.
 *
 * Every constraint (contacts, joints, etc.) begins with this header.
 * The solver can sort/group constraints by type for cache-friendly
 * batch processing.
 */
struct ConstraintHeader {
    ConstraintType type;    ///< What kind of constraint this is.
    uint8_t        flags;   ///< Reserved for future flags (enabled, broken, etc.).
    uint16_t       _pad;    ///< Padding for alignment.
    uint32_t       bodyIdA; ///< Body A identifier.
    uint32_t       bodyIdB; ///< Body B identifier.

    /// Default: contact constraint with invalid body IDs.
    PULSE_FORCE_INLINE ConstraintHeader() noexcept
        : type(ConstraintType::Contact),
          flags(0),
          _pad(0),
          bodyIdA(0xFFFFFFFFu),
          bodyIdB(0xFFFFFFFFu)
    {}

    /// Construct with explicit type and body IDs.
    PULSE_FORCE_INLINE ConstraintHeader(ConstraintType t,
                                         uint32_t idA, uint32_t idB) noexcept
        : type(t), flags(0), _pad(0), bodyIdA(idA), bodyIdB(idB)
    {}

    /// Is this constraint enabled?
    [[nodiscard]] PULSE_FORCE_INLINE bool isEnabled() const noexcept {
        return (flags & 0x01) == 0; // Bit 0 = disabled flag (inverted).
    }

    /// Enable or disable this constraint.
    PULSE_FORCE_INLINE void setEnabled(bool enabled) noexcept {
        if (enabled) flags &= ~0x01u;
        else         flags |= 0x01u;
    }
};

// ── Contact constraint group ─────────────────────────────────────────────────

/**
 * @struct ContactConstraintGroup
 * @brief Groups all contacts from a single manifold for solver dispatch.
 *
 * The solver processes manifolds as groups — all contacts between a
 * single body pair are solved together for better convergence.
 */
struct ContactConstraintGroup {
    ConstraintHeader header;       ///< Type tag + body IDs.
    uint32_t         startIndex;   ///< Index into VelocityConstraint array.
    uint32_t         count;        ///< Number of contacts in this group.
    float            friction;     ///< Combined friction for this pair.
    float            restitution;  ///< Combined restitution for this pair.

    PULSE_FORCE_INLINE ContactConstraintGroup() noexcept
        : header(ConstraintType::Contact, 0xFFFFFFFFu, 0xFFFFFFFFu),
          startIndex(0), count(0),
          friction(0.0f), restitution(0.0f)
    {}
};

} // namespace pulse
