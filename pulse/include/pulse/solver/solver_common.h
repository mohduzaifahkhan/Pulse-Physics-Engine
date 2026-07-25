/**
 * @file solver_common.h
 * @brief Shared types and configuration for the constraint solver module.
 *
 * Defines the solver configuration (iteration counts, Baumgarte factor,
 * slop), the lightweight SolverBody view, and the pre-computed per-contact
 * constraint structures consumed by the velocity and position solvers.
 *
 * Design: SolverBody is a *temporary view* — the solver copies data in,
 * works on it, and copies results back.  The real body storage lives in
 * Module 11 (RigidBody).  For now, users build SolverBody manually.
 *
 * Inertia is represented as a diagonal Vec3 (invIxx, invIyy, invIzz).
 * This is exact for axis-aligned boxes/spheres/cylinders and a good
 * approximation otherwise.  Module 11 will upgrade to full Mat3.
 */

#pragma once

#include <pulse/math/math_common.h>
#include <pulse/math/vec3.h>

#include <cstdint>

namespace pulse {

// ── Solver configuration ─────────────────────────────────────────────────────

/**
 * @struct SolverConfig
 * @brief Parameters controlling the iterative constraint solver.
 */
struct SolverConfig {
    /// Number of sequential-impulse velocity iterations per step.
    uint32_t velocityIterations;

    /// Number of split-impulse position correction iterations per step.
    uint32_t positionIterations;

    /// Baumgarte stabilisation factor.
    /// Bias = (baumgarte / dt) * max(penetration - slop, 0).
    float baumgarte;

    /// Penetration allowance (metres) before position correction activates.
    float slop;

    /// Closing speed threshold (m/s) below which restitution is zeroed.
    float restitutionThreshold;

    /// Maximum position correction per iteration (metres).
    float maxPositionCorrection;

    /// Enable/disable warm starting from previous-frame impulses.
    bool warmStarting;

    /// Default configuration with production-tuned values.
    PULSE_FORCE_INLINE SolverConfig() noexcept
        : velocityIterations(8),
          positionIterations(3),
          baumgarte(0.2f),
          slop(0.005f),
          restitutionThreshold(1.0f),
          maxPositionCorrection(0.2f),
          warmStarting(true)
    {}

    /// Custom configuration.
    PULSE_FORCE_INLINE SolverConfig(uint32_t velIter, uint32_t posIter,
                                     float baum, float sl, float restThresh,
                                     float maxPosCorr, bool warmStart) noexcept
        : velocityIterations(velIter),
          positionIterations(posIter),
          baumgarte(baum),
          slop(sl),
          restitutionThreshold(restThresh),
          maxPositionCorrection(maxPosCorr),
          warmStarting(warmStart)
    {}
};

// ── Solver body ──────────────────────────────────────────────────────────────

/**
 * @struct SolverBody
 * @brief Lightweight per-body state consumed by the solver.
 *
 * This is a *temporary view* — not the authoritative body storage.
 * The solver reads velocities in, applies impulses, and the caller
 * reads updated velocities back out.
 *
 * Inertia: diagonal approximation (Vec3).  Sufficient for spheres,
 * boxes, and capsules aligned with principal axes.  Module 11 will
 * provide a full Mat3 inverse inertia tensor.
 */
struct SolverBody {
    Vec3  position;         ///< Centre of mass (world space).
    Vec3  linearVelocity;   ///< Linear velocity (m/s).
    Vec3  angularVelocity;  ///< Angular velocity (rad/s).

    float invMass;          ///< Inverse mass (0 = static/kinematic).
    Vec3  invInertia;       ///< Inverse inertia (diagonal: invIxx, invIyy, invIzz).

    float restitution;      ///< Coefficient of restitution [0, 1].
    float friction;         ///< Coefficient of friction [0, ∞).

    uint32_t bodyId;        ///< Identifier for lookup/matching.

    /// Default: static body at origin.
    PULSE_FORCE_INLINE SolverBody() noexcept
        : position(Vec3::zero()),
          linearVelocity(Vec3::zero()),
          angularVelocity(Vec3::zero()),
          invMass(0.0f),
          invInertia(Vec3::zero()),
          restitution(0.0f),
          friction(0.4f),
          bodyId(0xFFFFFFFFu)
    {}

    /// Convenience: construct a dynamic body.
    PULSE_FORCE_INLINE SolverBody(Vec3 pos, float mass, Vec3 inertia,
                                   float rest, float fric, uint32_t id) noexcept
        : position(pos),
          linearVelocity(Vec3::zero()),
          angularVelocity(Vec3::zero()),
          invMass(mass > math::Epsilon ? 1.0f / mass : 0.0f),
          invInertia(inertia.getX() > math::Epsilon ? Vec3(1.0f / inertia.getX(),
                     inertia.getY() > math::Epsilon ? 1.0f / inertia.getY() : 0.0f,
                     inertia.getZ() > math::Epsilon ? 1.0f / inertia.getZ() : 0.0f)
                   : Vec3::zero()),
          restitution(rest),
          friction(fric),
          bodyId(id)
    {}

    /// Is this body static (infinite mass)?
    [[nodiscard]] PULSE_FORCE_INLINE bool isStatic() const noexcept {
        return invMass < math::Epsilon;
    }

    /// Apply a linear impulse.
    PULSE_FORCE_INLINE void applyLinearImpulse(Vec3 impulse) noexcept {
        linearVelocity += impulse * invMass;
    }

    /// Apply an angular impulse.
    /// Uses diagonal inverse inertia: ω += invI ⊙ (r × impulse)
    PULSE_FORCE_INLINE void applyAngularImpulse(Vec3 torqueImpulse) noexcept {
        angularVelocity += invInertia * torqueImpulse;
    }
};

// ── Per-contact velocity constraint ──────────────────────────────────────────

/**
 * @struct VelocityConstraint
 * @brief Pre-computed data for one contact point in the velocity solver.
 *
 * Contains the contact basis, lever arms, effective masses, and
 * accumulated impulses for sequential-impulse iteration.
 */
struct VelocityConstraint {
    // ── Body references ──────────────────────────────────────────────────
    uint32_t indexA;    ///< Index into SolverBody array for body A.
    uint32_t indexB;    ///< Index into SolverBody array for body B.

    // ── Contact geometry ─────────────────────────────────────────────────
    Vec3 rA;            ///< Contact point offset from body A centre.
    Vec3 rB;            ///< Contact point offset from body B centre.
    Vec3 normal;        ///< Contact normal (from B toward A).
    Vec3 tangent0;      ///< First friction tangent.
    Vec3 tangent1;      ///< Second friction tangent.

    // ── Effective masses ─────────────────────────────────────────────────
    float normalMass;   ///< 1 / K_normal.
    float tangentMass0; ///< 1 / K_tangent0.
    float tangentMass1; ///< 1 / K_tangent1.

    // ── Bias & material ──────────────────────────────────────────────────
    float velocityBias; ///< Restitution bias = -e * v_closing (if above threshold).
    float friction;     ///< Combined friction coefficient for this pair.

    // ── Accumulated impulses ─────────────────────────────────────────────
    float normalImpulse;   ///< Accumulated normal impulse (≥ 0).
    float tangentImpulse0; ///< Accumulated friction impulse (tangent 0).
    float tangentImpulse1; ///< Accumulated friction impulse (tangent 1).

    /// Default: zero everything.
    PULSE_FORCE_INLINE VelocityConstraint() noexcept
        : indexA(0), indexB(0),
          rA(Vec3::zero()), rB(Vec3::zero()),
          normal(Vec3::zero()), tangent0(Vec3::zero()), tangent1(Vec3::zero()),
          normalMass(0.0f), tangentMass0(0.0f), tangentMass1(0.0f),
          velocityBias(0.0f), friction(0.0f),
          normalImpulse(0.0f), tangentImpulse0(0.0f), tangentImpulse1(0.0f)
    {}
};

// ── Per-contact position constraint ──────────────────────────────────────────

/**
 * @struct PositionConstraint
 * @brief Pre-computed data for one contact point in the position solver.
 *
 * Used by the split-impulse position correction pass to resolve
 * remaining penetration without affecting velocity.
 */
struct PositionConstraint {
    uint32_t indexA;    ///< Index into SolverBody array for body A.
    uint32_t indexB;    ///< Index into SolverBody array for body B.

    Vec3  rA;           ///< Contact offset from body A centre (local).
    Vec3  rB;           ///< Contact offset from body B centre (local).
    Vec3  normal;       ///< Contact normal (from B toward A).
    float penetration;  ///< Penetration depth (positive = overlap).
    float normalMass;   ///< 1 / K_normal (for position correction).

    /// Default: zero everything.
    PULSE_FORCE_INLINE PositionConstraint() noexcept
        : indexA(0), indexB(0),
          rA(Vec3::zero()), rB(Vec3::zero()),
          normal(Vec3::zero()),
          penetration(0.0f), normalMass(0.0f)
    {}
};

// ── Solver statistics ────────────────────────────────────────────────────────

/**
 * @struct SolverStats
 * @brief Diagnostic output from a solver pass.
 */
struct SolverStats {
    uint32_t velocityIterationsUsed; ///< Velocity iterations actually run.
    uint32_t positionIterationsUsed; ///< Position iterations actually run.
    uint32_t totalContacts;          ///< Total contact constraints solved.
    uint32_t totalManifolds;         ///< Total manifolds processed.
    float    maxPenetrationBefore;   ///< Worst penetration before solving.
    float    maxPenetrationAfter;    ///< Worst penetration after solving.
    bool     positionSolved;         ///< True if position solver converged.

    PULSE_FORCE_INLINE SolverStats() noexcept
        : velocityIterationsUsed(0),
          positionIterationsUsed(0),
          totalContacts(0),
          totalManifolds(0),
          maxPenetrationBefore(0.0f),
          maxPenetrationAfter(0.0f),
          positionSolved(false)
    {}
};

} // namespace pulse
