/**
 * @file distance_constraint.h
 * @brief Distance (rod/spring) constraint between two rigid bodies.
 *
 * Maintains a fixed distance L₀ between two anchor points, one on each
 * body.  Can operate as a rigid rod (hard constraint) or as a soft
 * spring-damper (via SoftConstraintParams).
 *
 * Constraint equation:
 *   C = |pB - pA| - L₀ = 0
 *
 * Jacobian (1 row):
 *   J = [-n̂, -(rA × n̂), n̂, (rB × n̂)]
 *   where n̂ = (pB - pA) / |pB - pA|
 *
 * When L₀ = 0, the constraint degenerates to a ball-socket joint
 * (point-to-point).  The direction is computed from any non-degenerate
 * separation; if both anchors coincide, the constraint is satisfied
 * and no impulse is applied.
 */

#pragma once

#include "constraint_common.h"
#include <pulse/solver/solver_common.h>
#include <pulse/solver/constraint_base.h>
#include <pulse/math/vec3.h>
#include <pulse/math/quat.h>
#include <pulse/math/math_common.h>

#include <cstdint>

namespace pulse {

/**
 * @class DistanceConstraint
 * @brief Constrains two anchor points to maintain a fixed distance.
 */
class DistanceConstraint {
public:
    // ── Configuration ────────────────────────────────────────────────────

    Vec3  localAnchorA;      ///< Anchor point in body A's local space.
    Vec3  localAnchorB;      ///< Anchor point in body B's local space.
    float restLength;        ///< Target distance L₀ (metres).

    ConstraintHeader header; ///< Type tag + body IDs + flags.

    SoftConstraintParams softParams;  ///< Optional spring-damper softness.

    // ── Construction ─────────────────────────────────────────────────────

    PULSE_FORCE_INLINE DistanceConstraint() noexcept
        : localAnchorA(Vec3::zero()),
          localAnchorB(Vec3::zero()),
          restLength(1.0f),
          header(ConstraintType::Distance, 0xFFFFFFFFu, 0xFFFFFFFFu),
          softParams()
    {}

    /**
     * @brief Construct a distance constraint.
     * @param anchorA  Anchor in body A's local space.
     * @param anchorB  Anchor in body B's local space.
     * @param length   Rest length (0 = ball-socket).
     * @param idA      Body A identifier.
     * @param idB      Body B identifier.
     */
    PULSE_FORCE_INLINE DistanceConstraint(Vec3 anchorA, Vec3 anchorB,
                                           float length,
                                           uint32_t idA, uint32_t idB) noexcept
        : localAnchorA(anchorA),
          localAnchorB(anchorB),
          restLength(length),
          header(ConstraintType::Distance, idA, idB),
          softParams()
    {}

    // ── Solver interface ─────────────────────────────────────────────────

    /**
     * @brief Pre-compute the Jacobian row for this constraint.
     *
     * Call once per frame before velocity iterations begin.
     *
     * @param bodies     Array of solver bodies.
     * @param bodyCount  Number of bodies.
     * @param config     Solver configuration (baumgarte factor).
     * @param dt         Time step (seconds).
     */
    void initialize(const SolverBody* bodies, uint32_t bodyCount,
                    const SolverConfig& config, float dt) noexcept
    {
        float invDt = (dt > math::Epsilon) ? 1.0f / dt : 0.0f;

        // Find body indices
        row_.indexA = findBodyIndex(bodies, bodyCount, header.bodyIdA);
        row_.indexB = findBodyIndex(bodies, bodyCount, header.bodyIdB);

        if (row_.indexA >= bodyCount || row_.indexB >= bodyCount) return;

        const SolverBody& bodyA = bodies[row_.indexA];
        const SolverBody& bodyB = bodies[row_.indexB];

        // Compute world-space anchor positions
        // Without quaternion rotation (Module 11 will provide body orientation),
        // we treat local anchors as offsets from the body centre.
        Vec3 rA = localAnchorA;
        Vec3 rB = localAnchorB;

        Vec3 pA = bodyA.position + rA;
        Vec3 pB = bodyB.position + rB;

        // Separation vector
        Vec3 delta = pB - pA;
        float currentLength = delta.length();

        // Direction (handle degenerate case)
        Vec3 n;
        if (currentLength > math::Epsilon) {
            n = delta / currentLength;
        } else {
            n = Vec3::unitY(); // Arbitrary fallback
            currentLength = 0.0f;
        }

        // Position error
        float error = currentLength - restLength;

        // Compute soft params if enabled
        if (softParams.enabled) {
            softParams.compute(dt);
        }

        // Setup the linear Jacobian row: J = [-n, -(rA×n), n, rB×n]
        setupLinearRow(row_, bodyA, bodyB, rA, rB, n, error,
                       config.baumgarte, invDt, softParams);
    }

    /**
     * @brief Warm-start by applying the cached impulse from last frame.
     */
    void warmStart(SolverBody* bodies) const noexcept {
        if (row_.indexA >= 0xFFFFFFFFu || row_.indexB >= 0xFFFFFFFFu) return;
        // Apply cached impulse along the constraint direction
        SolverBody& bodyA = bodies[row_.indexA];
        SolverBody& bodyB = bodies[row_.indexB];

        Vec3 impulse = row_.jvA * row_.accumulatedImpulse;
        bodyA.linearVelocity  += impulse * bodyA.invMass;
        bodyA.angularVelocity += bodyA.invInertia * (row_.jwA * row_.accumulatedImpulse);

        impulse = row_.jvB * row_.accumulatedImpulse;
        bodyB.linearVelocity  += impulse * bodyB.invMass;
        bodyB.angularVelocity += bodyB.invInertia * (row_.jwB * row_.accumulatedImpulse);
    }

    /**
     * @brief Run one velocity iteration for this constraint.
     */
    void solveVelocity(SolverBody* bodies) noexcept {
        if (row_.indexA >= 0xFFFFFFFFu || row_.indexB >= 0xFFFFFFFFu) return;
        solveJointRow(row_, bodies[row_.indexA], bodies[row_.indexB]);
    }

    /// Access the internal row for testing/diagnostics.
    [[nodiscard]] const JointRow& getRow() const noexcept { return row_; }

    /// Set the cached impulse for warm starting next frame.
    void setWarmStartImpulse(float impulse) noexcept {
        row_.accumulatedImpulse = impulse;
    }

    /// Get the accumulated impulse after solving.
    [[nodiscard]] float getAccumulatedImpulse() const noexcept {
        return row_.accumulatedImpulse;
    }

    /// Number of constraint rows.
    [[nodiscard]] static constexpr uint32_t rowCount() noexcept { return 1; }

private:
    JointRow row_;  ///< Single Jacobian row for the distance constraint.

    /// Find body index by ID (O(N) scan — Module 13 will use hash map).
    [[nodiscard]] static PULSE_FORCE_INLINE uint32_t findBodyIndex(
        const SolverBody* bodies, uint32_t count, uint32_t bodyId) noexcept
    {
        for (uint32_t i = 0; i < count; ++i) {
            if (bodies[i].bodyId == bodyId) return i;
        }
        return 0xFFFFFFFFu;
    }
};

} // namespace pulse
