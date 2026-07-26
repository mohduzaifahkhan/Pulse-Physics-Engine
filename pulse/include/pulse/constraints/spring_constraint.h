/**
 * @file spring_constraint.h
 * @brief Hookean spring constraint between two anchor points.
 *
 * Applies a spring force F = -k·x - c·v between two body anchor points,
 * where k = stiffness (N/m) and c = damping (Ns/m).
 *
 * Internally wraps a DistanceConstraint with soft parameters computed
 * from the user-facing stiffness and damping values. The spring has
 * a configurable rest length.
 *
 * The conversion from (stiffness, damping) to (frequency, dampingRatio)
 * uses the reduced mass of the two-body system:
 *   ω = sqrt(k / m_eff)
 *   ζ = c / (2 · m_eff · ω)
 */

#pragma once

#include "constraint_common.h"
#include "distance_constraint.h"
#include <pulse/solver/solver_common.h>
#include <pulse/math/vec3.h>
#include <pulse/math/math_common.h>

#include <cstdint>

namespace pulse {

/**
 * @class SpringConstraint
 * @brief A spring connecting two anchor points with configurable stiffness and damping.
 */
class SpringConstraint {
public:
    // ── Configuration ────────────────────────────────────────────────────

    Vec3  localAnchorA;      ///< Anchor in body A's local space.
    Vec3  localAnchorB;      ///< Anchor in body B's local space.
    float restLength;        ///< Natural length of the spring (metres).
    float stiffness;         ///< Spring stiffness k (N/m).
    float damping;           ///< Damping coefficient c (Ns/m).

    ConstraintHeader header;

    // ── Construction ─────────────────────────────────────────────────────

    PULSE_FORCE_INLINE SpringConstraint() noexcept
        : localAnchorA(Vec3::zero()),
          localAnchorB(Vec3::zero()),
          restLength(1.0f),
          stiffness(100.0f),
          damping(5.0f),
          header(ConstraintType::Spring, 0xFFFFFFFFu, 0xFFFFFFFFu)
    {}

    PULSE_FORCE_INLINE SpringConstraint(Vec3 anchorA, Vec3 anchorB,
                                         float length, float k, float c,
                                         uint32_t idA, uint32_t idB) noexcept
        : localAnchorA(anchorA),
          localAnchorB(anchorB),
          restLength(length),
          stiffness(k),
          damping(c),
          header(ConstraintType::Spring, idA, idB)
    {}

    // ── Solver interface ─────────────────────────────────────────────────

    void initialize(const SolverBody* bodies, uint32_t bodyCount,
                    const SolverConfig& config, float dt) noexcept
    {
        // Build the underlying distance constraint
        distConstraint_ = DistanceConstraint(
            localAnchorA, localAnchorB, restLength,
            header.bodyIdA, header.bodyIdB
        );

        // Compute reduced mass for spring parameter conversion
        uint32_t idxA = findBodyIndex(bodies, bodyCount, header.bodyIdA);
        uint32_t idxB = findBodyIndex(bodies, bodyCount, header.bodyIdB);

        float mEff = 1.0f; // Default
        if (idxA < bodyCount && idxB < bodyCount) {
            float mA = (bodies[idxA].invMass > math::Epsilon) ?
                        1.0f / bodies[idxA].invMass : 0.0f;
            float mB = (bodies[idxB].invMass > math::Epsilon) ?
                        1.0f / bodies[idxB].invMass : 0.0f;
            if (mA > math::Epsilon && mB > math::Epsilon) {
                mEff = (mA * mB) / (mA + mB); // Reduced mass
            } else if (mA > math::Epsilon) {
                mEff = mA;
            } else if (mB > math::Epsilon) {
                mEff = mB;
            }
        }

        // Convert stiffness/damping to frequency/dampingRatio
        distConstraint_.softParams = SoftConstraintParams::fromStiffness(
            stiffness, damping, mEff
        );

        distConstraint_.initialize(bodies, bodyCount, config, dt);
    }

    void warmStart(SolverBody* bodies) const noexcept {
        distConstraint_.warmStart(bodies);
    }

    void solveVelocity(SolverBody* bodies) noexcept {
        distConstraint_.solveVelocity(bodies);
    }

    [[nodiscard]] const JointRow& getRow() const noexcept {
        return distConstraint_.getRow();
    }

    [[nodiscard]] float getAccumulatedImpulse() const noexcept {
        return distConstraint_.getAccumulatedImpulse();
    }

    [[nodiscard]] static constexpr uint32_t rowCount() noexcept { return 1; }

private:
    DistanceConstraint distConstraint_;

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
