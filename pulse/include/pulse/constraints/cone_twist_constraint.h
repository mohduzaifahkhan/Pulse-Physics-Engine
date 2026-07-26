/**
 * @file cone_twist_constraint.h
 * @brief Cone-twist joint for ragdoll shoulders, hips, and similar joints.
 *
 * Constrains two bodies so that:
 *  1. Their anchor points coincide (3 linear point-to-point rows).
 *  2. Optionally: the swing angle is limited within a cone (1 row).
 *  3. Optionally: the twist angle is limited (1 row).
 *
 * Total: 3 base rows + up to 1 cone limit row + up to 1 twist limit row = 5 max.
 *
 * Swing/twist decomposition:
 *   The relative rotation q_rel = q_B * q_A⁻¹ is decomposed into:
 *   - Swing: rotation that moves the twist axis
 *   - Twist: rotation around the twist axis
 *   The swing angle θ = acos(dot(axisA_world, axisB_world)).
 */

#pragma once

#include "constraint_common.h"
#include <pulse/solver/solver_common.h>
#include <pulse/solver/constraint_base.h>
#include <pulse/math/vec3.h>
#include <pulse/math/quat.h>
#include <pulse/math/math_common.h>

#include <cstdint>
#include <cmath>

namespace pulse {

/**
 * @class ConeTwistConstraint
 * @brief Ball-socket joint with optional cone and twist limits.
 */
class ConeTwistConstraint {
public:
    static constexpr uint32_t kMaxRows = 5;

    // ── Configuration ────────────────────────────────────────────────────

    Vec3  localAnchorA;      ///< Anchor in body A's local space.
    Vec3  localAnchorB;      ///< Anchor in body B's local space.
    Vec3  localTwistAxisA;   ///< Twist axis in body A's local space.
    Vec3  localTwistAxisB;   ///< Twist axis in body B's local space.

    ConstraintHeader header;

    float swingLimit;        ///< Maximum swing angle (radians), 0 = no limit.
    float twistLimitLower;   ///< Lower twist limit (radians).
    float twistLimitUpper;   ///< Upper twist limit (radians).
    bool  swingLimitEnabled; ///< Whether the swing cone limit is active.
    bool  twistLimitEnabled; ///< Whether the twist limit is active.
    float softness;          ///< Softness factor [0,1] for limit enforcement.

    // ── Construction ─────────────────────────────────────────────────────

    PULSE_FORCE_INLINE ConeTwistConstraint() noexcept
        : localAnchorA(Vec3::zero()),
          localAnchorB(Vec3::zero()),
          localTwistAxisA(Vec3::unitX()),
          localTwistAxisB(Vec3::unitX()),
          header(ConstraintType::ConeTwist, 0xFFFFFFFFu, 0xFFFFFFFFu),
          swingLimit(math::Pi),
          twistLimitLower(-math::Pi),
          twistLimitUpper(math::Pi),
          swingLimitEnabled(false),
          twistLimitEnabled(false),
          softness(0.0f),
          activeRowCount_(0),
          swingAngle_(0.0f),
          twistAngle_(0.0f)
    {}

    PULSE_FORCE_INLINE ConeTwistConstraint(Vec3 anchorA, Vec3 anchorB,
                                            Vec3 twistAxisA, Vec3 twistAxisB,
                                            uint32_t idA, uint32_t idB) noexcept
        : localAnchorA(anchorA),
          localAnchorB(anchorB),
          localTwistAxisA(twistAxisA.normalized()),
          localTwistAxisB(twistAxisB.normalized()),
          header(ConstraintType::ConeTwist, idA, idB),
          swingLimit(math::Pi),
          twistLimitLower(-math::Pi),
          twistLimitUpper(math::Pi),
          swingLimitEnabled(false),
          twistLimitEnabled(false),
          softness(0.0f),
          activeRowCount_(0),
          swingAngle_(0.0f),
          twistAngle_(0.0f)
    {}

    /// Set the swing cone limit.
    void setSwingLimit(float maxAngle) noexcept {
        swingLimit = maxAngle;
        swingLimitEnabled = true;
    }

    /// Set the twist limits.
    void setTwistLimit(float lower, float upper) noexcept {
        twistLimitLower = lower;
        twistLimitUpper = upper;
        twistLimitEnabled = true;
    }

    // ── Solver interface ─────────────────────────────────────────────────

    void initialize(const SolverBody* bodies, uint32_t bodyCount,
                    const SolverConfig& config, float dt) noexcept
    {
        float invDt = (dt > math::Epsilon) ? 1.0f / dt : 0.0f;
        activeRowCount_ = 0;

        uint32_t idxA = findBodyIndex(bodies, bodyCount, header.bodyIdA);
        uint32_t idxB = findBodyIndex(bodies, bodyCount, header.bodyIdB);
        if (idxA >= bodyCount || idxB >= bodyCount) return;

        const SolverBody& bodyA = bodies[idxA];
        const SolverBody& bodyB = bodies[idxB];

        Vec3 rA = localAnchorA;
        Vec3 rB = localAnchorB;
        Vec3 pA = bodyA.position + rA;
        Vec3 pB = bodyB.position + rB;

        // ── 3 linear rows: point-to-point ────────────────────────────────
        Vec3 linearError = pB - pA;

        auto addLinearRow = [&](Vec3 axis, float error) {
            JointRow& row = rows_[activeRowCount_];
            row.indexA = idxA;
            row.indexB = idxB;
            setupLinearRow(row, bodyA, bodyB, rA, rB, axis, error,
                          config.baumgarte, invDt);
            activeRowCount_++;
        };

        addLinearRow(Vec3::unitX(), linearError.getX());
        addLinearRow(Vec3::unitY(), linearError.getY());
        addLinearRow(Vec3::unitZ(), linearError.getZ());

        // World-space twist axes
        Vec3 twistA = localTwistAxisA;
        Vec3 twistB = localTwistAxisB;

        // ── Compute swing angle ──────────────────────────────────────────
        float cosSwing = twistA.dot(twistB);
        cosSwing = math::clamp(cosSwing, -1.0f, 1.0f);
        swingAngle_ = std::acos(cosSwing);

        // ── Optional: swing limit row ────────────────────────────────────
        if (swingLimitEnabled && swingAngle_ > swingLimit) {
            // Swing axis = cross(twistA, twistB) normalized
            Vec3 swingAxis = twistA.cross(twistB);
            float swingAxisLen = swingAxis.length();
            if (swingAxisLen > math::Epsilon) {
                swingAxis = swingAxis / swingAxisLen;

                float error = swingAngle_ - swingLimit;
                float baumFactor = config.baumgarte * (1.0f - softness);

                JointRow& row = rows_[activeRowCount_];
                row.indexA = idxA;
                row.indexB = idxB;
                setupAngularRow(row, bodyA, bodyB, swingAxis, error,
                               baumFactor, invDt);
                // Swing limit: only push back into cone (λ ≥ 0)
                row.minImpulse = 0.0f;
                row.maxImpulse = math::Infinity;
                activeRowCount_++;
            }
        }

        // ── Compute twist angle ──────────────────────────────────────────
        // Project twistB onto the plane perpendicular to twistA
        Vec3 perpU, perpV;
        math::orthonormalBasis(twistA, perpU, perpV);
        twistAngle_ = std::atan2(perpV.dot(twistB), perpU.dot(twistB));

        // ── Optional: twist limit row ────────────────────────────────────
        if (twistLimitEnabled) {
            JointLimit tLimit(twistLimitLower, twistLimitUpper);
            JointLimitState tState = tLimit.classify(twistAngle_);
            if (tState != JointLimitState::Inactive) {
                float twistError = 0.0f;
                if (tState == JointLimitState::AtLower) {
                    twistError = twistAngle_ - twistLimitLower;
                } else if (tState == JointLimitState::AtUpper) {
                    twistError = twistAngle_ - twistLimitUpper;
                }

                float baumFactor = config.baumgarte * (1.0f - softness);

                JointRow& row = rows_[activeRowCount_];
                row.indexA = idxA;
                row.indexB = idxB;
                setupLimitRow(row, bodyA, bodyB, twistA, twistError,
                             tState, baumFactor, invDt, false);
                activeRowCount_++;
            }
        }
    }

    void warmStart(SolverBody* bodies) const noexcept {
        for (uint32_t i = 0; i < activeRowCount_; ++i) {
            const JointRow& row = rows_[i];
            if (row.indexA >= 0xFFFFFFFFu || row.indexB >= 0xFFFFFFFFu) continue;
            SolverBody& bodyA = bodies[row.indexA];
            SolverBody& bodyB = bodies[row.indexB];
            float imp = row.accumulatedImpulse;
            bodyA.linearVelocity  += row.jvA * (bodyA.invMass * imp);
            bodyA.angularVelocity += bodyA.invInertia * (row.jwA * imp);
            bodyB.linearVelocity  += row.jvB * (bodyB.invMass * imp);
            bodyB.angularVelocity += bodyB.invInertia * (row.jwB * imp);
        }
    }

    void solveVelocity(SolverBody* bodies) noexcept {
        for (uint32_t i = 0; i < activeRowCount_; ++i) {
            JointRow& row = rows_[i];
            if (row.indexA >= 0xFFFFFFFFu || row.indexB >= 0xFFFFFFFFu) continue;
            solveJointRow(row, bodies[row.indexA], bodies[row.indexB]);
        }
    }

    [[nodiscard]] uint32_t activeRowCount() const noexcept { return activeRowCount_; }
    [[nodiscard]] const JointRow& getRow(uint32_t i) const noexcept { return rows_[i]; }
    [[nodiscard]] float getSwingAngle() const noexcept { return swingAngle_; }
    [[nodiscard]] float getTwistAngle() const noexcept { return twistAngle_; }
    [[nodiscard]] static constexpr uint32_t maxRowCount() noexcept { return kMaxRows; }

private:
    JointRow rows_[kMaxRows];
    uint32_t activeRowCount_;
    float    swingAngle_;
    float    twistAngle_;

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
