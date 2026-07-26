/**
 * @file hinge_constraint.h
 * @brief Revolute (hinge) joint — allows rotation around one axis only.
 *
 * Constrains two bodies so that:
 *  1. Their anchor points coincide (3 linear rows).
 *  2. Two perpendicular axes stay aligned (2 angular rows).
 *  3. Optionally: angular limits restrict the hinge angle.
 *  4. Optionally: a motor drives the hinge to a target angular velocity.
 *
 * Total: 5 base rows + up to 1 limit row + up to 1 motor row = 7 max.
 *
 * The hinge axis is defined in each body's local frame. After the bodies
 * rotate, the world-space axes should remain parallel (the 2 angular
 * constraint rows enforce this).
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
 * @class HingeConstraint
 * @brief Revolute joint allowing rotation around a single axis.
 */
class HingeConstraint {
public:
    static constexpr uint32_t kMaxRows = 7;

    // ── Configuration ────────────────────────────────────────────────────

    Vec3  localAnchorA;      ///< Anchor point in body A's local space.
    Vec3  localAnchorB;      ///< Anchor point in body B's local space.
    Vec3  localAxisA;        ///< Hinge axis in body A's local space (unit vector).
    Vec3  localAxisB;        ///< Hinge axis in body B's local space (unit vector).

    ConstraintHeader header; ///< Type tag + body IDs.
    JointLimit angularLimit; ///< Optional angular limits around the hinge axis.
    JointMotor motor;        ///< Optional motor drive.

    // ── Construction ─────────────────────────────────────────────────────

    PULSE_FORCE_INLINE HingeConstraint() noexcept
        : localAnchorA(Vec3::zero()),
          localAnchorB(Vec3::zero()),
          localAxisA(Vec3::unitY()),
          localAxisB(Vec3::unitY()),
          header(ConstraintType::Hinge, 0xFFFFFFFFu, 0xFFFFFFFFu),
          angularLimit(),
          motor(),
          activeRowCount_(0),
          hingeAngle_(0.0f)
    {}

    /**
     * @brief Construct a hinge constraint.
     * @param anchorA   Anchor in body A's local space.
     * @param anchorB   Anchor in body B's local space.
     * @param axisA     Hinge axis in body A's local space (normalized).
     * @param axisB     Hinge axis in body B's local space (normalized).
     * @param idA       Body A identifier.
     * @param idB       Body B identifier.
     */
    PULSE_FORCE_INLINE HingeConstraint(Vec3 anchorA, Vec3 anchorB,
                                        Vec3 axisA, Vec3 axisB,
                                        uint32_t idA, uint32_t idB) noexcept
        : localAnchorA(anchorA),
          localAnchorB(anchorB),
          localAxisA(axisA.normalized()),
          localAxisB(axisB.normalized()),
          header(ConstraintType::Hinge, idA, idB),
          angularLimit(),
          motor(),
          activeRowCount_(0),
          hingeAngle_(0.0f)
    {}

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

        // World-space anchors (treating local anchors as offsets without rotation
        // until Module 11 provides body orientation).
        Vec3 rA = localAnchorA;
        Vec3 rB = localAnchorB;
        Vec3 pA = bodyA.position + rA;
        Vec3 pB = bodyB.position + rB;

        // World-space hinge axes (without body rotation, use local directly)
        Vec3 axisA = localAxisA;
        Vec3 axisB = localAxisB;

        // ── 3 linear rows: anchor coincidence ────────────────────────────
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

        // ── 2 angular rows: perpendicular axis alignment ─────────────────
        // Build an orthonormal frame around axisA
        Vec3 perpU, perpV;
        math::orthonormalBasis(axisA, perpU, perpV);

        // Angular errors: the components of axisB perpendicular to axisA
        // should be zero.
        float angErrorU = perpU.dot(axisB);
        float angErrorV = perpV.dot(axisB);

        auto addAngularRow = [&](Vec3 axis, float error) {
            JointRow& row = rows_[activeRowCount_];
            row.indexA = idxA;
            row.indexB = idxB;
            setupAngularRow(row, bodyA, bodyB, axis, error,
                           config.baumgarte, invDt);
            activeRowCount_++;
        };

        addAngularRow(perpU, angErrorU);
        addAngularRow(perpV, angErrorV);

        // ── Compute hinge angle ──────────────────────────────────────────
        // Project axisB onto the plane perpendicular to axisA
        hingeAngle_ = std::atan2(perpV.dot(axisB), perpU.dot(axisB));

        // ── Optional: angular limit row ──────────────────────────────────
        if (angularLimit.enabled) {
            JointLimitState limitState = angularLimit.classify(hingeAngle_);
            if (limitState != JointLimitState::Inactive) {
                float limitError = 0.0f;
                if (limitState == JointLimitState::AtLower) {
                    limitError = hingeAngle_ - angularLimit.lowerLimit;
                } else if (limitState == JointLimitState::AtUpper) {
                    limitError = hingeAngle_ - angularLimit.upperLimit;
                } else if (limitState == JointLimitState::Locked) {
                    limitError = hingeAngle_ - angularLimit.lowerLimit;
                }

                JointRow& row = rows_[activeRowCount_];
                row.indexA = idxA;
                row.indexB = idxB;
                setupLimitRow(row, bodyA, bodyB, axisA, limitError,
                             limitState, config.baumgarte, invDt, false);
                activeRowCount_++;
            }
        }

        // ── Optional: motor row ──────────────────────────────────────────
        if (motor.enabled) {
            JointRow& row = rows_[activeRowCount_];
            row.indexA = idxA;
            row.indexB = idxB;
            setupMotorRow(row, bodyA, bodyB, axisA, motor, dt, false);
            activeRowCount_++;
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
    [[nodiscard]] float getHingeAngle() const noexcept { return hingeAngle_; }
    [[nodiscard]] static constexpr uint32_t maxRowCount() noexcept { return kMaxRows; }

private:
    JointRow rows_[kMaxRows];
    uint32_t activeRowCount_;
    float    hingeAngle_;

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
