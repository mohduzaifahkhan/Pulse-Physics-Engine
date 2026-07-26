/**
 * @file slider_constraint.h
 * @brief Prismatic (slider) joint — allows translation along one axis only.
 *
 * Constrains two bodies so that:
 *  1. Relative motion is restricted to one axis (2 linear rows for transverse).
 *  2. Relative orientation is locked (3 angular rows).
 *  3. Optionally: linear limits restrict the sliding distance.
 *  4. Optionally: a motor drives the slider to a target linear velocity.
 *
 * Total: 5 base rows + up to 1 limit row + up to 1 motor row = 7 max.
 */

#pragma once

#include "constraint_common.h"
#include <pulse/solver/solver_common.h>
#include <pulse/solver/constraint_base.h>
#include <pulse/math/vec3.h>
#include <pulse/math/math_common.h>

#include <cstdint>

namespace pulse {

/**
 * @class SliderConstraint
 * @brief Prismatic joint allowing translation along a single axis.
 */
class SliderConstraint {
public:
    static constexpr uint32_t kMaxRows = 7;

    // ── Configuration ────────────────────────────────────────────────────

    Vec3  localAnchorA;      ///< Anchor point in body A's local space.
    Vec3  localAnchorB;      ///< Anchor point in body B's local space.
    Vec3  localAxisA;        ///< Slider axis in body A's local space (unit vector).

    ConstraintHeader header; ///< Type tag + body IDs.
    JointLimit linearLimit;  ///< Optional linear limits along the slider axis.
    JointMotor motor;        ///< Optional linear motor drive.

    // ── Construction ─────────────────────────────────────────────────────

    PULSE_FORCE_INLINE SliderConstraint() noexcept
        : localAnchorA(Vec3::zero()),
          localAnchorB(Vec3::zero()),
          localAxisA(Vec3::unitX()),
          header(ConstraintType::Slider, 0xFFFFFFFFu, 0xFFFFFFFFu),
          linearLimit(),
          motor(),
          activeRowCount_(0),
          sliderPosition_(0.0f)
    {}

    PULSE_FORCE_INLINE SliderConstraint(Vec3 anchorA, Vec3 anchorB,
                                         Vec3 axisA,
                                         uint32_t idA, uint32_t idB) noexcept
        : localAnchorA(anchorA),
          localAnchorB(anchorB),
          localAxisA(axisA.normalized()),
          header(ConstraintType::Slider, idA, idB),
          linearLimit(),
          motor(),
          activeRowCount_(0),
          sliderPosition_(0.0f)
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

        Vec3 rA = localAnchorA;
        Vec3 rB = localAnchorB;
        Vec3 pA = bodyA.position + rA;
        Vec3 pB = bodyB.position + rB;

        Vec3 sliderAxis = localAxisA;
        Vec3 delta = pB - pA;

        // Build orthonormal basis perpendicular to slider axis
        Vec3 perpU, perpV;
        math::orthonormalBasis(sliderAxis, perpU, perpV);

        // Slider position = projection of delta onto the slider axis
        sliderPosition_ = delta.dot(sliderAxis);

        // ── 2 linear rows: constrain transverse motion ───────────────────
        float errorU = delta.dot(perpU);
        float errorV = delta.dot(perpV);

        auto addLinearRow = [&](Vec3 axis, float error) {
            JointRow& row = rows_[activeRowCount_];
            row.indexA = idxA;
            row.indexB = idxB;
            setupLinearRow(row, bodyA, bodyB, rA, rB, axis, error,
                          config.baumgarte, invDt);
            activeRowCount_++;
        };

        addLinearRow(perpU, errorU);
        addLinearRow(perpV, errorV);

        // ── 3 angular rows: lock relative orientation ────────────────────
        // Angular error = 0 (orientation should not change).
        // We use the 3 world axes as constraint axes.
        auto addAngularRow = [&](Vec3 axis) {
            JointRow& row = rows_[activeRowCount_];
            row.indexA = idxA;
            row.indexB = idxB;
            setupAngularRow(row, bodyA, bodyB, axis, 0.0f,
                           config.baumgarte, invDt);
            activeRowCount_++;
        };

        addAngularRow(Vec3::unitX());
        addAngularRow(Vec3::unitY());
        addAngularRow(Vec3::unitZ());

        // ── Optional: linear limit row ───────────────────────────────────
        if (linearLimit.enabled) {
            JointLimitState limitState = linearLimit.classify(sliderPosition_);
            if (limitState != JointLimitState::Inactive) {
                float limitError = 0.0f;
                if (limitState == JointLimitState::AtLower) {
                    limitError = sliderPosition_ - linearLimit.lowerLimit;
                } else if (limitState == JointLimitState::AtUpper) {
                    limitError = sliderPosition_ - linearLimit.upperLimit;
                } else if (limitState == JointLimitState::Locked) {
                    limitError = sliderPosition_ - linearLimit.lowerLimit;
                }

                JointRow& row = rows_[activeRowCount_];
                row.indexA = idxA;
                row.indexB = idxB;
                setupLimitRow(row, bodyA, bodyB, sliderAxis, limitError,
                             limitState, config.baumgarte, invDt, true);
                activeRowCount_++;
            }
        }

        // ── Optional: motor row ──────────────────────────────────────────
        if (motor.enabled) {
            JointRow& row = rows_[activeRowCount_];
            row.indexA = idxA;
            row.indexB = idxB;
            setupMotorRow(row, bodyA, bodyB, sliderAxis, motor, dt, true);
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
    [[nodiscard]] float getSliderPosition() const noexcept { return sliderPosition_; }
    [[nodiscard]] static constexpr uint32_t maxRowCount() noexcept { return kMaxRows; }

private:
    JointRow rows_[kMaxRows];
    uint32_t activeRowCount_;
    float    sliderPosition_;

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
