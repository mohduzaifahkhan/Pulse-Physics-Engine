/**
 * @file six_dof_constraint.h
 * @brief Fully configurable 6-DoF joint.
 *
 * Each of the 6 degrees of freedom (3 translational: X, Y, Z and
 * 3 rotational: RX, RY, RZ) can be independently set to:
 *  - Free:   no constraint on this DoF.
 *  - Locked: DoF is fully constrained (error-corrected to zero).
 *  - Limited: DoF has lower/upper bounds.
 *
 * Each DoF also supports an optional motor.
 *
 * Total: up to 6 rows (1 per non-free DoF) + up to 6 motor rows = 12 max.
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
 * @enum DofMode
 * @brief Mode for a single degree of freedom.
 */
enum class DofMode : uint8_t {
    Free    = 0,  ///< No constraint on this DoF.
    Locked  = 1,  ///< Fully constrained.
    Limited = 2   ///< Constrained within [lower, upper] bounds.
};

/**
 * @enum DofIndex
 * @brief Named indices for the 6 degrees of freedom.
 */
enum class DofIndex : uint8_t {
    TransX = 0, TransY = 1, TransZ = 2,
    RotX   = 3, RotY   = 4, RotZ   = 5,
    Count  = 6
};

/**
 * @class SixDofConstraint
 * @brief Generic 6-DoF configurable joint.
 */
class SixDofConstraint {
public:
    static constexpr uint32_t kDofCount = 6;
    static constexpr uint32_t kMaxRows = 12; // 6 constraint + 6 motor

    // ── Configuration ────────────────────────────────────────────────────

    Vec3  localAnchorA;
    Vec3  localAnchorB;

    ConstraintHeader header;

    DofMode    modes[kDofCount];    ///< Mode for each DoF.
    JointLimit limits[kDofCount];   ///< Limits for each DoF (used when Limited).
    JointMotor motors[kDofCount];   ///< Motors for each DoF.

    // ── Construction ─────────────────────────────────────────────────────

    PULSE_FORCE_INLINE SixDofConstraint() noexcept
        : localAnchorA(Vec3::zero()),
          localAnchorB(Vec3::zero()),
          header(ConstraintType::SixDof, 0xFFFFFFFFu, 0xFFFFFFFFu),
          activeRowCount_(0)
    {
        for (uint32_t i = 0; i < kDofCount; ++i) {
            modes[i] = DofMode::Free;
        }
    }

    PULSE_FORCE_INLINE SixDofConstraint(Vec3 anchorA, Vec3 anchorB,
                                         uint32_t idA, uint32_t idB) noexcept
        : localAnchorA(anchorA),
          localAnchorB(anchorB),
          header(ConstraintType::SixDof, idA, idB),
          activeRowCount_(0)
    {
        for (uint32_t i = 0; i < kDofCount; ++i) {
            modes[i] = DofMode::Free;
        }
    }

    /// Lock a specific DoF.
    void lockDof(DofIndex dof) noexcept {
        modes[static_cast<uint8_t>(dof)] = DofMode::Locked;
    }

    /// Free a specific DoF.
    void freeDof(DofIndex dof) noexcept {
        modes[static_cast<uint8_t>(dof)] = DofMode::Free;
    }

    /// Set a limited DoF with bounds.
    void limitDof(DofIndex dof, float lower, float upper) noexcept {
        uint8_t idx = static_cast<uint8_t>(dof);
        modes[idx] = DofMode::Limited;
        limits[idx] = JointLimit(lower, upper);
    }

    /// Set a motor on a specific DoF.
    void setMotor(DofIndex dof, float targetVelocity, float maxForce) noexcept {
        uint8_t idx = static_cast<uint8_t>(dof);
        motors[idx] = JointMotor(targetVelocity, maxForce);
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
        Vec3 linearError = pB - pA;

        // World-space axes for each DoF
        const Vec3 axes[kDofCount] = {
            Vec3::unitX(), Vec3::unitY(), Vec3::unitZ(),  // Translation
            Vec3::unitX(), Vec3::unitY(), Vec3::unitZ()   // Rotation
        };

        // Error values: [transX, transY, transZ, rotX, rotY, rotZ]
        // Angular errors are 0 for now (no body orientation tracking yet)
        float errors[kDofCount] = {
            linearError.getX(), linearError.getY(), linearError.getZ(),
            0.0f, 0.0f, 0.0f
        };

        // Process each DoF
        for (uint32_t d = 0; d < kDofCount; ++d) {
            bool isLinear = (d < 3);
            Vec3 axis = axes[d];
            float error = errors[d];

            // ── Constraint row for locked/limited DoFs ───────────────────
            if (modes[d] == DofMode::Locked) {
                JointRow& row = rows_[activeRowCount_];
                row.indexA = idxA;
                row.indexB = idxB;
                if (isLinear) {
                    setupLinearRow(row, bodyA, bodyB, rA, rB, axis, error,
                                  config.baumgarte, invDt);
                } else {
                    setupAngularRow(row, bodyA, bodyB, axis, error,
                                   config.baumgarte, invDt);
                }
                activeRowCount_++;
            } else if (modes[d] == DofMode::Limited) {
                JointLimitState state = limits[d].classify(error);
                if (state != JointLimitState::Inactive) {
                    float limitError = 0.0f;
                    if (state == JointLimitState::AtLower) {
                        limitError = error - limits[d].lowerLimit;
                    } else if (state == JointLimitState::AtUpper) {
                        limitError = error - limits[d].upperLimit;
                    } else if (state == JointLimitState::Locked) {
                        limitError = error - limits[d].lowerLimit;
                    }
                    JointRow& row = rows_[activeRowCount_];
                    row.indexA = idxA;
                    row.indexB = idxB;
                    setupLimitRow(row, bodyA, bodyB, axis, limitError,
                                 state, config.baumgarte, invDt, isLinear);
                    activeRowCount_++;
                }
            }

            // ── Motor row (if enabled, regardless of constraint mode) ────
            if (motors[d].enabled) {
                JointRow& row = rows_[activeRowCount_];
                row.indexA = idxA;
                row.indexB = idxB;
                setupMotorRow(row, bodyA, bodyB, axis, motors[d], dt, isLinear);
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
    [[nodiscard]] static constexpr uint32_t maxRowCount() noexcept { return kMaxRows; }

private:
    JointRow rows_[kMaxRows];
    uint32_t activeRowCount_;

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
