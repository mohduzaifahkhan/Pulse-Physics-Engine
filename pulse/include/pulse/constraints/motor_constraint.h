/**
 * @file motor_constraint.h
 * @brief Standalone velocity/position motor drive constraint.
 *
 * Drives a pair of bodies to achieve a target velocity (or position)
 * along a specified axis, either linear or angular.
 *
 * The motor applies impulses clamped to [-maxForce·dt, +maxForce·dt]
 * with a bias term set to the target velocity.  For position targeting,
 * the bias includes a PD-style error correction term.
 *
 * Uses 1 JointRow with the Jacobian set for either linear or angular mode.
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
 * @enum MotorMode
 * @brief Whether the motor drives linear or angular motion.
 */
enum class MotorMode : uint8_t {
    Linear  = 0,
    Angular = 1
};

/**
 * @enum MotorTarget
 * @brief Whether the motor targets a velocity or a position.
 */
enum class MotorTarget : uint8_t {
    Velocity = 0,  ///< Drive to target velocity (steady state).
    Position = 1   ///< Drive to target position (with PD control).
};

/**
 * @class MotorConstraint
 * @brief Standalone motor drive for linear or angular motion.
 */
class MotorConstraint {
public:
    // ── Configuration ────────────────────────────────────────────────────

    Vec3  axis;              ///< Motor drive axis (world space, unit vector).
    float targetVelocity;    ///< Target velocity (m/s or rad/s).
    float targetPosition;    ///< Target position (metres or radians) for Position mode.
    float maxForce;          ///< Maximum force/torque (N or N·m).
    float positionGain;      ///< P-gain for position mode error correction.

    MotorMode   mode;        ///< Linear or angular motor.
    MotorTarget target;      ///< Velocity or position targeting.

    ConstraintHeader header;

    // ── Construction ─────────────────────────────────────────────────────

    PULSE_FORCE_INLINE MotorConstraint() noexcept
        : axis(Vec3::unitY()),
          targetVelocity(0.0f),
          targetPosition(0.0f),
          maxForce(100.0f),
          positionGain(0.3f),
          mode(MotorMode::Angular),
          target(MotorTarget::Velocity),
          header(ConstraintType::Motor, 0xFFFFFFFFu, 0xFFFFFFFFu)
    {}

    PULSE_FORCE_INLINE MotorConstraint(Vec3 driveAxis, float maxF,
                                        MotorMode m, uint32_t idA, uint32_t idB) noexcept
        : axis(driveAxis.normalized()),
          targetVelocity(0.0f),
          targetPosition(0.0f),
          maxForce(maxF),
          positionGain(0.3f),
          mode(m),
          target(MotorTarget::Velocity),
          header(ConstraintType::Motor, idA, idB)
    {}

    // ── Solver interface ─────────────────────────────────────────────────

    void initialize(const SolverBody* bodies, uint32_t bodyCount,
                    const SolverConfig& config, float dt) noexcept
    {
        (void)config; // Not using baumgarte for motors

        uint32_t idxA = findBodyIndex(bodies, bodyCount, header.bodyIdA);
        uint32_t idxB = findBodyIndex(bodies, bodyCount, header.bodyIdB);

        row_.indexA = idxA;
        row_.indexB = idxB;

        if (idxA >= bodyCount || idxB >= bodyCount) return;

        const SolverBody& bodyA = bodies[idxA];
        const SolverBody& bodyB = bodies[idxB];

        bool isLinear = (mode == MotorMode::Linear);

        if (isLinear) {
            row_.jvA = axis;
            row_.jwA = Vec3::zero();
            row_.jvB = -axis;
            row_.jwB = Vec3::zero();
        } else {
            row_.jvA = Vec3::zero();
            row_.jwA = axis;
            row_.jvB = Vec3::zero();
            row_.jwB = -axis;
        }

        row_.effectiveMass = computeJointEffectiveMass(bodyA, bodyB, row_, 0.0f);

        // Compute bias
        if (target == MotorTarget::Position) {
            // PD control: bias = targetVelocity + positionGain * (targetPos - currentPos)
            float currentValue = 0.0f;
            if (isLinear) {
                Vec3 delta = bodyB.position - bodyA.position;
                currentValue = delta.dot(axis);
            }
            // Angular position tracking would need orientation (Module 11)

            float posError = targetPosition - currentValue;
            row_.bias = targetVelocity + positionGain * posError;
        } else {
            row_.bias = targetVelocity;
        }

        row_.accumulatedImpulse = 0.0f;
        float maxImpulse = maxForce * dt;
        row_.minImpulse = -maxImpulse;
        row_.maxImpulse = maxImpulse;
    }

    void warmStart(SolverBody* bodies) const noexcept {
        if (row_.indexA >= 0xFFFFFFFFu || row_.indexB >= 0xFFFFFFFFu) return;
        SolverBody& bodyA = bodies[row_.indexA];
        SolverBody& bodyB = bodies[row_.indexB];
        float imp = row_.accumulatedImpulse;
        bodyA.linearVelocity  += row_.jvA * (bodyA.invMass * imp);
        bodyA.angularVelocity += bodyA.invInertia * (row_.jwA * imp);
        bodyB.linearVelocity  += row_.jvB * (bodyB.invMass * imp);
        bodyB.angularVelocity += bodyB.invInertia * (row_.jwB * imp);
    }

    void solveVelocity(SolverBody* bodies) noexcept {
        if (row_.indexA >= 0xFFFFFFFFu || row_.indexB >= 0xFFFFFFFFu) return;
        solveJointRow(row_, bodies[row_.indexA], bodies[row_.indexB]);
    }

    [[nodiscard]] const JointRow& getRow() const noexcept { return row_; }
    [[nodiscard]] float getAccumulatedImpulse() const noexcept { return row_.accumulatedImpulse; }
    [[nodiscard]] static constexpr uint32_t rowCount() noexcept { return 1; }

private:
    JointRow row_;

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
