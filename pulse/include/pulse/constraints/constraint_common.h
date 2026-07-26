/**
 * @file constraint_common.h
 * @brief Shared types and utilities for all joint constraints (Module 10).
 *
 * Defines the fundamental building blocks used by every constraint type:
 *  - JointLimitState / JointLimit — angular or linear limit configuration.
 *  - JointMotor — velocity/position motor drive parameters.
 *  - SoftConstraintParams — spring-damper softness via CFM/ERP.
 *  - JointRow — pre-computed Jacobian row consumed by the solver.
 *  - solveJointRow() — the shared sequential-impulse solve kernel.
 *
 * Design: No virtual dispatch. Every constraint type fills an array of
 * JointRow during initialize() and the solver iterates them uniformly.
 * This keeps constraint data in flat, SIMD-friendly arrays.
 *
 * The solver uses the same diagonal inverse inertia (Vec3) as SolverBody
 * in Module 9.  When Module 11 upgrades to full Mat3, the effective mass
 * computation here will need a corresponding update.
 */

#pragma once

#include <pulse/math/math_common.h>
#include <pulse/math/vec3.h>
#include <pulse/math/quat.h>
#include <pulse/solver/solver_common.h>

#include <cstdint>
#include <cmath>

namespace pulse {

// ── Joint limit ──────────────────────────────────────────────────────────────

/**
 * @enum JointLimitState
 * @brief Current state of a joint limit during solving.
 */
enum class JointLimitState : uint8_t {
    Inactive = 0,  ///< Limit not active (within free range).
    AtLower  = 1,  ///< At or beyond the lower limit.
    AtUpper  = 2,  ///< At or beyond the upper limit.
    Locked   = 3   ///< Both limits equal — DoF is locked.
};

/**
 * @struct JointLimit
 * @brief Configuration for an angular or linear joint limit.
 */
struct JointLimit {
    float lowerLimit;        ///< Lower bound (radians or metres).
    float upperLimit;        ///< Upper bound (radians or metres).
    bool  enabled;           ///< Whether this limit is active.

    PULSE_FORCE_INLINE JointLimit() noexcept
        : lowerLimit(-math::Infinity), upperLimit(math::Infinity), enabled(false)
    {}

    PULSE_FORCE_INLINE JointLimit(float lower, float upper) noexcept
        : lowerLimit(lower), upperLimit(upper), enabled(true)
    {}

    /// Check if the limit is effectively locked (lower == upper within tolerance).
    [[nodiscard]] PULSE_FORCE_INLINE bool isLocked() const noexcept {
        return enabled && math::fastAbs(upperLimit - lowerLimit) < math::Epsilon;
    }

    /// Classify the current value against this limit.
    [[nodiscard]] PULSE_FORCE_INLINE JointLimitState classify(float value) const noexcept {
        if (!enabled) return JointLimitState::Inactive;
        if (isLocked()) return JointLimitState::Locked;
        if (value <= lowerLimit) return JointLimitState::AtLower;
        if (value >= upperLimit) return JointLimitState::AtUpper;
        return JointLimitState::Inactive;
    }
};

// ── Joint motor ──────────────────────────────────────────────────────────────

/**
 * @struct JointMotor
 * @brief Configuration for a velocity or position motor drive.
 */
struct JointMotor {
    float targetVelocity;    ///< Target velocity (rad/s or m/s).
    float maxForce;          ///< Maximum impulse per time step (N·s or N·m·s).
    bool  enabled;           ///< Whether this motor is active.

    PULSE_FORCE_INLINE JointMotor() noexcept
        : targetVelocity(0.0f), maxForce(0.0f), enabled(false)
    {}

    PULSE_FORCE_INLINE JointMotor(float vel, float force) noexcept
        : targetVelocity(vel), maxForce(force), enabled(true)
    {}
};

// ── Soft constraint parameters ───────────────────────────────────────────────

/**
 * @struct SoftConstraintParams
 * @brief Spring-damper softness parameters for compliant constraints.
 *
 * Converts user-facing frequency (Hz) and damping ratio (ζ) into
 * internal CFM (γ) and ERP (β) values used to modify the effective
 * mass and bias terms.
 *
 * Usage:
 *   SoftConstraintParams soft(30.0f, 0.7f);  // 30 Hz, ζ = 0.7
 *   soft.compute(dt);
 *   // Then: effectiveMass = 1.0 / (K + gamma)
 *   //       bias = beta * C * invDt
 */
struct SoftConstraintParams {
    float frequency;         ///< Natural frequency in Hz.
    float dampingRatio;      ///< Damping ratio ζ (1.0 = critical).
    bool  enabled;           ///< Whether soft mode is active.

    // Computed values (call compute() each frame with current dt):
    float gamma;             ///< Constraint force mixing (CFM).
    float beta;              ///< Error reduction parameter (ERP).

    PULSE_FORCE_INLINE SoftConstraintParams() noexcept
        : frequency(0.0f), dampingRatio(0.0f), enabled(false),
          gamma(0.0f), beta(0.0f)
    {}

    PULSE_FORCE_INLINE SoftConstraintParams(float freq, float zeta) noexcept
        : frequency(freq), dampingRatio(zeta), enabled(freq > 0.0f),
          gamma(0.0f), beta(0.0f)
    {}

    /// Compute CFM and ERP from frequency and damping ratio.
    /// Must be called once per frame with the current time step.
    PULSE_FORCE_INLINE void compute(float dt) noexcept {
        if (!enabled || dt < math::Epsilon) {
            gamma = 0.0f;
            beta = 1.0f;
            return;
        }
        const float omega = math::TwoPi * frequency;
        const float d = 2.0f * dampingRatio * omega; // 2ζω
        const float k = omega * omega;                // ω²
        const float denom = dt * (d + dt * k);        // dt * (2ζω + dt·ω²)
        if (denom < math::Epsilon) {
            gamma = 0.0f;
            beta = 1.0f;
            return;
        }
        gamma = 1.0f / denom;
        beta = dt * k / (d + dt * k);
    }

    /// Convert stiffness (N/m) and damping (Ns/m) to frequency/ratio.
    /// Requires the effective mass (reduced mass of the two-body system).
    [[nodiscard]] static PULSE_FORCE_INLINE SoftConstraintParams fromStiffness(
        float stiffness, float damping, float effectiveMass) noexcept
    {
        if (effectiveMass < math::Epsilon || stiffness < math::Epsilon) {
            return SoftConstraintParams();
        }
        float omega = math::fastSqrt(stiffness / effectiveMass);
        float freq = omega / math::TwoPi;
        float zeta = damping / (2.0f * effectiveMass * omega);
        return SoftConstraintParams(freq, zeta);
    }
};

// ── Joint row (Jacobian row) ─────────────────────────────────────────────────

/**
 * @struct JointRow
 * @brief A single pre-computed Jacobian row for the sequential-impulse solver.
 *
 * Each joint constraint is decomposed into one or more scalar constraint
 * equations C = 0.  For each equation, we pre-compute the 1×12 Jacobian
 * J = [J_vA, J_wA, J_vB, J_wB] and the scalar effective mass, bias,
 * and impulse bounds.
 *
 * The solver iterates over JointRows identically to how it iterates
 * over VelocityConstraints — computing Δλ = -K⁻¹(Jv + b) and applying
 * impulses with accumulated clamping.
 */
struct JointRow {
    // ── Jacobian components ──────────────────────────────────────────────
    Vec3  jvA;              ///< Linear Jacobian for body A.
    Vec3  jwA;              ///< Angular Jacobian for body A.
    Vec3  jvB;              ///< Linear Jacobian for body B.
    Vec3  jwB;              ///< Angular Jacobian for body B.

    // ── Pre-computed solver data ─────────────────────────────────────────
    float effectiveMass;    ///< 1 / (J M⁻¹ Jᵀ + γ).
    float bias;             ///< Constraint bias (Baumgarte + restitution + soft).
    float accumulatedImpulse; ///< Total impulse applied so far.
    float minImpulse;       ///< Lower impulse clamp.
    float maxImpulse;       ///< Upper impulse clamp.

    // ── Body references ──────────────────────────────────────────────────
    uint32_t indexA;        ///< Index into SolverBody array for body A.
    uint32_t indexB;        ///< Index into SolverBody array for body B.

    PULSE_FORCE_INLINE JointRow() noexcept
        : jvA(Vec3::zero()), jwA(Vec3::zero()),
          jvB(Vec3::zero()), jwB(Vec3::zero()),
          effectiveMass(0.0f), bias(0.0f),
          accumulatedImpulse(0.0f),
          minImpulse(-math::Infinity),
          maxImpulse(math::Infinity),
          indexA(0xFFFFFFFFu), indexB(0xFFFFFFFFu)
    {}
};

// ── Solver helpers ───────────────────────────────────────────────────────────

/**
 * @brief Compute the effective mass for a single Jacobian row.
 *
 * K = m_A⁻¹·|J_vA|² + (J_wA ⊙ invI_A)·J_wA + m_B⁻¹·|J_vB|² + (J_wB ⊙ invI_B)·J_wB
 *
 * Uses the diagonal inverse inertia (Vec3) from SolverBody, consistent
 * with ContactSolver::computeEffectiveMass().
 *
 * @param bodyA  Solver body A.
 * @param bodyB  Solver body B.
 * @param row    The Jacobian row to compute effective mass for.
 * @param gamma  Soft constraint CFM term (0 for rigid constraints).
 * @return Effective mass (1/K), or 0 if K is near-zero.
 */
[[nodiscard]] PULSE_FORCE_INLINE float computeJointEffectiveMass(
    const SolverBody& bodyA, const SolverBody& bodyB,
    const JointRow& row, float gamma = 0.0f) noexcept
{
    float K = 0.0f;

    // Body A contribution
    K += bodyA.invMass * row.jvA.dot(row.jvA);
    Vec3 iwA = bodyA.invInertia * row.jwA; // Component-wise: invI ⊙ J_wA
    K += iwA.dot(row.jwA);

    // Body B contribution
    K += bodyB.invMass * row.jvB.dot(row.jvB);
    Vec3 iwB = bodyB.invInertia * row.jwB;
    K += iwB.dot(row.jwB);

    // Add soft constraint term
    K += gamma;

    return (K > math::Epsilon) ? 1.0f / K : 0.0f;
}

/**
 * @brief Solve a single Jacobian row — the core sequential-impulse kernel.
 *
 * Computes Δλ = -effectiveMass * (Jv + bias), applies accumulated
 * clamping, and updates both body velocities.
 *
 * This function is called for every JointRow during each velocity
 * iteration, identically to how contact constraints are solved.
 *
 * @param row    The Jacobian row (accumulated impulse is modified).
 * @param bodyA  Solver body A (velocities modified).
 * @param bodyB  Solver body B (velocities modified).
 */
PULSE_FORCE_INLINE void solveJointRow(JointRow& row,
                                       SolverBody& bodyA,
                                       SolverBody& bodyB) noexcept
{
    // Compute Jv = J_vA · vA + J_wA · ωA + J_vB · vB + J_wB · ωB
    float jv = row.jvA.dot(bodyA.linearVelocity)
             + row.jwA.dot(bodyA.angularVelocity)
             + row.jvB.dot(bodyB.linearVelocity)
             + row.jwB.dot(bodyB.angularVelocity);

    // Impulse magnitude
    float lambda = -row.effectiveMass * (jv + row.bias);

    // Accumulated clamping
    float oldImpulse = row.accumulatedImpulse;
    row.accumulatedImpulse = math::clamp(oldImpulse + lambda,
                                          row.minImpulse, row.maxImpulse);
    float delta = row.accumulatedImpulse - oldImpulse;

    // Apply impulse to body A: v += invM * J_v * δ,  ω += invI ⊙ (J_w * δ)
    bodyA.linearVelocity  += row.jvA * (bodyA.invMass * delta);
    bodyA.angularVelocity += bodyA.invInertia * (row.jwA * delta);

    // Apply impulse to body B
    bodyB.linearVelocity  += row.jvB * (bodyB.invMass * delta);
    bodyB.angularVelocity += bodyB.invInertia * (row.jwB * delta);
}

// ── Utility functions ────────────────────────────────────────────────────────

/**
 * @brief Compute the relative angular velocity between two bodies along an axis.
 */
[[nodiscard]] PULSE_FORCE_INLINE float relativeAngularVelocity(
    const SolverBody& bodyA, const SolverBody& bodyB, Vec3 axis) noexcept
{
    return (bodyA.angularVelocity - bodyB.angularVelocity).dot(axis);
}

/**
 * @brief Compute the relative linear velocity between two bodies along an axis.
 *
 * Takes into account angular velocity contributions at the anchor offsets.
 */
[[nodiscard]] PULSE_FORCE_INLINE float relativeLinearVelocity(
    const SolverBody& bodyA, const SolverBody& bodyB,
    Vec3 rA, Vec3 rB, Vec3 axis) noexcept
{
    Vec3 vA = bodyA.linearVelocity + bodyA.angularVelocity.cross(rA);
    Vec3 vB = bodyB.linearVelocity + bodyB.angularVelocity.cross(rB);
    return (vA - vB).dot(axis);
}

/**
 * @brief Set up a point-to-point (linear) Jacobian row.
 *
 * For constraint: (pA + rA) - (pB + rB) projected onto 'axis' = 0.
 * Jacobian: J = [axis, rA × axis, -axis, -(rB × axis)]
 *
 * Note the sign convention: J_vA = +axis, J_vB = -axis means
 * a positive impulse pushes A in +axis and B in -axis direction.
 */
PULSE_FORCE_INLINE void setupLinearRow(JointRow& row,
                                        const SolverBody& bodyA,
                                        const SolverBody& bodyB,
                                        Vec3 rA, Vec3 rB, Vec3 axis,
                                        float positionError,
                                        float baumgarte, float invDt,
                                        const SoftConstraintParams& soft = SoftConstraintParams()) noexcept
{
    row.jvA = -axis;
    row.jwA = -(rA.cross(axis));
    row.jvB = axis;
    row.jwB = rB.cross(axis);

    float gamma = soft.enabled ? soft.gamma : 0.0f;
    row.effectiveMass = computeJointEffectiveMass(bodyA, bodyB, row, gamma);

    if (soft.enabled) {
        row.bias = soft.beta * positionError * invDt;
    } else {
        row.bias = baumgarte * positionError * invDt;
    }

    row.accumulatedImpulse = 0.0f;
    row.minImpulse = -math::Infinity;
    row.maxImpulse = math::Infinity;
}

/**
 * @brief Set up an angular Jacobian row.
 *
 * For constraint: relative angular displacement projected onto 'axis' = 0.
 * Jacobian: J = [0, axis, 0, -axis]
 */
PULSE_FORCE_INLINE void setupAngularRow(JointRow& row,
                                          const SolverBody& bodyA,
                                          const SolverBody& bodyB,
                                          Vec3 axis,
                                          float angularError,
                                          float baumgarte, float invDt,
                                          const SoftConstraintParams& soft = SoftConstraintParams()) noexcept
{
    row.jvA = Vec3::zero();
    row.jwA = axis;
    row.jvB = Vec3::zero();
    row.jwB = -axis;

    float gamma = soft.enabled ? soft.gamma : 0.0f;
    row.effectiveMass = computeJointEffectiveMass(bodyA, bodyB, row, gamma);

    if (soft.enabled) {
        row.bias = soft.beta * angularError * invDt;
    } else {
        row.bias = baumgarte * angularError * invDt;
    }

    row.accumulatedImpulse = 0.0f;
    row.minImpulse = -math::Infinity;
    row.maxImpulse = math::Infinity;
}

/**
 * @brief Set up a motor Jacobian row for angular velocity drive.
 *
 * The bias is set to the target velocity (not a position error).
 * Impulse is clamped to [-maxForce*dt, +maxForce*dt].
 */
PULSE_FORCE_INLINE void setupMotorRow(JointRow& row,
                                        const SolverBody& bodyA,
                                        const SolverBody& bodyB,
                                        Vec3 axis,
                                        const JointMotor& motor,
                                        float dt, bool isLinear = false) noexcept
{
    if (isLinear) {
        row.jvA = axis;
        row.jwA = Vec3::zero();
        row.jvB = -axis;
        row.jwB = Vec3::zero();
    } else {
        row.jvA = Vec3::zero();
        row.jwA = axis;
        row.jvB = Vec3::zero();
        row.jwB = -axis;
    }

    row.effectiveMass = computeJointEffectiveMass(bodyA, bodyB, row, 0.0f);
    row.bias = motor.targetVelocity; // Drive towards target velocity
    row.accumulatedImpulse = 0.0f;

    float maxImpulse = motor.maxForce * dt;
    row.minImpulse = -maxImpulse;
    row.maxImpulse = maxImpulse;
}

/**
 * @brief Set up a limit Jacobian row.
 *
 * Configures impulse bounds based on which limit is violated:
 * - AtLower: λ ≥ 0 (push away from lower limit)
 * - AtUpper: λ ≤ 0 (push away from upper limit)
 * - Locked:  unconstrained λ (both directions)
 */
PULSE_FORCE_INLINE void setupLimitRow(JointRow& row,
                                        const SolverBody& bodyA,
                                        const SolverBody& bodyB,
                                        Vec3 axis,
                                        float error,
                                        JointLimitState state,
                                        float baumgarte, float invDt,
                                        bool isLinear = false) noexcept
{
    if (isLinear) {
        row.jvA = axis;
        row.jwA = Vec3::zero();
        row.jvB = -axis;
        row.jwB = Vec3::zero();
    } else {
        row.jvA = Vec3::zero();
        row.jwA = axis;
        row.jvB = Vec3::zero();
        row.jwB = -axis;
    }

    row.effectiveMass = computeJointEffectiveMass(bodyA, bodyB, row, 0.0f);
    row.bias = baumgarte * error * invDt;
    row.accumulatedImpulse = 0.0f;

    switch (state) {
        case JointLimitState::AtLower:
            row.minImpulse = 0.0f;
            row.maxImpulse = math::Infinity;
            break;
        case JointLimitState::AtUpper:
            row.minImpulse = -math::Infinity;
            row.maxImpulse = 0.0f;
            break;
        case JointLimitState::Locked:
            row.minImpulse = -math::Infinity;
            row.maxImpulse = math::Infinity;
            break;
        default:
            row.minImpulse = 0.0f;
            row.maxImpulse = 0.0f;
            break;
    }
}

} // namespace pulse
