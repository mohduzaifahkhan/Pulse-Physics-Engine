/**
 * @file integration_common.h
 * @brief Shared types and configuration for the integration module (Module 12).
 *
 * Defines the integrator type selector, global integration parameters,
 * and velocity/speed clamping utilities used by all integrators
 * (Semi-Implicit Euler, Velocity Verlet, RK4).
 *
 * Design: Integrators operate directly on RigidBodyStore SoA arrays for
 * maximum cache efficiency. Static, kinematic, and sleeping bodies are
 * skipped automatically. Per-body gravity scale is honoured.
 */

#pragma once

#include <pulse/math/math_common.h>
#include <pulse/math/vec3.h>
#include <pulse/math/quat.h>

#include <cstdint>

namespace pulse {

// ── Integrator type selector ─────────────────────────────────────────────────

/**
 * @enum IntegratorType
 * @brief Selects the numerical integration method.
 */
enum class IntegratorType : uint8_t {
    /// Semi-Implicit (Symplectic) Euler — default for real-time physics.
    /// First-order, energy-preserving.  v += a*dt; x += v*dt.
    SemiImplicitEuler = 0,

    /// Velocity Verlet (Kick-Drift-Kick) — second-order accurate.
    /// Better energy conservation than Euler.  Popular for molecular dynamics.
    VelocityVerlet = 1,

    /// Runge-Kutta 4th order — four derivative evaluations per step.
    /// Fourth-order accurate.  Use for orbital mechanics or high-accuracy sims.
    RK4 = 2
};

// ── Integration configuration ────────────────────────────────────────────────

/**
 * @struct IntegrationConfig
 * @brief Global parameters for the integration step.
 */
struct IntegrationConfig {
    IntegratorType type;                ///< Which integrator to use.
    Vec3           gravity;             ///< World gravity (m/s²).
    float          maxLinearSpeed;      ///< Clamp linear speed (m/s).  0 = no clamp.
    float          maxAngularSpeed;     ///< Clamp angular speed (rad/s).  0 = no clamp.

    /// Default: semi-implicit Euler with Earth gravity, generous speed limits.
    PULSE_FORCE_INLINE IntegrationConfig() noexcept
        : type(IntegratorType::SemiImplicitEuler),
          gravity(Vec3(0.0f, -9.81f, 0.0f)),
          maxLinearSpeed(500.0f),
          maxAngularSpeed(100.0f)
    {}

    /// Custom configuration.
    PULSE_FORCE_INLINE IntegrationConfig(IntegratorType t, Vec3 grav,
                                          float maxLin, float maxAng) noexcept
        : type(t),
          gravity(grav),
          maxLinearSpeed(maxLin),
          maxAngularSpeed(maxAng)
    {}
};

// ── Velocity clamping utility ────────────────────────────────────────────────

namespace integration_detail {

    /// Clamp a vector's magnitude to maxSpeed.  If maxSpeed <= 0, no clamping.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 clampSpeed(Vec3 v, float maxSpeed) noexcept {
        if (maxSpeed <= 0.0f) return v;
        float speedSq = v.lengthSq();
        if (speedSq > maxSpeed * maxSpeed) {
            float invSpeed = maxSpeed * math::fastInvSqrt(speedSq);
            return v * invSpeed;
        }
        return v;
    }

    /// Apply velocity damping:  v *= 1 / (1 + damping * dt).
    /// This is unconditionally stable (never amplifies) unlike v *= (1 - d*dt).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 applyDamping(Vec3 v, float damping, float dt) noexcept {
        float factor = 1.0f / (1.0f + damping * dt);
        return v * factor;
    }

    /// Integrate quaternion rotation by angular velocity:
    ///   q' = normalize( q + 0.5 * dt * Quat(omega, 0) * q )
    [[nodiscard]] PULSE_FORCE_INLINE Quat integrateRotation(Quat q, Vec3 omega, float dt) noexcept {
        // Quaternion derivative: dq/dt = 0.5 * omega_quat * q
        // where omega_quat = Quat(omega.x, omega.y, omega.z, 0)
        Quat omegaQ(omega.getX(), omega.getY(), omega.getZ(), 0.0f);
        Quat qdot = omegaQ * q * 0.5f;
        Quat result = q + qdot * dt;
        result.normalize();
        return result;
    }

} // namespace integration_detail

} // namespace pulse
