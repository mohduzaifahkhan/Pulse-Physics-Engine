/**
 * @file rk4.h
 * @brief Runge-Kutta 4th order (RK4) integrator — high-accuracy integration.
 *
 * Fourth-order accurate: the local truncation error is O(dt^5), making it
 * dramatically more accurate than Euler or Verlet for smooth trajectories.
 * The trade-off is 4× the derivative evaluations per step.
 *
 * Algorithm per body:
 *   k1 = f(t,       y)
 *   k2 = f(t+dt/2,  y + dt/2 * k1)
 *   k3 = f(t+dt/2,  y + dt/2 * k2)
 *   k4 = f(t+dt,    y + dt   * k3)
 *   y(t+dt) = y(t) + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
 *
 * Where y = (position, velocity) and f returns (velocity, acceleration).
 *
 * Use cases: orbital mechanics, spring systems requiring long-term stability,
 * any scenario where accuracy trumps raw throughput.
 *
 * Limitations: operates on RigidBodyStore with constant forces per step
 * (no mid-step force recomputation from the broader pipeline). For
 * position-dependent forces, the user should use smaller timesteps or
 * the Verlet integrator with force recomputation.
 */

#pragma once

#include "integration_common.h"
#include <pulse/rigidbody/rigid_body.h>
#include <pulse/rigidbody/rigid_body_common.h>
#include <pulse/math/mat3.h>
#include <pulse/math/quat.h>

namespace pulse {

namespace rk4_detail {

    /// Per-body derivative: rates of change for position and velocity.
    struct Derivative {
        Vec3 dx;    ///< Rate of change of position   = velocity.
        Vec3 dv;    ///< Rate of change of velocity    = linear acceleration.
        Vec3 dtheta;///< Rate of change of orientation = angular velocity.
        Vec3 dw;    ///< Rate of change of angular vel = angular acceleration.
    };

    /// Evaluate derivatives at the given offset from the current state.
    /// acceleration = force * invMass + gravity * gravityScale   (constant)
    /// angAcceleration = worldInvI * torque                      (constant)
    /// At offset: velocity = baseVelocity + dv * offsetDt
    ///            (position is not needed since forces are constant)
    static PULSE_FORCE_INLINE Derivative evaluate(
        Vec3 baseLinVel, Vec3 baseAngVel,
        Vec3 linAccel, Vec3 angAccel,
        const Derivative& d, float offsetDt) noexcept
    {
        Derivative out;
        // Velocity at offset = base + dv * offsetDt
        out.dx = baseLinVel + d.dv * offsetDt;
        out.dv = linAccel;   // Constant force model
        out.dtheta = baseAngVel + d.dw * offsetDt;
        out.dw = angAccel;   // Constant torque model
        return out;
    }

} // namespace rk4_detail

// ── RK4: Full Integration Step ───────────────────────────────────────────────

/**
 * @brief Full RK4 integration step for all bodies in the store.
 *
 * For each awake dynamic body, computes four derivative evaluations
 * and applies the weighted average to position, velocity, and rotation.
 *
 * @param store          Body data store.
 * @param gravity        World gravity vector (m/s²).
 * @param dt             Time step (seconds).
 * @param maxLinearSpeed  Speed clamp.  0 = no clamp.
 * @param maxAngularSpeed Speed clamp.  0 = no clamp.
 */
static inline void rk4Integrate(
    RigidBodyStore& store,
    Vec3 gravity,
    float dt,
    float maxLinearSpeed  = 500.0f,
    float maxAngularSpeed = 100.0f) noexcept
{
    using namespace rk4_detail;
    using namespace integration_detail;

    const std::size_t n = store.size();
    const float halfDt = dt * 0.5f;
    const float sixthDt = dt / 6.0f;

    for (std::size_t i = 0; i < n; ++i) {
        if (!store.isActive(i)) continue;

        // Current state
        Vec3 pos     = store.position(i);
        Quat rot     = store.rotation(i);
        Vec3 linVel  = store.linearVelocity(i);
        Vec3 angVel  = store.angularVelocity(i);

        // Constant acceleration (forces don't change within a single step)
        float invMass = store.invMass(i);
        Vec3 linAccel = (invMass > 0.0f) ? (store.force(i) * invMass + gravity * store.gravityScale(i)) : Vec3::zero();
        Vec3 angAccel = store.worldInvInertia(i) * store.torque(i);

        // ── k1: derivatives at t ──
        Derivative k1;
        k1.dx     = linVel;
        k1.dv     = linAccel;
        k1.dtheta = angVel;
        k1.dw     = angAccel;

        // ── k2: derivatives at t + dt/2, using k1 ──
        Derivative k2 = evaluate(linVel, angVel, linAccel, angAccel, k1, halfDt);

        // ── k3: derivatives at t + dt/2, using k2 ──
        Derivative k3 = evaluate(linVel, angVel, linAccel, angAccel, k2, halfDt);

        // ── k4: derivatives at t + dt, using k3 ──
        Derivative k4 = evaluate(linVel, angVel, linAccel, angAccel, k3, dt);

        // ── Weighted average ──
        // y(t+dt) = y(t) + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        // Position
        Vec3 dxAvg = (k1.dx + k2.dx * 2.0f + k3.dx * 2.0f + k4.dx) * sixthDt;
        Vec3 newPos = pos + dxAvg;

        // Linear velocity
        Vec3 dvAvg = (k1.dv + k2.dv * 2.0f + k3.dv * 2.0f + k4.dv) * sixthDt;
        Vec3 newLinVel = linVel + dvAvg;

        // Angular velocity
        Vec3 dwAvg = (k1.dw + k2.dw * 2.0f + k3.dw * 2.0f + k4.dw) * sixthDt;
        Vec3 newAngVel = angVel + dwAvg;

        // Rotation: integrate using the average angular velocity over the step
        Vec3 avgOmega = (k1.dtheta + k2.dtheta * 2.0f + k3.dtheta * 2.0f + k4.dtheta) * (1.0f / 6.0f);
        Quat newRot = integrateRotation(rot, avgOmega, dt);

        // Apply damping
        newLinVel = applyDamping(newLinVel, store.linearDamping(i), dt);
        newAngVel = applyDamping(newAngVel, store.angularDamping(i), dt);

        // Speed clamp
        newLinVel = clampSpeed(newLinVel, maxLinearSpeed);
        newAngVel = clampSpeed(newAngVel, maxAngularSpeed);

        // Write back
        store.setPosition(i, newPos);
        store.setRotation(i, newRot);
        store.linearVelocity(i)  = newLinVel;
        store.angularVelocity(i) = newAngVel;
    }
}

} // namespace pulse
