/**
 * @file velocity_verlet.h
 * @brief Velocity Verlet (Kick-Drift-Kick) integrator — second-order accurate.
 *
 * The Velocity Verlet method splits the velocity update into two half-steps
 * around the position update, achieving second-order accuracy and superior
 * energy conservation compared to semi-implicit Euler.
 *
 * Full step (when force is constant):
 *   v(t+dt/2)  = v(t)     + 0.5 * a(t)     * dt   // first half-kick
 *   x(t+dt)    = x(t)     + v(t+dt/2) * dt         // drift
 *   v(t+dt)    = v(t+dt/2) + 0.5 * a(t+dt)  * dt   // second half-kick
 *
 * The split API allows the caller to recompute forces between drift and
 * the second half-kick for position-dependent forces:
 *   1. verletHalfKickAndDrift()    // first half-kick + drift
 *   2. (recompute forces)
 *   3. verletSecondHalfKick()      // second half-kick + damping
 *
 * For constant forces, use verletIntegrate() which combines all phases.
 *
 * Popular in molecular dynamics (LAMMPS, GROMACS) and N-body simulations.
 * All functions operate on RigidBodyStore SoA arrays.
 */

#pragma once

#include "integration_common.h"
#include <pulse/rigidbody/rigid_body.h>
#include <pulse/rigidbody/rigid_body_common.h>
#include <pulse/math/mat3.h>
#include <pulse/math/quat.h>

namespace pulse {

// ── Velocity Verlet: First Half-Kick + Drift ─────────────────────────────────

/**
 * @brief First half-kick velocity update, then drift position update.
 *
 * For each awake dynamic body:
 *   linVel += 0.5 * (force * invMass + gravity * gravityScale) * dt
 *   angVel += 0.5 * (worldInvInertia * torque) * dt
 *   position += linVel * dt
 *   rotation += quaternion integration of angVel * dt
 *
 * After calling, recompute forces (gravity is constant, but contact/spring
 * forces may change with position), then call verletSecondHalfKick().
 *
 * @param store   Body data store.
 * @param gravity World gravity vector (m/s²).
 * @param dt      Time step (seconds).
 */
static inline void verletHalfKickAndDrift(
    RigidBodyStore& store,
    Vec3 gravity,
    float dt) noexcept
{
    using namespace integration_detail;

    const std::size_t n = store.size();
    const float halfDt = dt * 0.5f;

    for (std::size_t i = 0; i < n; ++i) {
        if (!store.isActive(i)) continue;

        // ── Half-kick: velocity += 0.5 * acceleration * dt ──
        float invMass = store.invMass(i);
        Vec3 linAcc = (invMass > 0.0f) ? (store.force(i) * invMass + gravity * store.gravityScale(i)) : Vec3::zero();
        Vec3 linVel = store.linearVelocity(i) + linAcc * halfDt;

        const Mat3& worldInvI = store.worldInvInertia(i);
        Vec3 angAcc = worldInvI * store.torque(i);
        Vec3 angVel = store.angularVelocity(i) + angAcc * halfDt;

        store.linearVelocity(i) = linVel;
        store.angularVelocity(i) = angVel;

        // ── Drift: position += velocity * dt ──
        Transform& tx = store.transform(i);
        tx.position = tx.position + linVel * dt;
        tx.rotation = integrateRotation(tx.rotation, angVel, dt);
    }
}

// ── Velocity Verlet: Second Half-Kick ────────────────────────────────────────

/**
 * @brief Second half-kick velocity update with damping and clamping.
 *
 * Call AFTER forces have been recomputed for the new positions.
 *
 * For each awake dynamic body:
 *   linVel += 0.5 * (newForce * invMass + gravity * gravityScale) * dt
 *   angVel += 0.5 * (worldInvInertia * newTorque) * dt
 *   apply damping
 *   clamp speeds
 *
 * @param store          Body data store.
 * @param gravity        World gravity vector (m/s²).
 * @param dt             Time step (seconds).
 * @param maxLinearSpeed  Speed clamp.  0 = no clamp.
 * @param maxAngularSpeed Speed clamp.  0 = no clamp.
 */
static inline void verletSecondHalfKick(
    RigidBodyStore& store,
    Vec3 gravity,
    float dt,
    float maxLinearSpeed  = 500.0f,
    float maxAngularSpeed = 100.0f) noexcept
{
    using namespace integration_detail;

    const std::size_t n = store.size();
    const float halfDt = dt * 0.5f;

    for (std::size_t i = 0; i < n; ++i) {
        if (!store.isActive(i)) continue;

        // ── Second half-kick ──
        float invMass = store.invMass(i);
        Vec3 linAcc = store.force(i) * invMass + gravity * store.gravityScale(i);
        Vec3 linVel = store.linearVelocity(i) + linAcc * halfDt;

        const Mat3& worldInvI = store.worldInvInertia(i);
        Vec3 angAcc = worldInvI * store.torque(i);
        Vec3 angVel = store.angularVelocity(i) + angAcc * halfDt;

        // Damping
        linVel = applyDamping(linVel, store.linearDamping(i), dt);
        angVel = applyDamping(angVel, store.angularDamping(i), dt);

        // Speed clamp
        linVel = clampSpeed(linVel, maxLinearSpeed);
        angVel = clampSpeed(angVel, maxAngularSpeed);

        store.linearVelocity(i) = linVel;
        store.angularVelocity(i) = angVel;
    }
}

// ── Velocity Verlet: Combined Step ───────────────────────────────────────────

/**
 * @brief Full Velocity Verlet step (constant-force approximation).
 *
 * Combines half-kick + drift + second-half-kick using the SAME forces
 * for both half-kicks. This is valid when forces do not depend on position
 * (gravity, external impulses) or when the timestep is small enough that
 * force variation is negligible.
 *
 * @param store          Body data store.
 * @param gravity        World gravity vector.
 * @param dt             Time step.
 * @param maxLinearSpeed  Speed clamp.
 * @param maxAngularSpeed Speed clamp.
 */
static inline void verletIntegrate(
    RigidBodyStore& store,
    Vec3 gravity,
    float dt,
    float maxLinearSpeed  = 500.0f,
    float maxAngularSpeed = 100.0f) noexcept
{
    verletHalfKickAndDrift(store, gravity, dt);
    verletSecondHalfKick(store, gravity, dt, maxLinearSpeed, maxAngularSpeed);
}

} // namespace pulse
