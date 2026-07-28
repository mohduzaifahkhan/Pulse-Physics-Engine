/**
 * @file semi_implicit_euler.h
 * @brief Semi-Implicit (Symplectic) Euler integrator — the default for real-time physics.
 *
 * The semi-implicit Euler method updates velocity FIRST from forces, then
 * updates position from the NEW velocity. This makes it symplectic: it
 * preserves phase-space volume and avoids the energy gain of explicit Euler.
 *
 *   v(t+dt) = v(t) + a(t) * dt           // velocity from acceleration
 *   x(t+dt) = x(t) + v(t+dt) * dt        // position from NEW velocity
 *
 * This is the integrator used by Box2D, Bullet, and PhysX for game physics.
 * First-order accurate but extremely stable for typical game timesteps.
 *
 * Functions:
 *  - semiImplicitEulerIntegrateVelocities: velocity phase (before solver)
 *  - semiImplicitEulerIntegratePositions:  position phase (after solver)
 *  - semiImplicitEulerIntegrate:           both phases combined
 *
 * All functions operate on RigidBodyStore SoA arrays. Static, kinematic,
 * and sleeping bodies are automatically skipped.
 */

#pragma once

#include "integration_common.h"
#include <pulse/rigidbody/rigid_body.h>
#include <pulse/rigidbody/rigid_body_common.h>
#include <pulse/math/mat3.h>
#include <pulse/math/quat.h>

namespace pulse {

// ── Semi-Implicit Euler: Velocity Phase ──────────────────────────────────────

/**
 * @brief Integrate velocities from forces and gravity (Semi-Implicit Euler).
 *
 * For each awake dynamic body:
 *   linVel += (force * invMass + gravity * gravityScale) * dt
 *   angVel += worldInvInertia * torque * dt
 *   apply damping
 *   clamp speeds
 *
 * Call BEFORE the constraint solver so the solver works on pre-integrated
 * velocities.
 *
 * @param store   Body data store (read/write velocities, read forces).
 * @param gravity World gravity vector (m/s²).
 * @param dt      Time step (seconds).
 * @param maxLinearSpeed  Speed clamp for linear velocity.  0 = no clamp.
 * @param maxAngularSpeed Speed clamp for angular velocity. 0 = no clamp.
 */
static inline void semiImplicitEulerIntegrateVelocities(
    RigidBodyStore& store,
    Vec3 gravity,
    float dt,
    float maxLinearSpeed  = 500.0f,
    float maxAngularSpeed = 100.0f) noexcept
{
    using namespace integration_detail;

    const std::size_t n = store.size();

    for (std::size_t i = 0; i < n; ++i) {
        // Skip non-dynamic bodies (static, kinematic) and sleeping bodies.
        if (!store.isActive(i)) continue;

        // ── Linear velocity ──
        float invMass = store.invMass(i);
        Vec3 linAcc = (invMass > 0.0f) ? (store.force(i) * invMass + gravity * store.gravityScale(i)) : Vec3::zero();
        Vec3 linVel = store.linearVelocity(i) + linAcc * dt;

        // Damping: v *= 1 / (1 + damping * dt)
        linVel = applyDamping(linVel, store.linearDamping(i), dt);

        // Speed clamp
        linVel = clampSpeed(linVel, maxLinearSpeed);

        store.linearVelocity(i) = linVel;

        // ── Angular velocity ──
        const Mat3& worldInvI = store.worldInvInertia(i);
        Vec3 angAcc = worldInvI * store.torque(i);
        Vec3 angVel = store.angularVelocity(i) + angAcc * dt;

        // Damping
        angVel = applyDamping(angVel, store.angularDamping(i), dt);

        // Speed clamp
        angVel = clampSpeed(angVel, maxAngularSpeed);

        store.angularVelocity(i) = angVel;
    }
}

// ── Semi-Implicit Euler: Position Phase ──────────────────────────────────────

/**
 * @brief Integrate positions from velocities (Semi-Implicit Euler).
 *
 * For each awake dynamic body:
 *   position += linearVelocity * dt
 *   rotation  = normalize(rotation + 0.5 * dt * Quat(angVel, 0) * rotation)
 *
 * Call AFTER the constraint solver so positions reflect the corrected
 * velocities.
 *
 * @param store  Body data store (read velocities, write transforms).
 * @param dt     Time step (seconds).
 */
static inline void semiImplicitEulerIntegratePositions(
    RigidBodyStore& store,
    float dt) noexcept
{
    using namespace integration_detail;

    const std::size_t n = store.size();

    for (std::size_t i = 0; i < n; ++i) {
        if (!store.isActive(i)) continue;

        Transform& tx = store.transform(i);

        // Linear: position += velocity * dt
        tx.position = tx.position + store.linearVelocity(i) * dt;

        // Angular: quaternion integration
        tx.rotation = integrateRotation(tx.rotation, store.angularVelocity(i), dt);
    }
}

// ── Semi-Implicit Euler: Combined Step ───────────────────────────────────────

/**
 * @brief Full Semi-Implicit Euler step — velocity then position.
 *
 * Convenience function that calls both phases. Use when there is no
 * constraint solver between the velocity and position phases.
 *
 * @param store   Body data store.
 * @param gravity World gravity vector.
 * @param dt      Time step.
 * @param maxLinearSpeed  Speed clamp for linear velocity.
 * @param maxAngularSpeed Speed clamp for angular velocity.
 */
static inline void semiImplicitEulerIntegrate(
    RigidBodyStore& store,
    Vec3 gravity,
    float dt,
    float maxLinearSpeed  = 500.0f,
    float maxAngularSpeed = 100.0f) noexcept
{
    semiImplicitEulerIntegrateVelocities(store, gravity, dt, maxLinearSpeed, maxAngularSpeed);
    semiImplicitEulerIntegratePositions(store, dt);
}

} // namespace pulse
