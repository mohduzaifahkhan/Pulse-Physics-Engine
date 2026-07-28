/**
 * @file integrator.h
 * @brief Unified integration dispatch — selects integrator by IntegratorType.
 *
 * Provides a single entry point that dispatches to the correct integrator
 * based on IntegrationConfig::type. This is the public API module consumers
 * should prefer for flexibility.
 *
 * For maximum control (e.g., interleaving solver between velocity and position
 * phases), use the individual integrator headers directly.
 */

#pragma once

#include "integration_common.h"
#include "semi_implicit_euler.h"
#include "velocity_verlet.h"
#include "rk4.h"
#include <pulse/rigidbody/rigid_body.h>

namespace pulse {

// ── Unified velocity integration ─────────────────────────────────────────────

/**
 * @brief Integrate velocities from forces (dispatched by IntegratorType).
 *
 * For Euler: full velocity phase.
 * For Verlet: first half-kick (velocity += 0.5 * a * dt).
 * For RK4:   no-op (RK4 integrates velocity and position together).
 *
 * @param store  Body data store.
 * @param config Integration configuration.
 * @param dt     Time step (seconds).
 */
static inline void integrateVelocities(
    RigidBodyStore& store,
    const IntegrationConfig& config,
    float dt) noexcept
{
    switch (config.type) {
    case IntegratorType::SemiImplicitEuler:
        semiImplicitEulerIntegrateVelocities(
            store, config.gravity, dt,
            config.maxLinearSpeed, config.maxAngularSpeed);
        break;

    case IntegratorType::VelocityVerlet:
        // For Verlet, the "velocity integration" step is the first half-kick.
        // The drift is NOT included here — it happens in integratePositions().
        {
            using namespace integration_detail;
            const std::size_t n = store.size();
            const float halfDt = dt * 0.5f;
            for (std::size_t i = 0; i < n; ++i) {
                if (!store.isActive(i)) continue;
                float invMass = store.invMass(i);
                Vec3 linAcc = (invMass > 0.0f) ? (store.force(i) * invMass + config.gravity * store.gravityScale(i)) : Vec3::zero();
                store.linearVelocity(i) = store.linearVelocity(i) + linAcc * halfDt;
                const Mat3& worldInvI = store.worldInvInertia(i);
                Vec3 angAcc = worldInvI * store.torque(i);
                store.angularVelocity(i) = store.angularVelocity(i) + angAcc * halfDt;
            }
        }
        break;

    case IntegratorType::RK4:
        // RK4 does velocity and position in a single pass.
        // No-op here — the full step is done in integrate().
        break;
    }
}

// ── Unified position integration ─────────────────────────────────────────────

/**
 * @brief Integrate positions from velocities (dispatched by IntegratorType).
 *
 * For Euler: position phase (x += v * dt).
 * For Verlet: drift + second half-kick (with damping & clamping).
 * For RK4:   no-op (handled in integrate()).
 *
 * @param store  Body data store.
 * @param config Integration configuration.
 * @param dt     Time step (seconds).
 */
static inline void integratePositions(
    RigidBodyStore& store,
    const IntegrationConfig& config,
    float dt) noexcept
{
    switch (config.type) {
    case IntegratorType::SemiImplicitEuler:
        semiImplicitEulerIntegratePositions(store, dt);
        break;

    case IntegratorType::VelocityVerlet:
        // Drift: position += velocity * dt
        {
            using namespace integration_detail;
            const std::size_t n = store.size();
            for (std::size_t i = 0; i < n; ++i) {
                if (!store.isActive(i)) continue;
                Transform& tx = store.transform(i);
                tx.position = tx.position + store.linearVelocity(i) * dt;
                tx.rotation = integrateRotation(tx.rotation, store.angularVelocity(i), dt);
            }
        }
        // Second half-kick with damping
        verletSecondHalfKick(store, config.gravity, dt,
                             config.maxLinearSpeed, config.maxAngularSpeed);
        break;

    case IntegratorType::RK4:
        // No-op — RK4 is handled entirely in integrate().
        break;
    }
}

// ── Unified full integration ─────────────────────────────────────────────────

/**
 * @brief Full integration step (dispatched by IntegratorType).
 *
 * Combines velocity and position phases into a single call.
 * Use when there is no constraint solver between the phases.
 *
 * @param store  Body data store.
 * @param config Integration configuration.
 * @param dt     Time step (seconds).
 */
static inline void integrate(
    RigidBodyStore& store,
    const IntegrationConfig& config,
    float dt) noexcept
{
    switch (config.type) {
    case IntegratorType::SemiImplicitEuler:
        semiImplicitEulerIntegrate(store, config.gravity, dt,
                                   config.maxLinearSpeed, config.maxAngularSpeed);
        break;

    case IntegratorType::VelocityVerlet:
        verletIntegrate(store, config.gravity, dt,
                        config.maxLinearSpeed, config.maxAngularSpeed);
        break;

    case IntegratorType::RK4:
        rk4Integrate(store, config.gravity, dt,
                     config.maxLinearSpeed, config.maxAngularSpeed);
        break;
    }
}

} // namespace pulse
