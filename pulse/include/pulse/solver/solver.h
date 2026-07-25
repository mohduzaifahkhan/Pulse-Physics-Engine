/**
 * @file solver.h
 * @brief Top-level solver facade — convenience API for a full solve pass.
 *
 * Wraps the ContactSolver into a single `solve()` function call that
 * handles the complete pipeline: initialise → warm-start → velocity
 * iterations → position iterations → store impulses.
 *
 * For advanced control (e.g., interleaving with custom constraints),
 * use ContactSolver directly.
 */

#pragma once

#include "contact_solver.h"
#include "solver_common.h"
#include <pulse/contact/contact_manifold_persistent.h>
#include <pulse/math/math_common.h>

#include <cstdint>

namespace pulse {

// ── Top-level solve function ─────────────────────────────────────────────────

/**
 * @brief Run a complete constraint solve pass.
 *
 * Pipeline:
 *   1. Initialise ContactSolver from manifolds
 *   2. Warm-start (if enabled)
 *   3. Run N velocity iterations
 *   4. Run M position iterations (with early-out if converged)
 *   5. Store accumulated impulses back to manifolds
 *
 * @param bodies         Array of solver bodies (velocities/positions are modified).
 * @param bodyCount      Number of bodies.
 * @param manifolds      Array of persistent manifolds.
 * @param manifoldCount  Number of manifolds.
 * @param config         Solver configuration.
 * @param dt             Time step (seconds).
 * @return Diagnostic statistics from the solve pass.
 */
inline SolverStats solve(SolverBody* bodies, uint32_t bodyCount,
                          PersistentManifold* manifolds, uint32_t manifoldCount,
                          const SolverConfig& config, float dt) noexcept
{
    SolverStats stats;
    stats.totalManifolds = manifoldCount;

    // Count total contacts and record pre-solve max penetration
    uint32_t totalContacts = 0;
    float maxPenBefore = 0.0f;
    for (uint32_t m = 0; m < manifoldCount; ++m) {
        totalContacts += manifolds[m].contactCount;
        float pen = manifolds[m].getMaxPenetration();
        if (pen > maxPenBefore) maxPenBefore = pen;
    }
    stats.totalContacts = totalContacts;
    stats.maxPenetrationBefore = maxPenBefore;

    if (totalContacts == 0) {
        stats.positionSolved = true;
        return stats;
    }

    // 1. Initialise solver
    ContactSolver solver;
    solver.initialize(bodies, bodyCount, manifolds, manifoldCount, config, dt);

    // 2. Warm-start
    if (config.warmStarting) {
        solver.warmStart(bodies);
    }

    // 3. Velocity iterations
    for (uint32_t i = 0; i < config.velocityIterations; ++i) {
        solver.solveVelocityConstraints(bodies);
    }
    stats.velocityIterationsUsed = config.velocityIterations;

    // 4. Position iterations (with early-out)
    stats.positionSolved = false;
    for (uint32_t i = 0; i < config.positionIterations; ++i) {
        stats.positionIterationsUsed = i + 1;
        if (solver.solvePositionConstraints(bodies, config)) {
            stats.positionSolved = true;
            break;
        }
    }

    // 5. Store impulses back for warm starting next frame
    solver.storeImpulses(manifolds, manifoldCount);

    // Record post-solve max penetration
    float maxPenAfter = 0.0f;
    for (uint32_t m = 0; m < manifoldCount; ++m) {
        float pen = manifolds[m].getMaxPenetration();
        if (pen > maxPenAfter) maxPenAfter = pen;
    }
    stats.maxPenetrationAfter = maxPenAfter;

    return stats;
}

// ── Multi-step helper ────────────────────────────────────────────────────────

/**
 * @brief Advance body positions using updated velocities (semi-implicit Euler).
 *
 * This is a convenience helper until Module 12 (Integration) is implemented.
 * Apply after solve() to update body positions from solved velocities.
 *
 * @param bodies     Array of solver bodies.
 * @param bodyCount  Number of bodies.
 * @param dt         Time step (seconds).
 */
inline void integratePositions(SolverBody* bodies, uint32_t bodyCount,
                                float dt) noexcept
{
    for (uint32_t i = 0; i < bodyCount; ++i) {
        if (bodies[i].isStatic()) continue;
        bodies[i].position += bodies[i].linearVelocity * dt;
        // Angular: simplified (no quaternion integration until Module 11)
    }
}

/**
 * @brief Apply gravity to all dynamic bodies.
 *
 * Convenience helper until Module 12 (Integration) is implemented.
 *
 * @param bodies     Array of solver bodies.
 * @param bodyCount  Number of bodies.
 * @param gravity    Gravity acceleration (e.g., Vec3(0, -9.81, 0)).
 * @param dt         Time step (seconds).
 */
inline void applyGravity(SolverBody* bodies, uint32_t bodyCount,
                          Vec3 gravity, float dt) noexcept
{
    for (uint32_t i = 0; i < bodyCount; ++i) {
        if (bodies[i].isStatic()) continue;
        bodies[i].linearVelocity += gravity * dt;
    }
}

} // namespace pulse
