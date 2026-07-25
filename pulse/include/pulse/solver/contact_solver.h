/**
 * @file contact_solver.h
 * @brief Sequential Impulse / PGS contact constraint solver.
 *
 * Implements the core iterative solver for contact constraints:
 *  1. initialize()  — Pre-compute effective masses, biases, warm-start data.
 *  2. warmStart()   — Apply cached impulses from the previous frame.
 *  3. solveVelocityConstraints() — One velocity iteration (call N times).
 *  4. solvePositionConstraints() — One position correction iteration.
 *
 * Velocity solver: Sequential Impulse with Projected Gauss-Seidel (PGS).
 *   • Normal impulse: λn ≥ 0  (non-penetration).
 *   • Tangent impulse: |λt| ≤ μ·λn  (Coulomb friction cone).
 *   • Accumulated clamping (not delta clamping) for stability.
 *
 * Position solver: Split-impulse — corrects penetration via pseudo-velocities
 * applied to position, not to real velocity.  This prevents velocity artifacts
 * (jitter/bounce) from position correction.
 *
 * Design: Allocates constraint arrays with new[]/delete[] to match the
 * existing codebase pattern (ContactCache does the same).  The ContactSolver
 * is a per-step transient — create, solve, destroy.
 */

#pragma once

#include "solver_common.h"
#include "constraint_base.h"
#include <pulse/contact/contact_manifold_persistent.h>
#include <pulse/contact/persistent_contact.h>
#include <pulse/math/vec3.h>
#include <pulse/math/math_common.h>

#include <cstdint>
#include <cstring>

namespace pulse {

/**
 * @class ContactSolver
 * @brief Iterative impulse-based contact constraint solver.
 *
 * Typical per-frame usage:
 * @code
 *   ContactSolver solver;
 *   solver.initialize(bodies, bodyCount, manifolds, manifoldCount, config, dt);
 *   solver.warmStart(bodies);
 *   for (uint32_t i = 0; i < config.velocityIterations; ++i)
 *       solver.solveVelocityConstraints(bodies);
 *   for (uint32_t i = 0; i < config.positionIterations; ++i)
 *       if (solver.solvePositionConstraints(bodies)) break;
 *   solver.storeImpulses(manifolds, manifoldCount);
 * @endcode
 */
class ContactSolver {
public:
    // ── Construction / destruction ────────────────────────────────────────

    ContactSolver() noexcept
        : velConstraints_(nullptr),
          posConstraints_(nullptr),
          groups_(nullptr),
          velCount_(0),
          posCount_(0),
          groupCount_(0),
          dt_(0.0f),
          invDt_(0.0f)
    {}

    ~ContactSolver() noexcept {
        delete[] velConstraints_;
        delete[] posConstraints_;
        delete[] groups_;
    }

    // Non-copyable
    ContactSolver(const ContactSolver&) = delete;
    ContactSolver& operator=(const ContactSolver&) = delete;

    // ── Initialization ───────────────────────────────────────────────────

    /**
     * @brief Pre-compute all constraint data from persistent manifolds.
     *
     * For each contact in each manifold:
     *  • Computes lever arms rA, rB
     *  • Computes effective mass K = invMassA + invMassB + (rA×n)²·invIA + (rB×n)²·invIB
     *  • Computes restitution velocity bias
     *  • Initialises accumulated impulses from warm-start data
     *
     * Body lookup: the solver matches PersistentManifold::bodyIdA/B against
     * SolverBody::bodyId to find the index into the bodies array.  O(M×B)
     * where M = manifold count, B = body count.  For large scenes, a hash
     * map should be used (deferred to Module 13 — World).
     *
     * @param bodies         Array of solver bodies.
     * @param bodyCount      Number of bodies.
     * @param manifolds      Array of persistent manifolds.
     * @param manifoldCount  Number of manifolds.
     * @param config         Solver configuration.
     * @param dt             Time step (seconds).
     */
    void initialize(const SolverBody* bodies, uint32_t bodyCount,
                    const PersistentManifold* manifolds, uint32_t manifoldCount,
                    const SolverConfig& config, float dt) noexcept
    {
        dt_    = dt;
        invDt_ = (dt > math::Epsilon) ? 1.0f / dt : 0.0f;

        // Count total contacts
        uint32_t totalContacts = 0;
        for (uint32_t m = 0; m < manifoldCount; ++m) {
            totalContacts += manifolds[m].contactCount;
        }

        // Allocate constraint arrays
        delete[] velConstraints_;
        delete[] posConstraints_;
        delete[] groups_;

        velCount_   = totalContacts;
        posCount_   = totalContacts;
        groupCount_ = manifoldCount;

        velConstraints_ = (totalContacts > 0) ? new VelocityConstraint[totalContacts] : nullptr;
        posConstraints_ = (totalContacts > 0) ? new PositionConstraint[totalContacts] : nullptr;
        groups_         = (manifoldCount > 0) ? new ContactConstraintGroup[manifoldCount] : nullptr;

        // Fill constraints
        uint32_t ci = 0; // Global contact index
        for (uint32_t m = 0; m < manifoldCount; ++m) {
            const PersistentManifold& pm = manifolds[m];
            if (pm.contactCount == 0) {
                groups_[m] = ContactConstraintGroup();
                continue;
            }

            // Find body indices
            uint32_t idxA = findBodyIndex(bodies, bodyCount, pm.bodyIdA);
            uint32_t idxB = findBodyIndex(bodies, bodyCount, pm.bodyIdB);

            // Build group
            groups_[m].header = ConstraintHeader(ConstraintType::Contact, pm.bodyIdA, pm.bodyIdB);
            groups_[m].startIndex = ci;
            groups_[m].count = pm.contactCount;

            // Material combination (geometric mean for restitution, pythagorean for friction)
            if (idxA < bodyCount && idxB < bodyCount) {
                groups_[m].friction = combineFriction(bodies[idxA].friction, bodies[idxB].friction);
                groups_[m].restitution = combineRestitution(bodies[idxA].restitution, bodies[idxB].restitution);
            }

            // Process each contact
            for (uint32_t c = 0; c < pm.contactCount; ++c) {
                const PersistentContact& pc = pm.contacts[c];

                VelocityConstraint& vc = velConstraints_[ci];
                PositionConstraint& psc = posConstraints_[ci];

                vc.indexA = idxA;
                vc.indexB = idxB;

                // Contact geometry
                vc.normal   = pc.normal;
                vc.tangent0 = pc.tangent0;
                vc.tangent1 = pc.tangent1;

                // Lever arms from body centres to contact points
                if (idxA < bodyCount) {
                    vc.rA = pc.positionOnA - bodies[idxA].position;
                }
                if (idxB < bodyCount) {
                    vc.rB = pc.positionOnB - bodies[idxB].position;
                }

                // Compute effective masses
                float invMassSum = 0.0f;
                if (idxA < bodyCount) invMassSum += bodies[idxA].invMass;
                if (idxB < bodyCount) invMassSum += bodies[idxB].invMass;

                vc.normalMass   = computeEffectiveMass(bodies, bodyCount, idxA, idxB, vc.rA, vc.rB, vc.normal);
                vc.tangentMass0 = computeEffectiveMass(bodies, bodyCount, idxA, idxB, vc.rA, vc.rB, vc.tangent0);
                vc.tangentMass1 = computeEffectiveMass(bodies, bodyCount, idxA, idxB, vc.rA, vc.rB, vc.tangent1);

                // Restitution bias
                vc.velocityBias = 0.0f;
                if (idxA < bodyCount && idxB < bodyCount) {
                    float closingVel = computeClosingVelocity(bodies[idxA], bodies[idxB],
                                                               vc.rA, vc.rB, vc.normal);
                    if (closingVel < -config.restitutionThreshold) {
                        vc.velocityBias = -groups_[m].restitution * closingVel;
                    }
                }

                // Friction
                vc.friction = groups_[m].friction;

                // Warm-start impulses
                if (config.warmStarting) {
                    vc.normalImpulse   = pc.normalImpulse;
                    vc.tangentImpulse0 = pc.tangentImpulse0;
                    vc.tangentImpulse1 = pc.tangentImpulse1;
                } else {
                    vc.normalImpulse   = 0.0f;
                    vc.tangentImpulse0 = 0.0f;
                    vc.tangentImpulse1 = 0.0f;
                }

                // Position constraint
                psc.indexA = idxA;
                psc.indexB = idxB;
                psc.rA     = vc.rA;
                psc.rB     = vc.rB;
                psc.normal = pc.normal;
                psc.penetration = pc.penetration;
                psc.normalMass  = vc.normalMass;

                ci++;
            }
        }
    }

    // ── Warm start ───────────────────────────────────────────────────────

    /**
     * @brief Apply cached impulses from the previous frame to body velocities.
     *
     * This gives the solver a head start — typically halves the number of
     * iterations needed for convergence.
     */
    void warmStart(SolverBody* bodies) const noexcept {
        for (uint32_t i = 0; i < velCount_; ++i) {
            const VelocityConstraint& vc = velConstraints_[i];

            // Combined impulse = normal + tangent contributions
            Vec3 impulse = vc.normal * vc.normalImpulse
                         + vc.tangent0 * vc.tangentImpulse0
                         + vc.tangent1 * vc.tangentImpulse1;

            // Apply to body A (pushed in +n direction = away from B)
            if (vc.indexA < 0xFFFFFFFFu) {
                SolverBody& bodyA = bodies[vc.indexA];
                bodyA.linearVelocity  += impulse * bodyA.invMass;
                bodyA.angularVelocity += bodyA.invInertia * vc.rA.cross(impulse);
            }

            // Apply to body B (pushed in -n direction = away from A)
            if (vc.indexB < 0xFFFFFFFFu) {
                SolverBody& bodyB = bodies[vc.indexB];
                bodyB.linearVelocity  -= impulse * bodyB.invMass;
                bodyB.angularVelocity -= bodyB.invInertia * vc.rB.cross(impulse);
            }
        }
    }

    // ── Velocity solver ──────────────────────────────────────────────────

    /**
     * @brief Run one iteration of the sequential-impulse velocity solver.
     *
     * For each contact:
     *  1. Compute relative velocity at the contact point.
     *  2. Normal constraint: λn = -K_n * (v_rel·n + bias), clamped λn ≥ 0.
     *  3. Tangent constraints: λt = -K_t * (v_rel·t), clamped |λt| ≤ μ·λn.
     *  4. Apply impulse deltas to both bodies.
     *
     * Uses *accumulated clamping*: clamp the total accumulated impulse,
     * then apply only the delta.  This is more stable than clamping the
     * per-iteration delta.
     */
    void solveVelocityConstraints(SolverBody* bodies) noexcept {
        for (uint32_t i = 0; i < velCount_; ++i) {
            VelocityConstraint& vc = velConstraints_[i];

            SolverBody& bodyA = bodies[vc.indexA];
            SolverBody& bodyB = bodies[vc.indexB];

            // ── Relative velocity at contact point ──────────────────
            // Convention: vRel = vA - vB.  With normal from B→A,
            // vn < 0 means approaching (need impulse).
            Vec3 vA = bodyA.linearVelocity + bodyA.angularVelocity.cross(vc.rA);
            Vec3 vB = bodyB.linearVelocity + bodyB.angularVelocity.cross(vc.rB);
            Vec3 vRel = vA - vB;

            // ── Normal constraint ───────────────────────────────────
            {
                float vn = vRel.dot(vc.normal);
                float lambda = -vc.normalMass * (vn - vc.velocityBias);

                // Accumulated clamping: total impulse ≥ 0
                float oldImpulse = vc.normalImpulse;
                vc.normalImpulse = math::fastMax(oldImpulse + lambda, 0.0f);
                float delta = vc.normalImpulse - oldImpulse;

                Vec3 impulse = vc.normal * delta;
                bodyA.linearVelocity  += impulse * bodyA.invMass;
                bodyA.angularVelocity += bodyA.invInertia * vc.rA.cross(impulse);
                bodyB.linearVelocity  -= impulse * bodyB.invMass;
                bodyB.angularVelocity -= bodyB.invInertia * vc.rB.cross(impulse);
            }

            // ── Friction tangent 0 ──────────────────────────────────
            {
                // Recompute relative velocity after normal impulse
                vA = bodyA.linearVelocity + bodyA.angularVelocity.cross(vc.rA);
                vB = bodyB.linearVelocity + bodyB.angularVelocity.cross(vc.rB);
                vRel = vA - vB;

                float vt = vRel.dot(vc.tangent0);
                float lambda = -vc.tangentMass0 * vt;

                // Coulomb friction cone: |λt| ≤ μ·λn
                float maxFriction = vc.friction * vc.normalImpulse;
                float oldImpulse = vc.tangentImpulse0;
                vc.tangentImpulse0 = math::clamp(oldImpulse + lambda, -maxFriction, maxFriction);
                float delta = vc.tangentImpulse0 - oldImpulse;

                Vec3 impulse = vc.tangent0 * delta;
                bodyA.linearVelocity  += impulse * bodyA.invMass;
                bodyA.angularVelocity += bodyA.invInertia * vc.rA.cross(impulse);
                bodyB.linearVelocity  -= impulse * bodyB.invMass;
                bodyB.angularVelocity -= bodyB.invInertia * vc.rB.cross(impulse);
            }

            // ── Friction tangent 1 ──────────────────────────────────
            {
                vA = bodyA.linearVelocity + bodyA.angularVelocity.cross(vc.rA);
                vB = bodyB.linearVelocity + bodyB.angularVelocity.cross(vc.rB);
                vRel = vA - vB;

                float vt = vRel.dot(vc.tangent1);
                float lambda = -vc.tangentMass1 * vt;

                float maxFriction = vc.friction * vc.normalImpulse;
                float oldImpulse = vc.tangentImpulse1;
                vc.tangentImpulse1 = math::clamp(oldImpulse + lambda, -maxFriction, maxFriction);
                float delta = vc.tangentImpulse1 - oldImpulse;

                Vec3 impulse = vc.tangent1 * delta;
                bodyA.linearVelocity  += impulse * bodyA.invMass;
                bodyA.angularVelocity += bodyA.invInertia * vc.rA.cross(impulse);
                bodyB.linearVelocity  -= impulse * bodyB.invMass;
                bodyB.angularVelocity -= bodyB.invInertia * vc.rB.cross(impulse);
            }
        }
    }

    // ── Position solver ──────────────────────────────────────────────────

    /**
     * @brief Run one iteration of split-impulse position correction.
     *
     * Directly adjusts body positions (not velocities) to resolve
     * remaining penetration.  Returns true if all contacts are within
     * tolerance (early-out for subsequent iterations).
     *
     * Split impulse: position correction is *separated* from velocity
     * correction.  This prevents the position fix from introducing
     * spurious velocity (the classic Baumgarte "buzz" artifact).
     *
     * @param bodies  Array of solver bodies (positions are modified).
     * @param config  Solver configuration (slop, baumgarte, maxPositionCorrection).
     * @return true if all penetrations are resolved within slop.
     */
    bool solvePositionConstraints(SolverBody* bodies,
                                  const SolverConfig& config) const noexcept
    {
        float maxCorrection = 0.0f;

        for (uint32_t i = 0; i < posCount_; ++i) {
            const PositionConstraint& pc = posConstraints_[i];

            SolverBody& bodyA = bodies[pc.indexA];
            SolverBody& bodyB = bodies[pc.indexB];

            // Current separation (recompute from updated positions)
            Vec3 pA = bodyA.position + pc.rA;
            Vec3 pB = bodyB.position + pc.rB;
            float separation = (pB - pA).dot(pc.normal) + pc.penetration;

            // Position error (positive = penetrating beyond slop)
            float C = math::clamp(config.baumgarte * (separation - config.slop),
                                  -config.maxPositionCorrection, 0.0f);

            // Only correct if penetrating
            if (C >= -math::Epsilon) continue;

            float impulse = -pc.normalMass * C;
            maxCorrection = math::fastMax(maxCorrection, math::fastAbs(C));

            // Apply position correction (A pushed in +n = away from B)
            Vec3 correction = pc.normal * impulse;
            bodyA.position += correction * bodyA.invMass;
            bodyB.position -= correction * bodyB.invMass;
        }

        return maxCorrection < config.slop;
    }

    // ── Impulse storage ──────────────────────────────────────────────────

    /**
     * @brief Write accumulated impulses back to the persistent manifolds.
     *
     * This closes the warm-starting loop: the solver writes its final
     * impulses into the manifold contacts, which the ContactCache
     * persists into the next frame.
     */
    void storeImpulses(PersistentManifold* manifolds, uint32_t manifoldCount) const noexcept {
        uint32_t ci = 0;
        for (uint32_t m = 0; m < manifoldCount; ++m) {
            PersistentManifold& pm = manifolds[m];
            for (uint32_t c = 0; c < pm.contactCount; ++c) {
                if (ci < velCount_) {
                    const VelocityConstraint& vc = velConstraints_[ci];
                    pm.contacts[c].normalImpulse   = vc.normalImpulse;
                    pm.contacts[c].tangentImpulse0 = vc.tangentImpulse0;
                    pm.contacts[c].tangentImpulse1 = vc.tangentImpulse1;
                    pm.contacts[c].flags |= ContactFlags::HasImpulse;
                }
                ci++;
            }
        }
    }

    // ── Accessors ────────────────────────────────────────────────────────

    /// Number of velocity constraints (total contacts).
    [[nodiscard]] uint32_t velocityConstraintCount() const noexcept { return velCount_; }

    /// Number of position constraints (total contacts).
    [[nodiscard]] uint32_t positionConstraintCount() const noexcept { return posCount_; }

    /// Number of constraint groups (manifolds).
    [[nodiscard]] uint32_t groupCount() const noexcept { return groupCount_; }

    /// Access a velocity constraint by index (for testing/diagnostics).
    [[nodiscard]] const VelocityConstraint& velocityConstraint(uint32_t i) const noexcept {
        return velConstraints_[i];
    }

    /// Access a position constraint by index (for testing/diagnostics).
    [[nodiscard]] const PositionConstraint& positionConstraint(uint32_t i) const noexcept {
        return posConstraints_[i];
    }

private:
    VelocityConstraint*      velConstraints_;  ///< Velocity constraint array.
    PositionConstraint*      posConstraints_;  ///< Position constraint array.
    ContactConstraintGroup*  groups_;           ///< Per-manifold groups.
    uint32_t                 velCount_;         ///< Total velocity constraints.
    uint32_t                 posCount_;         ///< Total position constraints.
    uint32_t                 groupCount_;       ///< Number of groups (manifolds).
    float                    dt_;               ///< Time step.
    float                    invDt_;            ///< Inverse time step.

    // ── Helpers ──────────────────────────────────────────────────────────

    /// Find the index of a body by its ID.  O(N) linear scan.
    [[nodiscard]] static PULSE_FORCE_INLINE uint32_t findBodyIndex(
        const SolverBody* bodies, uint32_t count, uint32_t bodyId) noexcept
    {
        for (uint32_t i = 0; i < count; ++i) {
            if (bodies[i].bodyId == bodyId) return i;
        }
        return 0xFFFFFFFFu; // Not found
    }

    /// Compute effective mass for a constraint direction.
    ///
    /// K = invMassA + invMassB + (rA × dir)² · invIA + (rB × dir)² · invIB
    /// effectiveMass = 1 / K
    [[nodiscard]] static PULSE_FORCE_INLINE float computeEffectiveMass(
        const SolverBody* bodies, uint32_t bodyCount,
        uint32_t idxA, uint32_t idxB,
        Vec3 rA, Vec3 rB, Vec3 dir) noexcept
    {
        float K = 0.0f;

        if (idxA < bodyCount) {
            K += bodies[idxA].invMass;
            Vec3 rAxDir = rA.cross(dir);
            // Diagonal inertia: K += (rA×dir)·(invI ⊙ (rA×dir))
            K += (rAxDir * bodies[idxA].invInertia).dot(rAxDir);
        }

        if (idxB < bodyCount) {
            K += bodies[idxB].invMass;
            Vec3 rBxDir = rB.cross(dir);
            K += (rBxDir * bodies[idxB].invInertia).dot(rBxDir);
        }

        return (K > math::Epsilon) ? 1.0f / K : 0.0f;
    }

    /// Compute separation velocity along the contact normal.
    /// vn = (vA - vB) · n.  Negative = approaching, positive = separating.
    [[nodiscard]] static PULSE_FORCE_INLINE float computeClosingVelocity(
        const SolverBody& bodyA, const SolverBody& bodyB,
        Vec3 rA, Vec3 rB, Vec3 normal) noexcept
    {
        Vec3 vA = bodyA.linearVelocity + bodyA.angularVelocity.cross(rA);
        Vec3 vB = bodyB.linearVelocity + bodyB.angularVelocity.cross(rB);
        return (vA - vB).dot(normal);
    }

    /// Combine friction (geometric mean).
    [[nodiscard]] static PULSE_FORCE_INLINE float combineFriction(float a, float b) noexcept {
        return math::fastSqrt(a * b);
    }

    /// Combine restitution (maximum).
    [[nodiscard]] static PULSE_FORCE_INLINE float combineRestitution(float a, float b) noexcept {
        return math::fastMax(a, b);
    }
};

} // namespace pulse
