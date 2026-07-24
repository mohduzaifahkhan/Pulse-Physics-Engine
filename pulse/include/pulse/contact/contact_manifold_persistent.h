/**
 * @file contact_manifold_persistent.h
 * @brief Frame-persistent contact manifold with matching and pruning.
 *
 * Extends the single-frame ContactManifold concept into a persistent
 * structure that lives across physics frames. When new narrow-phase
 * contacts arrive, they are matched against existing contacts (by
 * feature ID first, then by position proximity) so that accumulated
 * solver impulses are preserved for warm starting.
 *
 * Design: Up to 4 PersistentContacts per pair. When more arrive, the
 * narrowphase ContactManifold has already been reduced to 4. This
 * manifold's job is to merge those 4 (or fewer) new contacts against
 * the previous frame's contacts, carrying impulses forward.
 */

#pragma once

#include "persistent_contact.h"
#include "contact_common.h"
#include <pulse/narrowphase/narrowphase_common.h>
#include <pulse/math/vec3.h>
#include <pulse/math/math_common.h>
#include <pulse/shapes/shape_common.h>

#include <cstdint>

namespace pulse {

/**
 * @struct PersistentManifold
 * @brief A set of up to 4 persistent contacts between a body pair.
 *
 * Manages the lifecycle of contacts across frames:
 * 1. Receives new contacts from the narrow phase.
 * 2. Matches them to existing contacts (preserving impulses).
 * 3. Prunes contacts that have separated.
 * 4. Prepares impulses for warm starting the solver.
 */
struct PersistentManifold {
    static constexpr uint32_t MaxContacts = 4;

    // ── Body identification ──────────────────────────────────────────────

    uint32_t  bodyIdA;      ///< Body A identifier.
    uint32_t  bodyIdB;      ///< Body B identifier.
    ShapeType shapeTypeA;   ///< Shape type of body A.
    ShapeType shapeTypeB;   ///< Shape type of body B.

    // ── Contact storage ──────────────────────────────────────────────────

    PersistentContact contacts[MaxContacts]; ///< Persistent contact points.
    uint32_t contactCount;                    ///< Number of valid contacts.

    // ── Lifecycle ────────────────────────────────────────────────────────

    uint32_t framesActive;   ///< How many frames this manifold has existed.
    bool     refreshedThisFrame; ///< True if narrow-phase produced contacts this frame.
    uint8_t  _pad[3];        ///< Padding.

    // ── Constructors ─────────────────────────────────────────────────────

    /// Default: empty manifold.
    PULSE_FORCE_INLINE PersistentManifold() noexcept
        : bodyIdA(0xFFFFFFFFu),
          bodyIdB(0xFFFFFFFFu),
          shapeTypeA(ShapeType::Sphere),
          shapeTypeB(ShapeType::Sphere),
          contactCount(0),
          framesActive(0),
          refreshedThisFrame(false)
    {
        _pad[0] = _pad[1] = _pad[2] = 0;
    }

    /// Construct for a specific body pair.
    PULSE_FORCE_INLINE PersistentManifold(uint32_t idA, uint32_t idB) noexcept
        : bodyIdA(idA),
          bodyIdB(idB),
          shapeTypeA(ShapeType::Sphere),
          shapeTypeB(ShapeType::Sphere),
          contactCount(0),
          framesActive(0),
          refreshedThisFrame(false)
    {
        _pad[0] = _pad[1] = _pad[2] = 0;
    }

    // ── Query ────────────────────────────────────────────────────────────

    /// Get the body pair key for this manifold.
    [[nodiscard]] PULSE_FORCE_INLINE BodyPairKey getKey() const noexcept {
        return BodyPairKey(bodyIdA, bodyIdB);
    }

    /// Is this manifold empty?
    [[nodiscard]] PULSE_FORCE_INLINE bool isEmpty() const noexcept {
        return contactCount == 0;
    }

    /// Get the deepest penetration.
    [[nodiscard]] PULSE_FORCE_INLINE float getMaxPenetration() const noexcept {
        float maxPen = 0.0f;
        for (uint32_t i = 0; i < contactCount; ++i) {
            if (contacts[i].penetration > maxPen)
                maxPen = contacts[i].penetration;
        }
        return maxPen;
    }

    /// Clear all contacts.
    PULSE_FORCE_INLINE void clear() noexcept {
        contactCount = 0;
    }

    // ── Contact matching and merging ─────────────────────────────────────

    /**
     * @brief Merge new narrow-phase contacts into this persistent manifold.
     *
     * For each new contact, attempts to match it against existing contacts:
     * 1. First by feature ID pair (exact match).
     * 2. Then by position proximity (within matchDistSq).
     *
     * Matched contacts preserve their accumulated impulses; unmatched new
     * contacts are added with zero impulses. Old contacts not matched by
     * any new contact are removed.
     *
     * @param newManifold  Fresh contacts from the narrow phase.
     * @param matchDistSq  Squared distance threshold for proximity matching.
     */
    PULSE_FORCE_INLINE void mergeContacts(const ContactManifold& newManifold,
                                          float matchDistSq) noexcept
    {
        // Save old contacts
        PersistentContact oldContacts[MaxContacts];
        uint32_t oldCount = contactCount;
        for (uint32_t i = 0; i < oldCount; ++i) {
            oldContacts[i] = contacts[i];
        }

        // Track which old contacts have been matched
        bool oldMatched[MaxContacts] = { false, false, false, false };

        // Reset contact count — we'll rebuild
        contactCount = 0;
        shapeTypeA = newManifold.shapeTypeA;
        shapeTypeB = newManifold.shapeTypeB;

        // Process each new contact
        for (uint32_t n = 0; n < newManifold.numContacts && contactCount < MaxContacts; ++n) {
            const ContactPoint& newCp = newManifold.points[n];
            int32_t bestMatch = -1;

            // Strategy 1: Match by feature ID
            if (newCp.featureIdA != 0xFFFFFFFFu && newCp.featureIdB != 0xFFFFFFFFu) {
                for (uint32_t o = 0; o < oldCount; ++o) {
                    if (oldMatched[o]) continue;
                    if (oldContacts[o].featureIdA == newCp.featureIdA &&
                        oldContacts[o].featureIdB == newCp.featureIdB) {
                        bestMatch = static_cast<int32_t>(o);
                        break;
                    }
                }
            }

            // Strategy 2: Match by position proximity
            if (bestMatch < 0) {
                float bestDistSq = matchDistSq;
                for (uint32_t o = 0; o < oldCount; ++o) {
                    if (oldMatched[o]) continue;
                    Vec3 diff = newCp.positionOnA - oldContacts[o].positionOnA;
                    float dSq = diff.lengthSq();
                    if (dSq < bestDistSq) {
                        bestDistSq = dSq;
                        bestMatch = static_cast<int32_t>(o);
                    }
                }
            }

            // Create the persistent contact
            if (bestMatch >= 0) {
                // Matched: preserve impulses, update geometry
                PersistentContact& pc = contacts[contactCount++];
                pc = oldContacts[bestMatch];
                pc.updateGeometry(newCp);
                pc.flags = ContactFlags::Persisted | ContactFlags::HasImpulse;
                pc.contactAge++;
                oldMatched[bestMatch] = true;
            } else {
                // New contact: zero impulses
                PersistentContact& pc = contacts[contactCount++];
                pc = PersistentContact(newCp);
                pc.flags = ContactFlags::New;
                pc.contactAge = 0;
            }
        }

        refreshedThisFrame = true;
        framesActive++;
    }

    // ── Pruning ──────────────────────────────────────────────────────────

    /**
     * @brief Remove contacts that have separated beyond the break distance.
     *
     * Checks both the penetration depth (should be positive) and the
     * tangential drift of the contact point. A contact is pruned if its
     * penetration is more negative than -breakDistance, or if the contact
     * point has moved too far from the original position.
     *
     * @param breakDistance  Distance threshold for contact breaking (positive).
     */
    PULSE_FORCE_INLINE void pruneStale(float breakDistance) noexcept {
        uint32_t writeIdx = 0;
        for (uint32_t i = 0; i < contactCount; ++i) {
            const PersistentContact& c = contacts[i];

            // Check if contact has separated
            if (c.penetration < -breakDistance) {
                continue; // Discard
            }

            // Check tangential drift: project the separation vector onto
            // the contact normal and check the perpendicular distance
            Vec3 rA = c.positionOnA - c.positionOnB;
            float normalSep = rA.dot(c.normal);
            Vec3 tangentialDrift = rA - c.normal * normalSep;
            float tangentialDistSq = tangentialDrift.lengthSq();

            // If tangential drift exceeds breakDistance squared, discard
            if (tangentialDistSq > breakDistance * breakDistance * 4.0f) {
                continue; // Discard
            }

            // Keep this contact
            if (writeIdx != i) {
                contacts[writeIdx] = contacts[i];
            }
            writeIdx++;
        }
        contactCount = writeIdx;
    }

    // ── Warm-start preparation ───────────────────────────────────────────

    /**
     * @brief Scale all accumulated impulses in this manifold for warm starting.
     *
     * @param factor  Scaling factor (0.0 = disable, 1.0 = full warm start).
     * @param newContactDamping  Extra damping for newly created contacts.
     */
    PULSE_FORCE_INLINE void prepareWarmStart(float factor, float newContactDamping) noexcept {
        for (uint32_t i = 0; i < contactCount; ++i) {
            PersistentContact& c = contacts[i];
            if (hasFlag(c.flags, ContactFlags::New)) {
                // New contacts get damped warm starting
                c.scaleImpulses(newContactDamping);
            } else {
                // Established contacts get full warm starting
                c.scaleImpulses(factor);
            }
            // Mark that we have impulse data
            c.flags |= ContactFlags::HasImpulse;
        }
    }
};

} // namespace pulse
