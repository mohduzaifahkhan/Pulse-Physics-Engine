/**
 * @file warm_start.h
 * @brief Warm-starting utilities for persistent contact manifolds.
 *
 * Provides convenience functions for batch warm-starting all manifolds
 * in a ContactCache. Warm starting initialises the solver's impulse
 * accumulators from the previous frame's solution, dramatically
 * reducing the number of iterations needed for convergence.
 *
 * Design: Two-tier damping — established contacts use the full warm-start
 * factor, while newly created contacts use a reduced damping factor to
 * prevent initial impulse spikes.
 */

#pragma once

#include "contact_cache.h"
#include "contact_common.h"
#include <pulse/math/math_common.h>

namespace pulse {

// ── Single-manifold warm start ───────────────────────────────────────────────

/**
 * @brief Apply warm-start scaling to a single persistent manifold.
 *
 * Scales accumulated impulses by the appropriate factor depending on
 * whether each contact is new or established.
 *
 * @param manifold        The manifold to warm-start.
 * @param factor          Scaling factor for established contacts (0.0–1.0).
 * @param newDamping      Scaling factor for new contacts (0.0–1.0).
 */
PULSE_FORCE_INLINE void warmStartManifold(PersistentManifold& manifold,
                                           float factor,
                                           float newDamping) noexcept
{
    manifold.prepareWarmStart(factor, newDamping);
}

// ── Batch warm start ─────────────────────────────────────────────────────────

/**
 * @brief Apply warm-start scaling to all manifolds in a contact cache.
 *
 * @param cache      The contact cache containing all active manifolds.
 * @param config     Contact configuration (provides warmStartFactor and newContactDamping).
 */
inline void warmStartAllManifolds(ContactCache& cache,
                                   const ContactConfig& config) noexcept
{
    cache.warmStartAll(config.warmStartFactor, config.newContactDamping);
}

/**
 * @brief Apply warm-start scaling with explicit factors.
 *
 * @param cache      The contact cache.
 * @param factor     Scaling factor for established contacts.
 * @param newDamping Scaling factor for new contacts.
 */
inline void warmStartAllManifolds(ContactCache& cache,
                                   float factor,
                                   float newDamping) noexcept
{
    cache.warmStartAll(factor, newDamping);
}

// ── Diagnostic utilities ─────────────────────────────────────────────────────

/**
 * @struct WarmStartStats
 * @brief Diagnostic statistics about warm-start coverage.
 */
struct WarmStartStats {
    uint32_t totalManifolds;     ///< Total active manifolds.
    uint32_t totalContacts;      ///< Total active contacts across all manifolds.
    uint32_t warmStartedContacts; ///< Contacts with non-zero impulses.
    uint32_t newContacts;        ///< Contacts created this frame.
    float    averageAge;         ///< Average contact age in frames.
};

/**
 * @brief Gather warm-start diagnostic statistics from the cache.
 */
[[nodiscard]] inline WarmStartStats getWarmStartStats(const ContactCache& cache) noexcept {
    WarmStartStats stats = {};
    uint32_t ageSum = 0;

    for (uint32_t i = 0; i < cache.capacity(); ++i) {
        if (!cache.isOccupied(i)) continue;
        const PersistentManifold& pm = cache.manifoldAt(i);
        if (pm.contactCount == 0) continue;

        stats.totalManifolds++;

        for (uint32_t c = 0; c < pm.contactCount; ++c) {
            stats.totalContacts++;
            ageSum += pm.contacts[c].contactAge;

            if (pm.contacts[c].hasImpulseData()) {
                stats.warmStartedContacts++;
            }
            if (hasFlag(pm.contacts[c].flags, ContactFlags::New)) {
                stats.newContacts++;
            }
        }
    }

    stats.averageAge = (stats.totalContacts > 0)
        ? static_cast<float>(ageSum) / static_cast<float>(stats.totalContacts)
        : 0.0f;

    return stats;
}

} // namespace pulse
