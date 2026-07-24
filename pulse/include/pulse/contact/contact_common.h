/**
 * @file contact_common.h
 * @brief Shared types and constants for the contact management module.
 *
 * Defines the body-pair key for cache lookups, per-contact status flags,
 * and configuration parameters that govern contact persistence, pruning,
 * and warm-starting behaviour.
 *
 * Design: BodyPairKey packs two 32-bit body IDs into a single canonical
 * 64-bit key (lower ID always first) so that (A,B) and (B,A) hash
 * identically and the cache never stores duplicate pairs.
 */

#pragma once

#include <pulse/math/math_common.h>
#include <cstdint>

namespace pulse {

// ── Body-pair key ────────────────────────────────────────────────────────────

/**
 * @struct BodyPairKey
 * @brief Canonical identifier for a pair of bodies.
 *
 * Stores two body IDs in sorted order (low, high) so that the pair
 * (A,B) and (B,A) produce the same key. Used as hash-map key in
 * the ContactCache.
 */
struct BodyPairKey {
    uint32_t low;   ///< The smaller body ID.
    uint32_t high;  ///< The larger body ID.

    /// Default: invalid pair (0xFFFFFFFF, 0xFFFFFFFF).
    PULSE_FORCE_INLINE BodyPairKey() noexcept
        : low(0xFFFFFFFFu), high(0xFFFFFFFFu)
    {}

    /// Construct from two body IDs — automatically sorted.
    PULSE_FORCE_INLINE BodyPairKey(uint32_t idA, uint32_t idB) noexcept {
        if (idA <= idB) { low = idA; high = idB; }
        else            { low = idB; high = idA; }
    }

    /// Pack into a single 64-bit value for fast comparison.
    [[nodiscard]] PULSE_FORCE_INLINE uint64_t packed() const noexcept {
        return (static_cast<uint64_t>(high) << 32u) | static_cast<uint64_t>(low);
    }

    /// Hash for use in open-addressing hash tables.
    /// Uses a Murmur-style finaliser on the packed value.
    [[nodiscard]] PULSE_FORCE_INLINE uint32_t hash() const noexcept {
        uint64_t h = packed();
        h ^= h >> 33;
        h *= 0xFF51AFD7ED558CCDull;
        h ^= h >> 33;
        h *= 0xC4CEB9FE1A85EC53ull;
        h ^= h >> 33;
        return static_cast<uint32_t>(h);
    }

    /// Check if this key represents a valid pair (not the sentinel).
    [[nodiscard]] PULSE_FORCE_INLINE bool isValid() const noexcept {
        return low != 0xFFFFFFFFu || high != 0xFFFFFFFFu;
    }

    // ── Comparison ────────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE bool operator==(const BodyPairKey& rhs) const noexcept {
        return low == rhs.low && high == rhs.high;
    }
    [[nodiscard]] PULSE_FORCE_INLINE bool operator!=(const BodyPairKey& rhs) const noexcept {
        return !(*this == rhs);
    }
    [[nodiscard]] PULSE_FORCE_INLINE bool operator<(const BodyPairKey& rhs) const noexcept {
        return packed() < rhs.packed();
    }
};

// ── Contact flags ────────────────────────────────────────────────────────────

/**
 * @enum ContactFlags
 * @brief Per-contact status bitmask.
 *
 * Tracks the lifecycle state of each contact point within a
 * persistent manifold.
 */
enum class ContactFlags : uint8_t {
    None       = 0,
    New        = 1 << 0,  ///< Contact was just created this frame.
    Persisted  = 1 << 1,  ///< Contact matched from a previous frame.
    Removed    = 1 << 2,  ///< Contact is flagged for removal.
    HasImpulse = 1 << 3,  ///< Contact carries warm-start impulse data.
};

/// Bitwise OR for ContactFlags.
[[nodiscard]] PULSE_FORCE_INLINE ContactFlags operator|(ContactFlags a, ContactFlags b) noexcept {
    return static_cast<ContactFlags>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

/// Bitwise AND for ContactFlags.
[[nodiscard]] PULSE_FORCE_INLINE ContactFlags operator&(ContactFlags a, ContactFlags b) noexcept {
    return static_cast<ContactFlags>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
}

/// Bitwise NOT for ContactFlags.
[[nodiscard]] PULSE_FORCE_INLINE ContactFlags operator~(ContactFlags a) noexcept {
    return static_cast<ContactFlags>(~static_cast<uint8_t>(a));
}

/// In-place OR for ContactFlags.
PULSE_FORCE_INLINE ContactFlags& operator|=(ContactFlags& a, ContactFlags b) noexcept {
    a = a | b;
    return a;
}

/// In-place AND for ContactFlags.
PULSE_FORCE_INLINE ContactFlags& operator&=(ContactFlags& a, ContactFlags b) noexcept {
    a = a & b;
    return a;
}

/// Test if a flag is set.
[[nodiscard]] PULSE_FORCE_INLINE bool hasFlag(ContactFlags flags, ContactFlags test) noexcept {
    return (static_cast<uint8_t>(flags) & static_cast<uint8_t>(test)) != 0;
}

// ── Contact configuration ────────────────────────────────────────────────────

/**
 * @struct ContactConfig
 * @brief Configuration parameters for the contact management system.
 */
struct ContactConfig {
    /// Distance beyond which a persisted contact is considered broken
    /// and removed from the manifold (metres).
    float breakDistance;

    /// Maximum position delta for matching a new contact point to an
    /// existing cached contact (squared, metres²).
    float matchDistanceSq;

    /// Warm-start impulse scaling factor (0.0 = no warm start, 1.0 = full).
    /// Typical: 0.8 – 0.95.  Values above 1.0 can cause instability.
    float warmStartFactor;

    /// Damping factor applied to warm-start impulses on newly created
    /// contacts (typically lower than warmStartFactor).
    float newContactDamping;

    /// Maximum number of body pairs the cache can hold.
    uint32_t maxCachedPairs;

    /// Default configuration with production-tuned values.
    PULSE_FORCE_INLINE ContactConfig() noexcept
        : breakDistance(0.02f),
          matchDistanceSq(0.04f * 0.04f),  // 4 cm match radius
          warmStartFactor(0.85f),
          newContactDamping(0.3f),
          maxCachedPairs(16384u)
    {}

    /// Custom configuration.
    PULSE_FORCE_INLINE ContactConfig(float breakDist, float matchDist,
                                     float wsFactor, float newDamp,
                                     uint32_t maxPairs) noexcept
        : breakDistance(breakDist),
          matchDistanceSq(matchDist * matchDist),
          warmStartFactor(wsFactor),
          newContactDamping(newDamp),
          maxCachedPairs(maxPairs)
    {}
};

} // namespace pulse
