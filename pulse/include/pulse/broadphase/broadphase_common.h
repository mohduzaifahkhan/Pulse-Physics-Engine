/**
 * @file broadphase_common.h
 * @brief Shared types for the broad-phase collision detection module.
 *
 * Defines the proxy handle system, proxy data, overlap pair representation,
 * and configuration structures used by all broad-phase algorithms.
 *
 * Design: All broad-phase structures operate on fat AABBs (inflated by a margin)
 * to reduce rebuilds when bodies move small distances. Proxies are identified by
 * opaque handles (index + generation counter) for safe external references.
 */

#pragma once

#include <pulse/math/aabb.h>
#include <pulse/math/vec3.h>
#include <cstdint>
#include <cstring>

namespace pulse {

// ── Proxy handle ─────────────────────────────────────────────────────────────

/**
 * @struct ProxyHandle
 * @brief Opaque handle to a broad-phase proxy (body AABB entry).
 *
 * Encodes an index and a generation counter so that stale handles can be
 * detected without pointer comparisons.
 *
 * Invalid handle: index = 0xFFFFFF, generation = 0xFF → raw = 0xFFFFFFFF.
 */
struct ProxyHandle {
    uint32_t raw; ///< Packed: [23:0] = index, [31:24] = generation.

    static constexpr uint32_t InvalidRaw    = 0xFFFFFFFFu;
    static constexpr uint32_t IndexMask     = 0x00FFFFFFu;
    static constexpr uint32_t GenerationShift = 24u;

    // ── Constructors ──────────────────────────────────────────────────────

    PULSE_FORCE_INLINE ProxyHandle() noexcept : raw(InvalidRaw) {}
    explicit PULSE_FORCE_INLINE ProxyHandle(uint32_t r) noexcept : raw(r) {}
    PULSE_FORCE_INLINE ProxyHandle(uint32_t index, uint8_t generation) noexcept
        : raw((static_cast<uint32_t>(generation) << GenerationShift) | (index & IndexMask))
    {}

    [[nodiscard]] static PULSE_FORCE_INLINE ProxyHandle invalid() noexcept {
        return ProxyHandle();
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE uint32_t index()      const noexcept { return raw & IndexMask; }
    [[nodiscard]] PULSE_FORCE_INLINE uint8_t  generation() const noexcept { return static_cast<uint8_t>(raw >> GenerationShift); }
    [[nodiscard]] PULSE_FORCE_INLINE bool     isValid()    const noexcept { return raw != InvalidRaw; }

    // ── Comparison ────────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE bool operator==(ProxyHandle rhs) const noexcept { return raw == rhs.raw; }
    [[nodiscard]] PULSE_FORCE_INLINE bool operator!=(ProxyHandle rhs) const noexcept { return raw != rhs.raw; }
    [[nodiscard]] PULSE_FORCE_INLINE bool operator< (ProxyHandle rhs) const noexcept { return raw <  rhs.raw; }
};

// ── Overlap pair ─────────────────────────────────────────────────────────────

/**
 * @struct OverlapPair
 * @brief A pair of proxies whose fat AABBs overlap.
 *
 * Always stored with a.raw < b.raw for canonical ordering (avoids duplicates).
 */
struct OverlapPair {
    ProxyHandle a; ///< First proxy (lower handle value).
    ProxyHandle b; ///< Second proxy (higher handle value).

    PULSE_FORCE_INLINE OverlapPair() noexcept {}
    PULSE_FORCE_INLINE OverlapPair(ProxyHandle x, ProxyHandle y) noexcept {
        // Sort so a.raw < b.raw for canonical form
        if (x.raw < y.raw) { a = x; b = y; }
        else               { a = y; b = x; }
    }

    [[nodiscard]] PULSE_FORCE_INLINE bool operator==(const OverlapPair& rhs) const noexcept {
        return a == rhs.a && b == rhs.b;
    }
    [[nodiscard]] PULSE_FORCE_INLINE bool operator<(const OverlapPair& rhs) const noexcept {
        if (a != rhs.a) return a < rhs.a;
        return b < rhs.b;
    }

    /// Hash for use in hash-set deduplication.
    [[nodiscard]] PULSE_FORCE_INLINE uint32_t hash() const noexcept {
        // Szudzik pairing function — unique for canonical pairs
        return (a.raw >= b.raw) ? (a.raw * a.raw + a.raw + b.raw)
                                : (a.raw + b.raw * b.raw);
    }
};

// ── Broad-phase proxy ─────────────────────────────────────────────────────────

/**
 * @struct BroadPhaseProxy
 * @brief Per-body data stored in the broad-phase.
 *
 * The fat AABB is slightly inflated compared to the body's tight AABB.
 * The proxy only moves if the body moves far enough to exit the fat AABB,
 * reducing the number of BVH updates per frame.
 */
struct PULSE_SIMD_ALIGN BroadPhaseProxy {
    AABB     fatAABB;    ///< Inflated bounding box (stored in the broadphase).
    void*    userData;   ///< Caller-owned data (e.g., body pointer or index).
    uint32_t group;      ///< Collision group bitmask.
    uint32_t mask;       ///< Collision mask — collides with (group & mask) != 0.
    bool     isSleeping; ///< If true, skip pair generation for this proxy.
    uint8_t  _pad[3];    ///< Padding.

    PULSE_FORCE_INLINE BroadPhaseProxy() noexcept
        : userData(nullptr), group(0xFFFFFFFFu), mask(0xFFFFFFFFu), isSleeping(false)
    { _pad[0] = _pad[1] = _pad[2] = 0; }
};

// ── Broad-phase configuration ─────────────────────────────────────────────────

/**
 * @struct BroadPhaseConfig
 * @brief Configuration parameters shared by all broad-phase implementations.
 */
struct BroadPhaseConfig {
    float    fatMargin;  ///< AABB inflation margin on each side (default 0.1 m).
    uint32_t maxProxies; ///< Maximum number of proxies (default 4096).
    uint32_t maxPairs;   ///< Maximum output overlap pairs (default 65536).

    PULSE_FORCE_INLINE BroadPhaseConfig() noexcept
        : fatMargin(0.1f), maxProxies(4096u), maxPairs(65536u)
    {}

    PULSE_FORCE_INLINE BroadPhaseConfig(float margin, uint32_t proxies, uint32_t pairs) noexcept
        : fatMargin(margin), maxProxies(proxies), maxPairs(pairs)
    {}
};

// ── Helper: fat AABB ──────────────────────────────────────────────────────────

/// Inflate an AABB by a uniform margin on all sides.
[[nodiscard]] PULSE_FORCE_INLINE AABB makeFatAABB(const AABB& tight, float margin) noexcept {
    return tight.expanded(margin);
}

/// Inflate an AABB by a margin and extend in the displacement direction.
/// This produces a predictive fat AABB that reduces re-insertions for
/// steadily-moving bodies.
[[nodiscard]] PULSE_FORCE_INLINE AABB makeFatAABB(const AABB& tight, float margin, Vec3 displacement) noexcept {
    return tight.expanded(margin).swept(displacement);
}

/// Test if two proxies can collide based on group/mask filters.
[[nodiscard]] PULSE_FORCE_INLINE bool canCollide(const BroadPhaseProxy& a,
                                                  const BroadPhaseProxy& b) noexcept {
    return (a.group & b.mask) != 0u && (b.group & a.mask) != 0u
        && !a.isSleeping && !b.isSleeping;
}

} // namespace pulse
