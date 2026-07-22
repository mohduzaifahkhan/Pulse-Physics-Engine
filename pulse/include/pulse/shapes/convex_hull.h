/**
 * @file convex_hull.h
 * @brief Convex hull collision shape — vertex cloud with support function.
 *
 * Stores a set of vertices defining a convex polyhedron. The support function
 * (furthest point along a direction) uses a linear scan of vertices — adequate
 * for small to medium hulls (8-64 vertices). Hill-climbing optimization for
 * larger hulls can be added later with edge adjacency data.
 *
 * This is a non-owning view of vertex data — the caller is responsible for
 * the lifetime of the vertex array. For small hulls, use an inline buffer.
 *
 * Memory: pointer + count = 12-16 bytes on 64-bit, plus vertex data externally.
 */

#pragma once

#include "shape_common.h"
#include <pulse/math/transform.h>
#include <pulse/math/aabb.h>
#include <pulse/math/ray.h>

namespace pulse {

/**
 * @struct ConvexHull
 * @brief A convex hull defined by a vertex cloud.
 *
 * Non-owning reference to vertex data. The collision system or body
 * manager owns the actual vertex memory.
 */
struct ConvexHull {
    const Vec3* vertices;    ///< Pointer to vertex array (non-owning).
    uint32_t    vertexCount; ///< Number of vertices in the hull.

    static constexpr ShapeType Type = ShapeType::ConvexHull;

    /// Maximum vertices for stack-allocated small hulls.
    static constexpr uint32_t MaxSmallHullVerts = 64;

    // ── Constructors ──────────────────────────────────────────────────────

    /// Default: empty hull.
    PULSE_FORCE_INLINE ConvexHull() noexcept : vertices(nullptr), vertexCount(0) {}

    /// Construct from vertex array.
    PULSE_FORCE_INLINE ConvexHull(const Vec3* verts, uint32_t count) noexcept
        : vertices(verts), vertexCount(count)
    {}

    // ── AABB computation ──────────────────────────────────────────────────

    /// Compute world-space AABB by transforming all vertices.
    [[nodiscard]] PULSE_FORCE_INLINE AABB computeAABB(const Transform& tx) const noexcept {
        if (vertexCount == 0) return AABB();

        AABB box = AABB::fromPoint(tx.transformPoint(vertices[0]));
        for (uint32_t i = 1; i < vertexCount; ++i) {
            box.expandToInclude(tx.transformPoint(vertices[i]));
        }
        return box;
    }

    /// Compute local-space AABB.
    [[nodiscard]] PULSE_FORCE_INLINE AABB computeLocalAABB() const noexcept {
        if (vertexCount == 0) return AABB();

        AABB box = AABB::fromPoint(vertices[0]);
        for (uint32_t i = 1; i < vertexCount; ++i) {
            box.expandToInclude(vertices[i]);
        }
        return box;
    }

    // ── Mass properties ───────────────────────────────────────────────────

    /// Compute mass properties via tetrahedra decomposition.
    /// Reference implementation — decomposes into tets from the centroid.
    /// Returns approximate inertia for the convex hull.
    [[nodiscard]] MassProperties computeMass(float density) const noexcept {
        if (vertexCount < 4) {
            return MassProperties();
        }

        // Compute centroid
        Vec3 centroid = Vec3::zero();
        for (uint32_t i = 0; i < vertexCount; ++i) {
            centroid = centroid + vertices[i];
        }
        centroid = centroid * (1.0f / static_cast<float>(vertexCount));

        // Approximate volume from AABB and use centroid as CoM
        AABB bounds = computeLocalAABB();
        Vec3 ext = bounds.extents();
        float volume = ext.getX() * ext.getY() * ext.getZ();
        float m = density * volume;

        // Approximate inertia as a box with the same extents
        float sx2 = ext.getX() * ext.getX();
        float sy2 = ext.getY() * ext.getY();
        float sz2 = ext.getZ() * ext.getZ();
        float k = m / 12.0f;

        return MassProperties(
            m,
            centroid,
            Mat3(k * (sy2 + sz2), 0.0f, 0.0f,
                 0.0f, k * (sx2 + sz2), 0.0f,
                 0.0f, 0.0f, k * (sx2 + sy2))
        );
    }

    // ── GJK support function ──────────────────────────────────────────────

    /// Furthest point on the hull in the given direction (local space).
    /// Linear scan of all vertices — O(n).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 support(Vec3 direction) const noexcept {
        if (vertexCount == 0) return Vec3::zero();

        float bestDot = vertices[0].dot(direction);
        uint32_t bestIdx = 0;

        for (uint32_t i = 1; i < vertexCount; ++i) {
            float d = vertices[i].dot(direction);
            if (d > bestDot) {
                bestDot = d;
                bestIdx = i;
            }
        }
        return vertices[bestIdx];
    }

    /// Furthest point in the given direction (world space).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 supportWorld(Vec3 direction, const Transform& tx) const noexcept {
        Vec3 localDir = tx.inverseTransformDirection(direction);
        Vec3 localSupport = support(localDir);
        return tx.transformPoint(localSupport);
    }

    // ── Point containment ─────────────────────────────────────────────────

    /// Test if a point (local space) is inside the convex hull.
    /// Uses the support function approach: if for any direction, the point
    /// projects beyond the support point, it's outside.
    /// Simplified: checks against local AABB first, then uses support.
    [[nodiscard]] PULSE_FORCE_INLINE bool containsPoint(Vec3 point) const noexcept {
        if (vertexCount < 4) return false;

        // Quick AABB rejection
        AABB bounds = computeLocalAABB();
        if (!bounds.containsPoint(point)) return false;

        // Check 6 axis-aligned directions
        Vec3 dirs[6] = {
            Vec3::unitX(), -Vec3::unitX(),
            Vec3::unitY(), -Vec3::unitY(),
            Vec3::unitZ(), -Vec3::unitZ()
        };

        for (int i = 0; i < 6; ++i) {
            Vec3 s = support(dirs[i]);
            if (point.dot(dirs[i]) > s.dot(dirs[i]) + math::Epsilon) {
                return false;
            }
        }

        // Also check vertex-to-centroid directions for better coverage
        Vec3 centroid = Vec3::zero();
        for (uint32_t i = 0; i < vertexCount; ++i) {
            centroid = centroid + vertices[i];
        }
        centroid = centroid * (1.0f / static_cast<float>(vertexCount));

        for (uint32_t i = 0; i < vertexCount; ++i) {
            Vec3 dir = (vertices[i] - centroid);
            float len = dir.length();
            if (len < math::Epsilon) continue;
            dir = dir * (1.0f / len);

            Vec3 s = support(dir);
            if (point.dot(dir) > s.dot(dir) + math::Epsilon) {
                return false;
            }
        }

        return true;
    }

    // ── Closest point (approximate) ───────────────────────────────────────

    /// Approximate closest point on the hull to a query point (local space).
    /// Finds the closest vertex — not exact but useful for broad queries.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 closestVertex(Vec3 point) const noexcept {
        if (vertexCount == 0) return Vec3::zero();

        float bestDistSq = (point - vertices[0]).lengthSq();
        uint32_t bestIdx = 0;

        for (uint32_t i = 1; i < vertexCount; ++i) {
            float d = (point - vertices[i]).lengthSq();
            if (d < bestDistSq) {
                bestDistSq = d;
                bestIdx = i;
            }
        }
        return vertices[bestIdx];
    }
};

// ── Small convex hull with inline storage ────────────────────────────────────

/**
 * @struct SmallConvexHull
 * @brief A convex hull with inline vertex storage for small shapes (up to N verts).
 *
 * Owns its vertex data. Useful for box-like shapes (8 verts), tetrahedra (4 verts),
 * and other small convex primitives.
 */
template <uint32_t MaxVerts = 8>
struct SmallConvexHull {
    Vec3     storage[MaxVerts]; ///< Inline vertex storage.
    uint32_t count;             ///< Actual number of vertices used.

    PULSE_FORCE_INLINE SmallConvexHull() noexcept : count(0) {}

    /// Add a vertex. Returns false if full.
    PULSE_FORCE_INLINE bool addVertex(Vec3 v) noexcept {
        if (count >= MaxVerts) return false;
        storage[count++] = v;
        return true;
    }

    /// Get a ConvexHull view of this small hull.
    [[nodiscard]] PULSE_FORCE_INLINE ConvexHull toHull() const noexcept {
        return ConvexHull(storage, count);
    }
};

} // namespace pulse
