/**
 * @file tri_mesh.h
 * @brief Triangle mesh collision shape — static environment geometry.
 *
 * A non-convex triangle mesh used exclusively as a static collider. Stores
 * vertex and index arrays for the triangle soup. Ray intersection uses the
 * Möller–Trumbore algorithm per triangle.
 *
 * BVH acceleration is deferred to Module 6 (BroadPhase). For now, queries
 * use brute-force iteration over all triangles.
 *
 * Non-owning view of vertex/index data — the caller owns the arrays.
 * No support function (not convex — cannot be used with GJK).
 * No mass properties (static-only, treated as infinite mass).
 */

#pragma once

#include "shape_common.h"
#include <pulse/math/transform.h>
#include <pulse/math/aabb.h>
#include <pulse/math/ray.h>

namespace pulse {

/**
 * @struct TriMesh
 * @brief A triangle mesh for static collision geometry.
 *
 * Non-owning. Vertices are Vec3, indices are uint32_t (3 per triangle).
 */
struct TriMesh {
    const Vec3*     vertices;      ///< Vertex positions (non-owning).
    const uint32_t* indices;       ///< Triangle indices, 3 per triangle (non-owning).
    uint32_t        vertexCount;   ///< Number of vertices.
    uint32_t        triangleCount; ///< Number of triangles (indexCount / 3).

    static constexpr ShapeType Type = ShapeType::TriMesh;

    // ── Constructors ──────────────────────────────────────────────────────

    /// Default: empty mesh.
    PULSE_FORCE_INLINE TriMesh() noexcept
        : vertices(nullptr), indices(nullptr), vertexCount(0), triangleCount(0)
    {}

    /// Construct from vertex and index arrays.
    PULSE_FORCE_INLINE TriMesh(const Vec3* verts, uint32_t vCount,
                               const uint32_t* idxs, uint32_t triCount) noexcept
        : vertices(verts), indices(idxs), vertexCount(vCount), triangleCount(triCount)
    {}

    // ── Triangle access ───────────────────────────────────────────────────

    /// Get the 3 vertices of triangle at the given index.
    PULSE_FORCE_INLINE void getTriangle(uint32_t triIdx, Vec3& v0, Vec3& v1, Vec3& v2) const noexcept {
        uint32_t base = triIdx * 3;
        v0 = vertices[indices[base + 0]];
        v1 = vertices[indices[base + 1]];
        v2 = vertices[indices[base + 2]];
    }

    /// Compute the face normal of a triangle (unnormalized).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 getTriangleNormal(uint32_t triIdx) const noexcept {
        Vec3 v0, v1, v2;
        getTriangle(triIdx, v0, v1, v2);
        return (v1 - v0).cross(v2 - v0);
    }

    // ── AABB computation ──────────────────────────────────────────────────

    /// Compute local-space AABB enclosing all vertices.
    [[nodiscard]] AABB computeLocalAABB() const noexcept {
        if (vertexCount == 0) return AABB();

        AABB box = AABB::fromPoint(vertices[0]);
        for (uint32_t i = 1; i < vertexCount; ++i) {
            box.expandToInclude(vertices[i]);
        }
        return box;
    }

    /// Compute world-space AABB by transforming all vertices.
    [[nodiscard]] AABB computeAABB(const Transform& tx) const noexcept {
        if (vertexCount == 0) return AABB();

        AABB box = AABB::fromPoint(tx.transformPoint(vertices[0]));
        for (uint32_t i = 1; i < vertexCount; ++i) {
            box.expandToInclude(tx.transformPoint(vertices[i]));
        }
        return box;
    }

    // ── Ray intersection (Möller–Trumbore) ────────────────────────────────

    /// Ray-triangle intersection using Möller–Trumbore algorithm.
    /// Tests a single triangle. Returns parametric distance t or < 0 on miss.
    [[nodiscard]] static PULSE_FORCE_INLINE float rayTriangle(
        Vec3 origin, Vec3 direction,
        Vec3 v0, Vec3 v1, Vec3 v2
    ) noexcept {
        Vec3 e1 = v1 - v0;
        Vec3 e2 = v2 - v0;
        Vec3 h = direction.cross(e2);
        float a = e1.dot(h);

        // Ray parallel to triangle
        if (a > -math::Epsilon && a < math::Epsilon) return -1.0f;

        float f = 1.0f / a;
        Vec3 s = origin - v0;
        float u = f * s.dot(h);
        if (u < 0.0f || u > 1.0f) return -1.0f;

        Vec3 q = s.cross(e1);
        float v = f * direction.dot(q);
        if (v < 0.0f || u + v > 1.0f) return -1.0f;

        float t = f * e2.dot(q);
        return t > math::Epsilon ? t : -1.0f;
    }

    /// Ray intersection against the entire mesh (brute force).
    /// Returns the closest hit.
    [[nodiscard]] ShapeRayResult rayIntersect(Vec3 origin, Vec3 direction) const noexcept {
        float bestT = math::Infinity;
        Vec3 bestNormal = Vec3::zero();
        bool anyHit = false;

        for (uint32_t i = 0; i < triangleCount; ++i) {
            Vec3 v0, v1, v2;
            getTriangle(i, v0, v1, v2);

            float t = rayTriangle(origin, direction, v0, v1, v2);
            if (t > 0.0f && t < bestT) {
                bestT = t;
                bestNormal = (v1 - v0).cross(v2 - v0).normalized();
                anyHit = true;
            }
        }

        if (!anyHit) return ShapeRayResult::miss();
        return ShapeRayResult(bestT, bestNormal);
    }

    // ── Closest point (brute force) ───────────────────────────────────────

    /// Find the closest point on the mesh surface to a query point.
    /// Brute-force scan over all triangles.
    [[nodiscard]] Vec3 closestPoint(Vec3 point) const noexcept {
        if (triangleCount == 0) return point;

        float bestDistSq = math::Infinity;
        Vec3 bestPoint = point;

        for (uint32_t i = 0; i < triangleCount; ++i) {
            Vec3 v0, v1, v2;
            getTriangle(i, v0, v1, v2);

            Vec3 cp = closestPointOnTriangle(point, v0, v1, v2);
            float d = (point - cp).lengthSq();
            if (d < bestDistSq) {
                bestDistSq = d;
                bestPoint = cp;
            }
        }
        return bestPoint;
    }

    // ── Point containment ─────────────────────────────────────────────────

    /// Point containment is not well-defined for non-watertight meshes.
    /// Returns false (triangle meshes are surface-only colliders).
    [[nodiscard]] PULSE_FORCE_INLINE bool containsPoint(Vec3 /*point*/) const noexcept {
        return false;
    }

private:
    /// Closest point on a triangle to a query point.
    /// Uses the Voronoi region method.
    [[nodiscard]] static PULSE_FORCE_INLINE Vec3 closestPointOnTriangle(
        Vec3 p, Vec3 a, Vec3 b, Vec3 c
    ) noexcept {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        Vec3 ap = p - a;

        float d1 = ab.dot(ap);
        float d2 = ac.dot(ap);
        if (d1 <= 0.0f && d2 <= 0.0f) return a; // Vertex A region

        Vec3 bp = p - b;
        float d3 = ab.dot(bp);
        float d4 = ac.dot(bp);
        if (d3 >= 0.0f && d4 <= d3) return b; // Vertex B region

        float vc = d1 * d4 - d3 * d2;
        if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
            float v = d1 / (d1 - d3);
            return a + ab * v; // Edge AB region
        }

        Vec3 cp_ = p - c;
        float d5 = ab.dot(cp_);
        float d6 = ac.dot(cp_);
        if (d6 >= 0.0f && d5 <= d6) return c; // Vertex C region

        float vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
            float w = d2 / (d2 - d6);
            return a + ac * w; // Edge AC region
        }

        float va = d3 * d6 - d5 * d4;
        if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
            float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return b + (c - b) * w; // Edge BC region
        }

        // Inside triangle
        float denom = 1.0f / (va + vb + vc);
        float v = vb * denom;
        float w = vc * denom;
        return a + ab * v + ac * w;
    }
};

} // namespace pulse
