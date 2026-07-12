/**
 * @file ray.h
 * @brief Ray — origin + direction, SSE4.2 accelerated.
 *
 * A ray defined by an origin point and a direction vector. The direction is
 * NOT required to be normalized (some algorithms benefit from unnormalized
 * directions), but a normalized() accessor is provided.
 *
 * Also stores the precomputed reciprocal of the direction for efficient AABB
 * slab tests (which are the hottest path in broadphase ray queries).
 *
 * Memory layout: origin (16) + direction (16) + invDirection (16) = 48 bytes, 16-byte aligned.
 * (We store invDirection because it's always needed for AABB tests and computing
 *  it once is far cheaper than computing it per-AABB.)
 */

#pragma once

#include "math_common.h"
#include "vec3.h"
#include "aabb.h"

namespace pulse {

/**
 * @struct Ray
 * @brief A ray with origin, direction, and precomputed reciprocal direction.
 *
 * 48 bytes, 16-byte aligned.
 */
struct PULSE_SIMD_ALIGN Ray {
    Vec3 origin;       ///< Ray origin.
    Vec3 direction;    ///< Ray direction (may or may not be normalized).
    Vec3 invDirection; ///< 1.0 / direction (precomputed for AABB slabs).

    // ── Constructors ──────────────────────────────────────────────────────

    /// Default: origin at zero, direction along +Z.
    PULSE_FORCE_INLINE Ray() noexcept
        : origin(Vec3::zero()),
          direction(Vec3::unitZ()),
          invDirection(Vec3(math::Infinity, math::Infinity, 1.0f))
    {}

    /// Construct from origin and direction. Direction does NOT need to be normalized.
    PULSE_FORCE_INLINE Ray(Vec3 orig, Vec3 dir) noexcept
        : origin(orig), direction(dir)
    {
        // Precompute reciprocal direction, handling near-zero components
        invDirection = Vec3(
            math::safeReciprocal(dir.getX()) == 0.0f ? math::Infinity : 1.0f / dir.getX(),
            math::safeReciprocal(dir.getY()) == 0.0f ? math::Infinity : 1.0f / dir.getY(),
            math::safeReciprocal(dir.getZ()) == 0.0f ? math::Infinity : 1.0f / dir.getZ()
        );
    }

    // ── Queries ───────────────────────────────────────────────────────────

    /// Get the point along the ray at parameter t: origin + t * direction.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 pointAt(float t) const noexcept {
        return origin + direction * t;
    }

    /// Get a normalized copy of the direction.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 normalizedDirection() const noexcept {
        return direction.normalized();
    }

    // ── Closest point ─────────────────────────────────────────────────────

    /// Closest point on the ray to a given point.
    /// @param point The query point.
    /// @return The closest point on the ray (clamped to t >= 0).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 closestPoint(Vec3 point) const noexcept {
        Vec3 diff = point - origin;
        float t = diff.dot(direction) / direction.dot(direction);
        t = math::fastMax(t, 0.0f);
        return pointAt(t);
    }

    /// Squared distance from the ray to a point.
    [[nodiscard]] PULSE_FORCE_INLINE float distanceSqTo(Vec3 point) const noexcept {
        return (point - closestPoint(point)).lengthSq();
    }

    /// Distance from the ray to a point.
    [[nodiscard]] PULSE_FORCE_INLINE float distanceTo(Vec3 point) const noexcept {
        return math::fastSqrt(distanceSqTo(point));
    }

    // ── Sphere intersection ───────────────────────────────────────────────

    /// Ray-sphere intersection.
    /// @param center Sphere center.
    /// @param radius Sphere radius.
    /// @param t [out] Parametric distance to the nearest intersection point.
    /// @return True if the ray hits the sphere.
    [[nodiscard]] PULSE_FORCE_INLINE bool intersectSphere(
        Vec3 center, float radius, float& t
    ) const noexcept {
        Vec3 oc = origin - center;
        float a = direction.dot(direction);
        float b = oc.dot(direction);
        float c = oc.dot(oc) - radius * radius;
        float discriminant = b * b - a * c;

        if (discriminant < 0.0f) return false;

        float sqrtD = math::fastSqrt(discriminant);
        float invA = 1.0f / a;

        // Try the closer intersection first
        float t0 = (-b - sqrtD) * invA;
        if (t0 >= 0.0f) {
            t = t0;
            return true;
        }

        // If t0 is negative, try the far intersection (ray origin inside sphere)
        float t1 = (-b + sqrtD) * invA;
        if (t1 >= 0.0f) {
            t = t1;
            return true;
        }

        return false;
    }

    // ── AABB intersection ─────────────────────────────────────────────────

    /// Ray-AABB intersection using the precomputed inverse direction.
    /// @param aabb The AABB to test.
    /// @param tMin [out] Entry distance.
    /// @param tMax [out] Exit distance.
    /// @return True if the ray intersects the AABB.
    [[nodiscard]] PULSE_FORCE_INLINE bool intersectAABB(
        const AABB& aabb, float& tMin, float& tMax
    ) const noexcept {
        return aabb.rayIntersect(origin, invDirection, tMin, tMax);
    }

    // ── Plane intersection ────────────────────────────────────────────────

    /// Ray-plane intersection.
    /// @param planeNormal Plane normal (must be unit length).
    /// @param planeD Signed distance from origin (plane equation: dot(n,p) + d = 0).
    /// @param t [out] Parametric distance to intersection.
    /// @return True if the ray hits the plane (not parallel).
    [[nodiscard]] PULSE_FORCE_INLINE bool intersectPlane(
        Vec3 planeNormal, float planeD, float& t
    ) const noexcept {
        float denom = planeNormal.dot(direction);
        if (math::fastAbs(denom) < math::Epsilon) return false;
        t = -(planeNormal.dot(origin) + planeD) / denom;
        return t >= 0.0f;
    }

    // ── Triangle intersection ─────────────────────────────────────────────

    /// Möller–Trumbore ray-triangle intersection.
    /// @param v0, v1, v2 Triangle vertices.
    /// @param t [out] Parametric distance.
    /// @param u [out] Barycentric coordinate u.
    /// @param v [out] Barycentric coordinate v.
    /// @return True if the ray hits the triangle.
    [[nodiscard]] PULSE_FORCE_INLINE bool intersectTriangle(
        Vec3 v0, Vec3 v1, Vec3 v2,
        float& t, float& u, float& v
    ) const noexcept {
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        Vec3 h = direction.cross(edge2);
        float a = edge1.dot(h);

        if (a > -math::Epsilon && a < math::Epsilon) return false; // Parallel

        float f = 1.0f / a;
        Vec3 s = origin - v0;
        u = f * s.dot(h);
        if (u < 0.0f || u > 1.0f) return false;

        Vec3 q = s.cross(edge1);
        v = f * direction.dot(q);
        if (v < 0.0f || u + v > 1.0f) return false;

        t = f * edge2.dot(q);
        return t > math::Epsilon;
    }

    // ── Closest point between two rays ────────────────────────────────────

    /// Find the closest points between two rays.
    /// @param other The other ray.
    /// @param t1 [out] Parameter on this ray for the closest point.
    /// @param t2 [out] Parameter on the other ray for the closest point.
    /// @return Squared distance between the two closest points.
    [[nodiscard]] PULSE_FORCE_INLINE float closestPointsBetweenRays(
        const Ray& other, float& t1, float& t2
    ) const noexcept {
        Vec3 w0 = origin - other.origin;
        float a = direction.dot(direction);
        float b = direction.dot(other.direction);
        float c = other.direction.dot(other.direction);
        float d = direction.dot(w0);
        float e = other.direction.dot(w0);

        float denom = a * c - b * b;
        if (math::fastAbs(denom) < math::Epsilon) {
            // Rays are nearly parallel
            t1 = 0.0f;
            t2 = (b > c ? d / b : e / c);
        } else {
            float invDenom = 1.0f / denom;
            t1 = (b * e - c * d) * invDenom;
            t2 = (a * e - b * d) * invDenom;
        }

        // Clamp to ray (t >= 0)
        t1 = math::fastMax(t1, 0.0f);
        t2 = math::fastMax(t2, 0.0f);

        Vec3 p1 = pointAt(t1);
        Vec3 p2 = other.pointAt(t2);
        return (p2 - p1).lengthSq();
    }

    // ── Transform ─────────────────────────────────────────────────────────

    /// Transform the ray by a 4×4 matrix (origin as point, direction as direction).
    [[nodiscard]] PULSE_FORCE_INLINE Ray transformed(const struct Mat4& mat) const noexcept;

    // ── Comparison ────────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE bool operator==(const Ray& rhs) const noexcept {
        return origin == rhs.origin && direction == rhs.direction;
    }

    [[nodiscard]] PULSE_FORCE_INLINE bool operator!=(const Ray& rhs) const noexcept {
        return !(*this == rhs);
    }
};

} // namespace pulse

// ── Deferred implementation (needs Mat4 to be fully defined) ──────────────────
#include "mat4.h"

namespace pulse {

PULSE_FORCE_INLINE Ray Ray::transformed(const Mat4& mat) const noexcept {
    Vec3 newOrigin = mat.transformPoint(origin);
    Vec3 newDir = mat.transformDirection(direction);
    return Ray(newOrigin, newDir);
}

} // namespace pulse
