/**
 * @file sphere.h
 * @brief Sphere collision shape — center at origin, defined by radius.
 *
 * The simplest collision primitive. Stored as a single float (radius).
 * Local-space center is always at the origin — world-space position comes
 * from the Transform applied at query time.
 *
 * Memory: 4 bytes (radius only). Padding to 16 bytes for SIMD alignment
 * when stored in arrays.
 */

#pragma once

#include "shape_common.h"
#include <pulse/math/transform.h>
#include <pulse/math/aabb.h>
#include <pulse/math/ray.h>

namespace pulse {

/**
 * @struct Sphere
 * @brief A sphere defined by its radius. Center is at local-space origin.
 */
struct Sphere {
    float radius; ///< Sphere radius. Must be > 0.

    static constexpr ShapeType Type = ShapeType::Sphere;

    // ── Constructors ──────────────────────────────────────────────────────

    /// Default: unit sphere.
    PULSE_FORCE_INLINE Sphere() noexcept : radius(1.0f) {}

    /// Construct with explicit radius.
    explicit PULSE_FORCE_INLINE Sphere(float r) noexcept : radius(r) {}

    // ── AABB computation ──────────────────────────────────────────────────

    /// Compute world-space AABB for a sphere at the given transform.
    [[nodiscard]] PULSE_FORCE_INLINE AABB computeAABB(const Transform& tx) const noexcept {
        Vec3 r(radius);
        return AABB(tx.position - r, tx.position + r);
    }

    /// Compute local-space AABB (centered at origin).
    [[nodiscard]] PULSE_FORCE_INLINE AABB computeLocalAABB() const noexcept {
        Vec3 r(radius);
        return AABB(-r, r);
    }

    // ── Mass properties ───────────────────────────────────────────────────

    /// Compute mass properties for a solid sphere with given density.
    /// Volume = (4/3)πr³, Inertia = (2/5)mr² on all axes.
    [[nodiscard]] PULSE_FORCE_INLINE MassProperties computeMass(float density) const noexcept {
        const float r2 = radius * radius;
        const float r3 = r2 * radius;
        const float volume = (4.0f / 3.0f) * math::Pi * r3;
        const float m = density * volume;
        const float i = (2.0f / 5.0f) * m * r2; // Diagonal inertia
        return MassProperties(
            m,
            Vec3::zero(),
            Mat3(i, 0.0f, 0.0f,
                 0.0f, i, 0.0f,
                 0.0f, 0.0f, i)
        );
    }

    // ── GJK support function ──────────────────────────────────────────────

    /// Furthest point on the sphere in the given direction (local space).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 support(Vec3 direction) const noexcept {
        float len = direction.length();
        if (len < math::Epsilon) {
            return Vec3(radius, 0.0f, 0.0f); // Arbitrary point on surface
        }
        return direction * (radius / len);
    }

    /// Furthest point on the sphere in the given direction (world space).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 supportWorld(Vec3 direction, const Transform& tx) const noexcept {
        float len = direction.length();
        if (len < math::Epsilon) {
            return tx.position + Vec3(radius, 0.0f, 0.0f);
        }
        return tx.position + direction * (radius / len);
    }

    // ── Point containment ─────────────────────────────────────────────────

    /// Test if a point (local space) is inside the sphere.
    [[nodiscard]] PULSE_FORCE_INLINE bool containsPoint(Vec3 point) const noexcept {
        return point.lengthSq() <= radius * radius;
    }

    // ── Closest point ─────────────────────────────────────────────────────

    /// Closest point on the sphere surface to a query point (local space).
    /// If the point is at the center, returns (radius, 0, 0).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 closestPoint(Vec3 point) const noexcept {
        float distSq = point.lengthSq();
        if (distSq < math::Epsilon * math::Epsilon) {
            return Vec3(radius, 0.0f, 0.0f);
        }
        float invDist = math::fastInvSqrt(distSq);
        return point * (radius * invDist);
    }

    // ── Ray intersection ──────────────────────────────────────────────────

    /// Ray-sphere intersection in local space.
    /// Ray: P(t) = origin + t * direction, sphere: |P|² = r²
    /// Solves: |o + t·d|² = r² → t²(d·d) + 2t(o·d) + (o·o - r²) = 0
    [[nodiscard]] PULSE_FORCE_INLINE ShapeRayResult rayIntersect(Vec3 origin, Vec3 direction) const noexcept {
        float a = direction.dot(direction);
        float b = 2.0f * origin.dot(direction);
        float c = origin.dot(origin) - radius * radius;

        float discriminant = b * b - 4.0f * a * c;
        if (discriminant < 0.0f) {
            return ShapeRayResult::miss();
        }

        float sqrtD = math::fastSqrt(discriminant);
        float inv2a = 0.5f / a;
        float t0 = (-b - sqrtD) * inv2a;
        float t1 = (-b + sqrtD) * inv2a;

        // Pick the nearest positive t
        float t = t0;
        if (t < 0.0f) {
            t = t1;
            if (t < 0.0f) {
                return ShapeRayResult::miss();
            }
        }

        Vec3 hitPoint = origin + direction * t;
        Vec3 normal = hitPoint * math::fastInvSqrt(hitPoint.lengthSq());
        return ShapeRayResult(t, normal);
    }
};

} // namespace pulse
