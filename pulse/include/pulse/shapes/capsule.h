/**
 * @file capsule.h
 * @brief Capsule collision shape — two hemispheres connected by a cylinder.
 *
 * Defined by radius and half-height along the local Y axis. The capsule
 * spans from (0, -halfHeight, 0) to (0, +halfHeight, 0) with hemispheres
 * at both endpoints.
 *
 * Total height = 2 * halfHeight + 2 * radius.
 * The central "segment" (the cylinder axis) runs between the hemisphere centers.
 *
 * Ideal for character controllers and limbs due to its smooth rolling behavior.
 *
 * Memory: 8 bytes (2 floats). Padded to 16 for alignment in arrays.
 */

#pragma once

#include "shape_common.h"
#include <pulse/math/transform.h>
#include <pulse/math/aabb.h>
#include <pulse/math/ray.h>

namespace pulse {

/**
 * @struct Capsule
 * @brief A capsule defined by radius and half-height along the local Y axis.
 */
struct Capsule {
    float radius;     ///< Radius of the hemispheres and cylinder.
    float halfHeight; ///< Half the distance between hemisphere centers (along Y).

    static constexpr ShapeType Type = ShapeType::Capsule;

    // ── Constructors ──────────────────────────────────────────────────────

    /// Default: radius=0.5, halfHeight=0.5 (total height = 2.0).
    PULSE_FORCE_INLINE Capsule() noexcept : radius(0.5f), halfHeight(0.5f) {}

    /// Construct with explicit radius and half-height.
    PULSE_FORCE_INLINE Capsule(float r, float hh) noexcept : radius(r), halfHeight(hh) {}

    // ── Geometry helpers ──────────────────────────────────────────────────

    /// Get the top hemisphere center (local space).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 getTopCenter() const noexcept {
        return Vec3(0.0f, halfHeight, 0.0f);
    }

    /// Get the bottom hemisphere center (local space).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 getBottomCenter() const noexcept {
        return Vec3(0.0f, -halfHeight, 0.0f);
    }

    /// Total height from bottom of bottom hemisphere to top of top hemisphere.
    [[nodiscard]] PULSE_FORCE_INLINE float totalHeight() const noexcept {
        return 2.0f * (halfHeight + radius);
    }

    // ── AABB computation ──────────────────────────────────────────────────

    /// Compute world-space AABB for a capsule at the given transform.
    [[nodiscard]] PULSE_FORCE_INLINE AABB computeAABB(const Transform& tx) const noexcept {
        // Transform the two hemisphere centers to world space
        Vec3 worldTop = tx.transformPoint(getTopCenter());
        Vec3 worldBot = tx.transformPoint(getBottomCenter());

        // Expand by radius in all directions
        Vec3 r(radius);
        AABB box;
        box.expandToInclude(worldTop + r);
        box.expandToInclude(worldTop - r);
        box.expandToInclude(worldBot + r);
        box.expandToInclude(worldBot - r);
        return box;
    }

    /// Compute local-space AABB.
    [[nodiscard]] PULSE_FORCE_INLINE AABB computeLocalAABB() const noexcept {
        return AABB(
            Vec3(-radius, -(halfHeight + radius), -radius),
            Vec3( radius,  (halfHeight + radius),  radius)
        );
    }

    // ── Mass properties ───────────────────────────────────────────────────

    /// Compute mass properties for a solid capsule with given density.
    /// Volume = π·r²·2h + (4/3)π·r³ (cylinder + sphere)
    [[nodiscard]] PULSE_FORCE_INLINE MassProperties computeMass(float density) const noexcept {
        const float r2 = radius * radius;
        const float h = 2.0f * halfHeight; // Cylinder height

        // Cylinder volume and mass
        const float cylVolume = math::Pi * r2 * h;
        const float cylMass = density * cylVolume;

        // Sphere (two hemispheres = one sphere) volume and mass
        const float sphVolume = (4.0f / 3.0f) * math::Pi * r2 * radius;
        const float sphMass = density * sphVolume;

        const float totalMass = cylMass + sphMass;

        // Cylinder inertia about its center (Y axis = symmetry axis)
        float cylIyy = cylMass * r2 * 0.5f;                                    // Along symmetry axis
        float cylIxx = cylMass * (3.0f * r2 + h * h) / 12.0f;                  // Perpendicular axes

        // Sphere inertia about its center
        float sphI = (2.0f / 5.0f) * sphMass * r2;

        // Parallel axis theorem: hemispheres displaced by ±halfHeight from CoM
        // Each hemisphere center is at ±(halfHeight + 3r/8) from capsule center
        // Simplified: use sphere inertia + parallel axis for combined hemisphere
        float sphOffset = halfHeight + (3.0f * radius / 8.0f);
        float sphIxx = sphI + sphMass * sphOffset * sphOffset;
        float sphIyy = sphI; // Along Y axis, no offset needed for symmetry

        float ixx = cylIxx + sphIxx;
        float iyy = cylIyy + sphIyy;

        return MassProperties(
            totalMass,
            Vec3::zero(),
            Mat3(ixx, 0.0f, 0.0f,
                 0.0f, iyy, 0.0f,
                 0.0f, 0.0f, ixx) // Izz = Ixx by rotational symmetry
        );
    }

    // ── GJK support function ──────────────────────────────────────────────

    /// Furthest point on the capsule in the given direction (local space).
    /// Pick the hemisphere center toward the direction, then offset by radius.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 support(Vec3 direction) const noexcept {
        // Choose top or bottom hemisphere center based on Y component
        Vec3 center = direction.getY() >= 0.0f ? getTopCenter() : getBottomCenter();

        // Add radius in the direction
        float len = direction.length();
        if (len < math::Epsilon) {
            return center + Vec3(0.0f, radius, 0.0f);
        }
        return center + direction * (radius / len);
    }

    /// Furthest point in the given direction (world space).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 supportWorld(Vec3 direction, const Transform& tx) const noexcept {
        Vec3 localDir = tx.inverseTransformDirection(direction);
        Vec3 localSupport = support(localDir);
        return tx.transformPoint(localSupport);
    }

    // ── Point containment ─────────────────────────────────────────────────

    /// Test if a point (local space) is inside the capsule.
    /// Point is inside if its distance to the central segment ≤ radius.
    [[nodiscard]] PULSE_FORCE_INLINE bool containsPoint(Vec3 point) const noexcept {
        // Central segment from (0, -halfHeight, 0) to (0, halfHeight, 0)
        // Clamp Y to the segment, then check distance
        float y = math::clamp(point.getY(), -halfHeight, halfHeight);
        Vec3 nearest(0.0f, y, 0.0f);
        return (point - nearest).lengthSq() <= radius * radius;
    }

    // ── Closest point ─────────────────────────────────────────────────────

    /// Closest point on the capsule surface to a query point (local space).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 closestPoint(Vec3 point) const noexcept {
        // Find closest point on the central segment
        float y = math::clamp(point.getY(), -halfHeight, halfHeight);
        Vec3 segPoint(0.0f, y, 0.0f);

        Vec3 diff = point - segPoint;
        float distSq = diff.lengthSq();

        if (distSq < math::Epsilon * math::Epsilon) {
            // Point is on the segment axis — return arbitrary surface point
            return segPoint + Vec3(radius, 0.0f, 0.0f);
        }

        float invDist = math::fastInvSqrt(distSq);
        return segPoint + diff * (radius * invDist);
    }

    // ── Ray intersection ──────────────────────────────────────────────────

    /// Ray-capsule intersection in local space.
    /// Tests against the cylinder body and both hemisphere caps.
    [[nodiscard]] PULSE_FORCE_INLINE ShapeRayResult rayIntersect(Vec3 origin, Vec3 direction) const noexcept {
        float bestT = math::Infinity;
        Vec3 bestNormal = Vec3::zero();
        bool anyHit = false;

        // ── Infinite cylinder test (ignore Y component for cylinder body) ──
        float ox = origin.getX(), oz = origin.getZ();
        float dx = direction.getX(), dz = direction.getZ();

        float a = dx * dx + dz * dz;
        float b = 2.0f * (ox * dx + oz * dz);
        float c = ox * ox + oz * oz - radius * radius;

        if (a > math::Epsilon) {
            float disc = b * b - 4.0f * a * c;
            if (disc >= 0.0f) {
                float sqrtD = math::fastSqrt(disc);
                float inv2a = 0.5f / a;
                float t0 = (-b - sqrtD) * inv2a;
                float t1 = (-b + sqrtD) * inv2a;

                for (int i = 0; i < 2; ++i) {
                    float t = (i == 0) ? t0 : t1;
                    if (t < 0.0f) continue;
                    float hitY = origin.getY() + t * direction.getY();
                    if (hitY >= -halfHeight && hitY <= halfHeight && t < bestT) {
                        bestT = t;
                        Vec3 hp = origin + direction * t;
                        bestNormal = Vec3(hp.getX(), 0.0f, hp.getZ()).normalized();
                        anyHit = true;
                    }
                }
            }
        }

        // ── Hemisphere tests ──
        Vec3 centers[2] = { getBottomCenter(), getTopCenter() };
        for (int s = 0; s < 2; ++s) {
            Vec3 oc = origin - centers[s];
            float sa = direction.dot(direction);
            float sb = 2.0f * oc.dot(direction);
            float sc = oc.dot(oc) - radius * radius;

            float disc = sb * sb - 4.0f * sa * sc;
            if (disc < 0.0f) continue;

            float sqrtD = math::fastSqrt(disc);
            float inv2a = 0.5f / sa;
            float t0 = (-sb - sqrtD) * inv2a;
            float t1 = (-sb + sqrtD) * inv2a;

            for (int i = 0; i < 2; ++i) {
                float t = (i == 0) ? t0 : t1;
                if (t < 0.0f) continue;
                float hitY = origin.getY() + t * direction.getY();
                // Top hemisphere: hitY >= halfHeight, Bottom hemisphere: hitY <= -halfHeight
                bool valid = (s == 0) ? (hitY <= -halfHeight + math::Epsilon)
                                      : (hitY >=  halfHeight - math::Epsilon);
                if (valid && t < bestT) {
                    bestT = t;
                    Vec3 hp = origin + direction * t;
                    bestNormal = (hp - centers[s]).normalized();
                    anyHit = true;
                }
            }
        }

        if (!anyHit) return ShapeRayResult::miss();
        return ShapeRayResult(bestT, bestNormal);
    }
};

} // namespace pulse
