/**
 * @file cylinder.h
 * @brief Cylinder collision shape — radius + half-height along Y axis.
 *
 * A solid cylinder with flat circular caps at ±halfHeight. Axis-aligned
 * along the local Y axis. Common in industrial simulations (wheels, barrels,
 * structural columns).
 *
 * Memory: 8 bytes (2 floats). Padded to 16 for alignment.
 */

#pragma once

#include "shape_common.h"
#include <pulse/math/transform.h>
#include <pulse/math/aabb.h>
#include <pulse/math/ray.h>

namespace pulse {

/**
 * @struct Cylinder
 * @brief A cylinder defined by radius and half-height along the local Y axis.
 */
struct Cylinder {
    float radius;     ///< Radius of the circular cross-section.
    float halfHeight; ///< Half the height along the Y axis.

    static constexpr ShapeType Type = ShapeType::Cylinder;

    // ── Constructors ──────────────────────────────────────────────────────

    /// Default: radius=0.5, halfHeight=0.5 (total height = 1.0).
    PULSE_FORCE_INLINE Cylinder() noexcept : radius(0.5f), halfHeight(0.5f) {}

    /// Construct with explicit radius and half-height.
    PULSE_FORCE_INLINE Cylinder(float r, float hh) noexcept : radius(r), halfHeight(hh) {}

    // ── AABB computation ──────────────────────────────────────────────────

    /// Compute world-space AABB for a rotated cylinder.
    [[nodiscard]] PULSE_FORCE_INLINE AABB computeAABB(const Transform& tx) const noexcept {
        // The cylinder's local Y axis in world space
        Vec3 localY = Vec3::unitY();
        Vec3 worldY = tx.transformDirection(localY);

        // For each world axis i, the half-extent is:
        //   |worldY.i| * halfHeight + sqrt(1 - worldY.i²) * radius
        // This accounts for the rotated cylinder's projection.
        float wy_x = worldY.getX(), wy_y = worldY.getY(), wy_z = worldY.getZ();

        float ex = math::fastAbs(wy_x) * halfHeight +
                   math::fastSqrt(math::fastMax(1.0f - wy_x * wy_x, 0.0f)) * radius;
        float ey = math::fastAbs(wy_y) * halfHeight +
                   math::fastSqrt(math::fastMax(1.0f - wy_y * wy_y, 0.0f)) * radius;
        float ez = math::fastAbs(wy_z) * halfHeight +
                   math::fastSqrt(math::fastMax(1.0f - wy_z * wy_z, 0.0f)) * radius;

        Vec3 worldExtent(ex, ey, ez);
        return AABB(tx.position - worldExtent, tx.position + worldExtent);
    }

    /// Compute local-space AABB.
    [[nodiscard]] PULSE_FORCE_INLINE AABB computeLocalAABB() const noexcept {
        return AABB(
            Vec3(-radius, -halfHeight, -radius),
            Vec3( radius,  halfHeight,  radius)
        );
    }

    // ── Mass properties ───────────────────────────────────────────────────

    /// Compute mass properties for a solid cylinder with given density.
    /// Volume = π·r²·2h
    [[nodiscard]] PULSE_FORCE_INLINE MassProperties computeMass(float density) const noexcept {
        const float r2 = radius * radius;
        const float h = 2.0f * halfHeight;
        const float volume = math::Pi * r2 * h;
        const float m = density * volume;

        // Inertia of a solid cylinder about its center
        // Along symmetry axis (Y): Iyy = (1/2)mr²
        // Perpendicular (X, Z):    Ixx = Izz = (1/12)m(3r² + h²)
        float iyy = 0.5f * m * r2;
        float ixx = m * (3.0f * r2 + h * h) / 12.0f;

        return MassProperties(
            m,
            Vec3::zero(),
            Mat3(ixx, 0.0f, 0.0f,
                 0.0f, iyy, 0.0f,
                 0.0f, 0.0f, ixx)
        );
    }

    // ── GJK support function ──────────────────────────────────────────────

    /// Furthest point on the cylinder in the given direction (local space).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 support(Vec3 direction) const noexcept {
        // Y component: pick top or bottom cap
        float y = direction.getY() >= 0.0f ? halfHeight : -halfHeight;

        // XZ component: project direction onto XZ plane and extend by radius
        float dx = direction.getX();
        float dz = direction.getZ();
        float lenXZ = math::fastSqrt(dx * dx + dz * dz);

        float x, z;
        if (lenXZ > math::Epsilon) {
            float invLen = radius / lenXZ;
            x = dx * invLen;
            z = dz * invLen;
        } else {
            // Direction is purely along Y — pick arbitrary XZ point
            x = radius;
            z = 0.0f;
        }

        return Vec3(x, y, z);
    }

    /// Furthest point in the given direction (world space).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 supportWorld(Vec3 direction, const Transform& tx) const noexcept {
        Vec3 localDir = tx.inverseTransformDirection(direction);
        Vec3 localSupport = support(localDir);
        return tx.transformPoint(localSupport);
    }

    // ── Point containment ─────────────────────────────────────────────────

    /// Test if a point (local space) is inside the cylinder.
    [[nodiscard]] PULSE_FORCE_INLINE bool containsPoint(Vec3 point) const noexcept {
        // Check height
        if (math::fastAbs(point.getY()) > halfHeight) return false;

        // Check radial distance in XZ plane
        float dx = point.getX(), dz = point.getZ();
        return (dx * dx + dz * dz) <= radius * radius;
    }

    // ── Closest point ─────────────────────────────────────────────────────

    /// Closest point on the cylinder surface to a query point (local space).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 closestPoint(Vec3 point) const noexcept {
        // Clamp Y to half-height
        float y = math::clamp(point.getY(), -halfHeight, halfHeight);

        // Project XZ to cylinder radius
        float px = point.getX(), pz = point.getZ();
        float distXZ = math::fastSqrt(px * px + pz * pz);

        float x, z;
        if (distXZ > math::Epsilon) {
            if (distXZ > radius) {
                // Outside cylinder — project to surface
                float scale = radius / distXZ;
                x = px * scale;
                z = pz * scale;
            } else {
                // Inside cylinder — find closest feature (cap or side)
                float distToCap = halfHeight - math::fastAbs(y);
                float distToSide = radius - distXZ;
                if (distToSide < distToCap) {
                    // Closer to cylinder side
                    float scale = radius / distXZ;
                    x = px * scale;
                    z = pz * scale;
                } else {
                    // Closer to cap
                    x = px;
                    z = pz;
                    y = (point.getY() >= 0.0f) ? halfHeight : -halfHeight;
                }
            }
        } else {
            // On the axis — closest point is on the nearer cap
            x = radius;
            z = 0.0f;
        }

        return Vec3(x, y, z);
    }

    // ── Ray intersection ──────────────────────────────────────────────────

    /// Ray-cylinder intersection in local space.
    /// Tests the infinite cylinder body (clipped to caps) and both flat caps.
    [[nodiscard]] PULSE_FORCE_INLINE ShapeRayResult rayIntersect(Vec3 origin, Vec3 direction) const noexcept {
        float bestT = math::Infinity;
        Vec3 bestNormal = Vec3::zero();
        bool anyHit = false;

        // ── Cylinder body (XZ plane) ──
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

        // ── Flat cap tests ──
        if (math::fastAbs(direction.getY()) > math::Epsilon) {
            float invDy = 1.0f / direction.getY();
            for (int s = 0; s < 2; ++s) {
                float capY = (s == 0) ? -halfHeight : halfHeight;
                float t = (capY - origin.getY()) * invDy;
                if (t < 0.0f || t >= bestT) continue;

                float hx = origin.getX() + t * direction.getX();
                float hz = origin.getZ() + t * direction.getZ();
                if (hx * hx + hz * hz <= radius * radius) {
                    bestT = t;
                    bestNormal = (s == 0) ? Vec3(0.0f, -1.0f, 0.0f) : Vec3(0.0f, 1.0f, 0.0f);
                    anyHit = true;
                }
            }
        }

        if (!anyHit) return ShapeRayResult::miss();
        return ShapeRayResult(bestT, bestNormal);
    }
};

} // namespace pulse
