/**
 * @file box.h
 * @brief Box collision shape — axis-aligned half-extents, rotated via Transform for OBB.
 *
 * Stored as Vec3 half-extents. In local space the box spans [-hx,hx] × [-hy,hy] × [-hz,hz].
 * Applied with a Transform, it becomes an Oriented Bounding Box (OBB).
 *
 * The support function for GJK is component-wise sign selection — extremely fast
 * with SIMD bitwise ops.
 *
 * Memory: 16 bytes (Vec3 halfExtents).
 */

#pragma once

#include "shape_common.h"
#include <pulse/math/transform.h>
#include <pulse/math/aabb.h>
#include <pulse/math/ray.h>

namespace pulse {

/**
 * @struct Box
 * @brief A box defined by half-extents centered at the local-space origin.
 *
 * Full extents = 2 × halfExtents. Rotated via Transform for OBB behavior.
 */
struct PULSE_SIMD_ALIGN Box {
    Vec3 halfExtents; ///< Half-size along each local axis. All components must be > 0.

    static constexpr ShapeType Type = ShapeType::Box;

    // ── Constructors ──────────────────────────────────────────────────────

    /// Default: unit cube (halfExtents = 0.5 on each axis).
    PULSE_FORCE_INLINE Box() noexcept : halfExtents(0.5f, 0.5f, 0.5f) {}

    /// Construct from half-extents.
    explicit PULSE_FORCE_INLINE Box(Vec3 he) noexcept : halfExtents(he) {}

    /// Construct from individual half-extents.
    PULSE_FORCE_INLINE Box(float hx, float hy, float hz) noexcept
        : halfExtents(hx, hy, hz)
    {}

    // ── AABB computation ──────────────────────────────────────────────────

    /// Compute world-space AABB for a rotated box.
    /// Projects the rotated half-extents onto each world axis.
    [[nodiscard]] PULSE_FORCE_INLINE AABB computeAABB(const Transform& tx) const noexcept {
        // Get rotation as Mat3
        Mat3 rot = tx.toMat3();

        // For each world axis, the extent is the sum of abs(dot(axis, rotated_local_axis)) * halfExtent
        // This is equivalent to: sum_j |R_ij| * halfExtent_j for each world axis i
        float hx = halfExtents.getX();
        float hy = halfExtents.getY();
        float hz = halfExtents.getZ();

        // Absolute rotation matrix values
        float ax = math::fastAbs(rot[0].getX()) * hx + math::fastAbs(rot[0].getY()) * hy + math::fastAbs(rot[0].getZ()) * hz;
        float ay = math::fastAbs(rot[1].getX()) * hx + math::fastAbs(rot[1].getY()) * hy + math::fastAbs(rot[1].getZ()) * hz;
        float az = math::fastAbs(rot[2].getX()) * hx + math::fastAbs(rot[2].getY()) * hy + math::fastAbs(rot[2].getZ()) * hz;

        Vec3 worldExtent(ax, ay, az);
        return AABB(tx.position - worldExtent, tx.position + worldExtent);
    }

    /// Compute local-space AABB (axis-aligned, centered at origin).
    [[nodiscard]] PULSE_FORCE_INLINE AABB computeLocalAABB() const noexcept {
        return AABB(-halfExtents, halfExtents);
    }

    // ── Mass properties ───────────────────────────────────────────────────

    /// Compute mass properties for a solid box with given density.
    /// Volume = 8·hx·hy·hz
    /// Inertia (about CoM): Ixx = m/12·((2hy)² + (2hz)²), etc.
    [[nodiscard]] PULSE_FORCE_INLINE MassProperties computeMass(float density) const noexcept {
        float hx = halfExtents.getX();
        float hy = halfExtents.getY();
        float hz = halfExtents.getZ();

        float volume = 8.0f * hx * hy * hz;
        float m = density * volume;

        // Full side lengths
        float sx2 = 4.0f * hx * hx; // (2hx)²
        float sy2 = 4.0f * hy * hy;
        float sz2 = 4.0f * hz * hz;

        float k = m / 12.0f;
        float ixx = k * (sy2 + sz2);
        float iyy = k * (sx2 + sz2);
        float izz = k * (sx2 + sy2);

        return MassProperties(
            m,
            Vec3::zero(),
            Mat3(ixx, 0.0f, 0.0f,
                 0.0f, iyy, 0.0f,
                 0.0f, 0.0f, izz)
        );
    }

    // ── GJK support function ──────────────────────────────────────────────

    /// Furthest point on the box in the given direction (local space).
    /// For a box, this is simply sign(dir) * halfExtents per component.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 support(Vec3 direction) const noexcept {
#ifdef PULSE_SIMD_SSE42
        // Extract signs of direction and apply to halfExtents
        __m128 signMask = _mm_set1_ps(-0.0f); // Sign bit mask
        __m128 signs = _mm_and_ps(direction.m128, signMask);
        __m128 result = _mm_or_ps(halfExtents.m128, signs);
        return Vec3(result);
#else
        return Vec3(
            direction.getX() >= 0.0f ? halfExtents.getX() : -halfExtents.getX(),
            direction.getY() >= 0.0f ? halfExtents.getY() : -halfExtents.getY(),
            direction.getZ() >= 0.0f ? halfExtents.getZ() : -halfExtents.getZ()
        );
#endif
    }

    /// Furthest point in the given direction (world space).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 supportWorld(Vec3 direction, const Transform& tx) const noexcept {
        // Transform direction to local space
        Vec3 localDir = tx.inverseTransformDirection(direction);
        // Get local support point
        Vec3 localSupport = support(localDir);
        // Transform back to world space
        return tx.transformPoint(localSupport);
    }

    // ── Point containment ─────────────────────────────────────────────────

    /// Test if a point (local space) is inside the box.
    [[nodiscard]] PULSE_FORCE_INLINE bool containsPoint(Vec3 point) const noexcept {
#ifdef PULSE_SIMD_SSE42
        // |point| <= halfExtents per component
        __m128 signMask = _mm_set1_ps(-0.0f);
        __m128 absPoint = _mm_andnot_ps(signMask, point.m128);
        __m128 cmp = _mm_cmple_ps(absPoint, halfExtents.m128);
        return (_mm_movemask_ps(cmp) & 0x7) == 0x7;
#else
        return math::fastAbs(point.getX()) <= halfExtents.getX() &&
               math::fastAbs(point.getY()) <= halfExtents.getY() &&
               math::fastAbs(point.getZ()) <= halfExtents.getZ();
#endif
    }

    // ── Closest point ─────────────────────────────────────────────────────

    /// Closest point on (or inside) the box to a query point (local space).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 closestPoint(Vec3 point) const noexcept {
        // Clamp each component to [-halfExtent, +halfExtent]
        return point.max(-halfExtents).min(halfExtents);
    }

    // ── Ray intersection ──────────────────────────────────────────────────

    /// Ray-box intersection in local space using the slab method.
    [[nodiscard]] PULSE_FORCE_INLINE ShapeRayResult rayIntersect(Vec3 origin, Vec3 direction) const noexcept {
        // Compute reciprocal direction — use 1/d directly.
        // The slab method correctly handles ±infinity for zero components.
        Vec3 invDir(
            1.0f / direction.getX(),
            1.0f / direction.getY(),
            1.0f / direction.getZ()
        );

        // Use AABB ray intersection
        AABB localBox(-halfExtents, halfExtents);
        float tMin, tMax;
        if (!localBox.rayIntersect(origin, invDir, tMin, tMax)) {
            return ShapeRayResult::miss();
        }

        float t = tMin >= 0.0f ? tMin : tMax;
        if (t < 0.0f) {
            return ShapeRayResult::miss();
        }

        // Compute hit normal — find which face was hit
        Vec3 hitPoint = origin + direction * t;
        Vec3 localNormal = Vec3::zero();

        // Find the axis with the smallest distance to a face
        float dx = math::fastAbs(math::fastAbs(hitPoint.getX()) - halfExtents.getX());
        float dy = math::fastAbs(math::fastAbs(hitPoint.getY()) - halfExtents.getY());
        float dz = math::fastAbs(math::fastAbs(hitPoint.getZ()) - halfExtents.getZ());

        if (dx < dy && dx < dz) {
            localNormal = Vec3(math::sign(hitPoint.getX()), 0.0f, 0.0f);
        } else if (dy < dz) {
            localNormal = Vec3(0.0f, math::sign(hitPoint.getY()), 0.0f);
        } else {
            localNormal = Vec3(0.0f, 0.0f, math::sign(hitPoint.getZ()));
        }

        return ShapeRayResult(t, localNormal);
    }

    // ── Vertex enumeration ────────────────────────────────────────────────

    /// Get the 8 corner vertices of the box in local space.
    /// Output array must have capacity for at least 8 Vec3s.
    PULSE_FORCE_INLINE void getVertices(Vec3* out) const noexcept {
        float hx = halfExtents.getX();
        float hy = halfExtents.getY();
        float hz = halfExtents.getZ();
        out[0] = Vec3(-hx, -hy, -hz);
        out[1] = Vec3( hx, -hy, -hz);
        out[2] = Vec3(-hx,  hy, -hz);
        out[3] = Vec3( hx,  hy, -hz);
        out[4] = Vec3(-hx, -hy,  hz);
        out[5] = Vec3( hx, -hy,  hz);
        out[6] = Vec3(-hx,  hy,  hz);
        out[7] = Vec3( hx,  hy,  hz);
    }
};

} // namespace pulse
