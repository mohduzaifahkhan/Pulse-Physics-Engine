/**
 * @file aabb.h
 * @brief Axis-Aligned Bounding Box — SSE4.2 accelerated.
 *
 * Stored as min/max Vec3 pair (2× __m128 = 32 bytes). This is the fundamental
 * bounding volume for the broad-phase collision detection. All overlap tests,
 * union, intersection, ray intersection, and expansion operations use SIMD.
 *
 * Memory layout: [min.x, min.y, min.z, 0, max.x, max.y, max.z, 0] — 32 bytes, 16-byte aligned.
 */

#pragma once

#include "math_common.h"
#include "vec3.h"

namespace pulse {

/**
 * @struct AABB
 * @brief An axis-aligned bounding box defined by min and max corners.
 *
 * 32 bytes, 16-byte aligned. Two AABBs fit in one cache line.
 */
struct PULSE_SIMD_ALIGN AABB {
    Vec3 min; ///< Minimum corner.
    Vec3 max; ///< Maximum corner.

    // ── Constructors ──────────────────────────────────────────────────────

    /// Default: invalid (inverted) AABB — ready for expansion.
    PULSE_FORCE_INLINE AABB() noexcept
        : min(Vec3( math::Infinity,  math::Infinity,  math::Infinity)),
          max(Vec3(math::NegInfinity, math::NegInfinity, math::NegInfinity))
    {}

    /// Construct from explicit min/max corners.
    PULSE_FORCE_INLINE AABB(Vec3 minCorner, Vec3 maxCorner) noexcept
        : min(minCorner), max(maxCorner)
    {}

    /// Construct from center and half-extents.
    [[nodiscard]] static PULSE_FORCE_INLINE AABB fromCenterExtents(Vec3 center, Vec3 halfExtents) noexcept {
        return AABB(center - halfExtents, center + halfExtents);
    }

    /// Construct a zero-volume AABB at a single point.
    [[nodiscard]] static PULSE_FORCE_INLINE AABB fromPoint(Vec3 point) noexcept {
        return AABB(point, point);
    }

    /// Construct from a sphere (center + radius).
    [[nodiscard]] static PULSE_FORCE_INLINE AABB fromSphere(Vec3 center, float radius) noexcept {
        Vec3 r(radius);
        return AABB(center - r, center + r);
    }

    // ── Queries ───────────────────────────────────────────────────────────

    /// Center point.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 center() const noexcept {
        return (min + max) * 0.5f;
    }

    /// Half-extents (half the size along each axis).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 halfExtents() const noexcept {
        return (max - min) * 0.5f;
    }

    /// Full extents (size along each axis).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 extents() const noexcept {
        return max - min;
    }

    /// Surface area (used for SAH heuristics in BVH).
    [[nodiscard]] PULSE_FORCE_INLINE float surfaceArea() const noexcept {
        Vec3 d = max - min;
        float dx = d.getX(), dy = d.getY(), dz = d.getZ();
        return 2.0f * (dx * dy + dy * dz + dz * dx);
    }

    /// Volume.
    [[nodiscard]] PULSE_FORCE_INLINE float volume() const noexcept {
        Vec3 d = max - min;
        return d.getX() * d.getY() * d.getZ();
    }

    /// Check if the AABB is valid (min <= max on all axes).
    [[nodiscard]] PULSE_FORCE_INLINE bool isValid() const noexcept {
#ifdef PULSE_SIMD_SSE42
        __m128 cmp = _mm_cmple_ps(min.m128, max.m128);
        return (_mm_movemask_ps(cmp) & 0x7) == 0x7;
#else
        return min.getX() <= max.getX() && min.getY() <= max.getY() && min.getZ() <= max.getZ();
#endif
    }

    /// Longest axis: 0=x, 1=y, 2=z.
    [[nodiscard]] PULSE_FORCE_INLINE int longestAxis() const noexcept {
        Vec3 d = extents();
        float dx = d.getX(), dy = d.getY(), dz = d.getZ();
        if (dx >= dy && dx >= dz) return 0;
        if (dy >= dz) return 1;
        return 2;
    }

    // ── Overlap / Containment tests ───────────────────────────────────────

    /// Test overlap with another AABB (inclusive boundaries).
    [[nodiscard]] PULSE_FORCE_INLINE bool overlaps(const AABB& other) const noexcept {
#ifdef PULSE_SIMD_SSE42
        // Overlap iff: this.min <= other.max AND other.min <= this.max (all axes)
        __m128 cmp1 = _mm_cmple_ps(min.m128, other.max.m128);
        __m128 cmp2 = _mm_cmple_ps(other.min.m128, max.m128);
        __m128 both = _mm_and_ps(cmp1, cmp2);
        return (_mm_movemask_ps(both) & 0x7) == 0x7;
#else
        return (min.getX() <= other.max.getX() && max.getX() >= other.min.getX()) &&
               (min.getY() <= other.max.getY() && max.getY() >= other.min.getY()) &&
               (min.getZ() <= other.max.getZ() && max.getZ() >= other.min.getZ());
#endif
    }

    /// Test if this AABB fully contains another.
    [[nodiscard]] PULSE_FORCE_INLINE bool contains(const AABB& other) const noexcept {
#ifdef PULSE_SIMD_SSE42
        __m128 cmp1 = _mm_cmple_ps(min.m128, other.min.m128);
        __m128 cmp2 = _mm_cmple_ps(other.max.m128, max.m128);
        __m128 both = _mm_and_ps(cmp1, cmp2);
        return (_mm_movemask_ps(both) & 0x7) == 0x7;
#else
        return (min.getX() <= other.min.getX() && max.getX() >= other.max.getX()) &&
               (min.getY() <= other.min.getY() && max.getY() >= other.max.getY()) &&
               (min.getZ() <= other.min.getZ() && max.getZ() >= other.max.getZ());
#endif
    }

    /// Test if a point is inside this AABB.
    [[nodiscard]] PULSE_FORCE_INLINE bool containsPoint(Vec3 point) const noexcept {
#ifdef PULSE_SIMD_SSE42
        __m128 cmp1 = _mm_cmple_ps(min.m128, point.m128);
        __m128 cmp2 = _mm_cmple_ps(point.m128, max.m128);
        __m128 both = _mm_and_ps(cmp1, cmp2);
        return (_mm_movemask_ps(both) & 0x7) == 0x7;
#else
        return (point.getX() >= min.getX() && point.getX() <= max.getX()) &&
               (point.getY() >= min.getY() && point.getY() <= max.getY()) &&
               (point.getZ() >= min.getZ() && point.getZ() <= max.getZ());
#endif
    }

    // ── Expansion / Combination ───────────────────────────────────────────

    /// Return the union of this AABB with another.
    [[nodiscard]] PULSE_FORCE_INLINE AABB merged(const AABB& other) const noexcept {
        return AABB(min.min(other.min), max.max(other.max));
    }

    /// Return the intersection of this AABB with another.
    [[nodiscard]] PULSE_FORCE_INLINE AABB intersection(const AABB& other) const noexcept {
        return AABB(min.max(other.min), max.min(other.max));
    }

    /// Expand the AABB to include a point.
    PULSE_FORCE_INLINE void expandToInclude(Vec3 point) noexcept {
        min = min.min(point);
        max = max.max(point);
    }

    /// Expand the AABB to include another AABB.
    PULSE_FORCE_INLINE void expandToInclude(const AABB& other) noexcept {
        min = min.min(other.min);
        max = max.max(other.max);
    }

    /// Expand uniformly by a margin (fatten).
    [[nodiscard]] PULSE_FORCE_INLINE AABB expanded(float margin) const noexcept {
        Vec3 m(margin);
        return AABB(min - m, max + m);
    }

    /// Expand the AABB by a velocity vector (for CCD sweeps).
    /// Extends the AABB in the direction of movement.
    [[nodiscard]] PULSE_FORCE_INLINE AABB swept(Vec3 velocity) const noexcept {
        AABB result = *this;
        // Extend min for negative velocity components, max for positive
        Vec3 vMin = velocity.min(Vec3::zero());
        Vec3 vMax = velocity.max(Vec3::zero());
        result.min = result.min + vMin;
        result.max = result.max + vMax;
        return result;
    }

    // ── Ray intersection ──────────────────────────────────────────────────

    /// Ray-AABB intersection test using the slab method.
    /// @param origin Ray origin.
    /// @param invDir Precomputed reciprocal of ray direction (1.0/dir for each axis).
    /// @param tMin [out] Parametric entry distance.
    /// @param tMax [out] Parametric exit distance.
    /// @return True if the ray intersects the AABB (and tMin <= tMax and tMax >= 0).
    [[nodiscard]] PULSE_FORCE_INLINE bool rayIntersect(
        Vec3 origin, Vec3 invDir, float& tMin, float& tMax
    ) const noexcept {
#ifdef PULSE_SIMD_SSE42
        __m128 t1 = _mm_mul_ps(_mm_sub_ps(min.m128, origin.m128), invDir.m128);
        __m128 t2 = _mm_mul_ps(_mm_sub_ps(max.m128, origin.m128), invDir.m128);

        __m128 tmin_v = _mm_min_ps(t1, t2);
        __m128 tmax_v = _mm_max_ps(t1, t2);

        // Horizontal max of tmin_v (x, y, z)
        // We want the largest of the three entry distances
        __m128 tmin_yz = _mm_shuffle_ps(tmin_v, tmin_v, _MM_SHUFFLE(3, 2, 2, 1));
        tmin_v = _mm_max_ps(tmin_v, tmin_yz);
        __m128 tmin_z = _mm_shuffle_ps(tmin_v, tmin_v, _MM_SHUFFLE(3, 3, 3, 2));
        tmin_v = _mm_max_ss(tmin_v, tmin_z);

        // Horizontal min of tmax_v (x, y, z)
        __m128 tmax_yz = _mm_shuffle_ps(tmax_v, tmax_v, _MM_SHUFFLE(3, 2, 2, 1));
        tmax_v = _mm_min_ps(tmax_v, tmax_yz);
        __m128 tmax_z = _mm_shuffle_ps(tmax_v, tmax_v, _MM_SHUFFLE(3, 3, 3, 2));
        tmax_v = _mm_min_ss(tmax_v, tmax_z);

        tMin = _mm_cvtss_f32(tmin_v);
        tMax = _mm_cvtss_f32(tmax_v);

        return tMax >= math::fastMax(tMin, 0.0f);
#else
        float t1x = (min.getX() - origin.getX()) * invDir.getX();
        float t2x = (max.getX() - origin.getX()) * invDir.getX();
        float t1y = (min.getY() - origin.getY()) * invDir.getY();
        float t2y = (max.getY() - origin.getY()) * invDir.getY();
        float t1z = (min.getZ() - origin.getZ()) * invDir.getZ();
        float t2z = (max.getZ() - origin.getZ()) * invDir.getZ();

        tMin = math::fastMax(math::fastMax(
            math::fastMin(t1x, t2x), math::fastMin(t1y, t2y)),
            math::fastMin(t1z, t2z));
        tMax = math::fastMin(math::fastMin(
            math::fastMax(t1x, t2x), math::fastMax(t1y, t2y)),
            math::fastMax(t1z, t2z));

        return tMax >= math::fastMax(tMin, 0.0f);
#endif
    }

    // ── Closest point ─────────────────────────────────────────────────────

    /// Closest point on the AABB surface to a given point.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 closestPoint(Vec3 point) const noexcept {
        return point.max(min).min(max);
    }

    /// Squared distance from a point to the AABB.
    [[nodiscard]] PULSE_FORCE_INLINE float distanceSqTo(Vec3 point) const noexcept {
        Vec3 closest = closestPoint(point);
        return (point - closest).lengthSq();
    }

    // ── Comparison ────────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE bool operator==(const AABB& rhs) const noexcept {
        return min == rhs.min && max == rhs.max;
    }

    [[nodiscard]] PULSE_FORCE_INLINE bool operator!=(const AABB& rhs) const noexcept {
        return !(*this == rhs);
    }
};

} // namespace pulse
