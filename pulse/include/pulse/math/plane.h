/**
 * @file plane.h
 * @brief Plane — SSE4.2 accelerated, stored as normal + distance.
 *
 * Represented as a 4-component vector: (nx, ny, nz, d) where the plane
 * equation is: dot(normal, point) + d = 0. This allows the signed distance
 * from a point to the plane to be computed as a single 4D dot product with
 * the homogeneous point (px, py, pz, 1).
 *
 * Memory layout: [nx, ny, nz, d] — 16 bytes, 16-byte aligned (__m128).
 */

#pragma once

#include "math_common.h"
#include "vec3.h"
#include "vec4.h"

namespace pulse {

/**
 * @struct Plane
 * @brief A plane defined by normal vector and signed distance from origin.
 *
 * Plane equation: dot(normal, point) + d = 0.
 * 16 bytes, 16-byte aligned.
 */
struct PULSE_SIMD_ALIGN Plane {
#ifdef PULSE_SIMD_SSE42
    __m128 m128; ///< [nx, ny, nz, d]
#else
    float nx, ny, nz, d;
#endif

    // ── Constructors ──────────────────────────────────────────────────────

    /// Default: XY plane (normal = +Z, d = 0).
    PULSE_FORCE_INLINE Plane() noexcept
#ifdef PULSE_SIMD_SSE42
        : m128(_mm_set_ps(0.0f, 1.0f, 0.0f, 0.0f))
#else
        : nx(0.0f), ny(0.0f), nz(1.0f), d(0.0f)
#endif
    {}

    /// Construct from normal and distance.
    PULSE_FORCE_INLINE Plane(Vec3 normal, float distance) noexcept
#ifdef PULSE_SIMD_SSE42
        : m128(_mm_blend_ps(normal.m128, _mm_set_ps(distance, 0.0f, 0.0f, 0.0f), 0b1000))
#else
        : nx(normal.getX()), ny(normal.getY()), nz(normal.getZ()), d(distance)
#endif
    {}

    /// Construct from normal and a point on the plane.
    PULSE_FORCE_INLINE Plane(Vec3 normal, Vec3 pointOnPlane) noexcept {
        float dist = -normal.dot(pointOnPlane);
#ifdef PULSE_SIMD_SSE42
        m128 = _mm_blend_ps(normal.m128, _mm_set_ps(dist, 0.0f, 0.0f, 0.0f), 0b1000);
#else
        nx = normal.getX(); ny = normal.getY(); nz = normal.getZ(); d = dist;
#endif
    }

    /// Construct from three points (counter-clockwise winding defines the normal direction).
    [[nodiscard]] static PULSE_FORCE_INLINE Plane fromPoints(Vec3 a, Vec3 b, Vec3 c) noexcept {
        Vec3 normal = (b - a).cross(c - a).normalized();
        return Plane(normal, a);
    }

#ifdef PULSE_SIMD_SSE42
    PULSE_FORCE_INLINE Plane(__m128 v) noexcept : m128(v) {}
#endif

    // ── Accessors ─────────────────────────────────────────────────────────

    /// Get the plane normal.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 normal() const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Vec3(simd::zeroW(m128), Vec3::TrustedTag{});
#else
        return Vec3(nx, ny, nz);
#endif
    }

    /// Get the signed distance from origin.
    [[nodiscard]] PULSE_FORCE_INLINE float distance() const noexcept {
#ifdef PULSE_SIMD_SSE42
        return _mm_cvtss_f32(_mm_shuffle_ps(m128, m128, _MM_SHUFFLE(3, 3, 3, 3)));
#else
        return d;
#endif
    }

    // ── Distance queries ──────────────────────────────────────────────────

    /// Signed distance from a point to the plane.
    /// Positive = same side as normal, negative = opposite side.
    [[nodiscard]] PULSE_FORCE_INLINE float signedDistanceTo(Vec3 point) const noexcept {
#ifdef PULSE_SIMD_SSE42
        // dot(normal, point) + d = dot4([nx,ny,nz,d], [px,py,pz,1])
        __m128 p = _mm_blend_ps(point.m128, _mm_set1_ps(1.0f), 0b1000);
        return _mm_cvtss_f32(_mm_dp_ps(m128, p, 0xF1));
#else
        return nx * point.getX() + ny * point.getY() + nz * point.getZ() + d;
#endif
    }

    /// Absolute distance from a point to the plane.
    [[nodiscard]] PULSE_FORCE_INLINE float distanceTo(Vec3 point) const noexcept {
        return math::fastAbs(signedDistanceTo(point));
    }

    /// Classify which side of the plane a point is on.
    /// Returns: +1 (front/positive), -1 (back/negative), 0 (on the plane).
    [[nodiscard]] PULSE_FORCE_INLINE int classify(Vec3 point, float epsilon = math::Epsilon) const noexcept {
        float dist = signedDistanceTo(point);
        if (dist > epsilon) return 1;
        if (dist < -epsilon) return -1;
        return 0;
    }

    // ── Projection ────────────────────────────────────────────────────────

    /// Project a point onto the plane.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 projectPoint(Vec3 point) const noexcept {
        return point - normal() * signedDistanceTo(point);
    }

    /// Reflect a point across the plane.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 reflectPoint(Vec3 point) const noexcept {
        return point - normal() * (2.0f * signedDistanceTo(point));
    }

    /// Reflect a direction across the plane normal.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 reflectDirection(Vec3 dir) const noexcept {
        return dir.reflect(normal());
    }

    // ── Ray intersection ──────────────────────────────────────────────────

    /// Ray-plane intersection.
    /// @param origin Ray origin.
    /// @param dir Ray direction (does not need to be normalized).
    /// @param t [out] Parametric distance along the ray.
    /// @return True if the ray intersects the plane (not parallel).
    [[nodiscard]] PULSE_FORCE_INLINE bool rayIntersect(Vec3 origin, Vec3 dir, float& t) const noexcept {
        Vec3 n = normal();
        float denom = n.dot(dir);
        if (math::fastAbs(denom) < math::Epsilon) return false; // Parallel
        t = -(n.dot(origin) + distance()) / denom;
        return true;
    }

    // ── Normalize ─────────────────────────────────────────────────────────

    /// Re-normalize the plane (if the normal has become non-unit due to transformations).
    PULSE_FORCE_INLINE void normalize() noexcept {
        Vec3 n = normal();
        float len = n.length();
        if (len < math::Epsilon) return;
        float invLen = 1.0f / len;
#ifdef PULSE_SIMD_SSE42
        m128 = _mm_mul_ps(m128, _mm_set1_ps(invLen));
#else
        nx *= invLen; ny *= invLen; nz *= invLen; d *= invLen;
#endif
    }

    /// Return a normalized copy.
    [[nodiscard]] PULSE_FORCE_INLINE Plane normalized() const noexcept {
        Plane p = *this;
        p.normalize();
        return p;
    }

    // ── Comparison ────────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE bool operator==(const Plane& rhs) const noexcept {
#ifdef PULSE_SIMD_SSE42
        __m128 diff = _mm_sub_ps(m128, rhs.m128);
        __m128 absDiff = _mm_andnot_ps(_mm_set1_ps(-0.0f), diff);
        __m128 eps = _mm_set1_ps(math::Epsilon);
        __m128 cmp = _mm_cmple_ps(absDiff, eps);
        return _mm_movemask_ps(cmp) == 0xF;
#else
        return math::approxEqual(nx, rhs.nx) && math::approxEqual(ny, rhs.ny) &&
               math::approxEqual(nz, rhs.nz) && math::approxEqual(d, rhs.d);
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE bool operator!=(const Plane& rhs) const noexcept {
        return !(*this == rhs);
    }
};

} // namespace pulse
