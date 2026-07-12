/**
 * @file vec3.h
 * @brief 3D vector — SSE4.2 accelerated, 16-byte aligned.
 *
 * Internally stored as __m128 with w = 0.0f. This allows all Vec3 operations
 * to use SSE intrinsics (dot, cross, normalize, add, mul, etc.) with zero
 * conversion overhead. The 4th float (w) is always kept at 0.0f and is
 * never exposed through the public API.
 *
 * Memory layout: [x, y, z, 0.0f] — 16 bytes, 16-byte aligned.
 * This matches cache-line boundaries when stored in arrays (4 Vec3s = 64 bytes = 1 cache line).
 */

#pragma once

#include "math_common.h"

namespace pulse {

struct Vec4; // Forward declaration

/**
 * @struct Vec3
 * @brief A 3-component floating-point vector backed by __m128 (SSE).
 *
 * The w component is always 0.0f internally. All operations use SIMD intrinsics
 * on supported platforms, with a scalar fallback path.
 */
struct PULSE_SIMD_ALIGN Vec3 {
#ifdef PULSE_SIMD_SSE42
    __m128 m128; ///< Internal SIMD register. w is always 0.0f.
#else
    float x, y, z;
    float _pad; // Maintain 16-byte size for ABI compatibility
#endif

    // ── Constructors ──────────────────────────────────────────────────────

    /// Default: zero-initialized.
    PULSE_FORCE_INLINE Vec3() noexcept
#ifdef PULSE_SIMD_SSE42
        : m128(_mm_setzero_ps())
#else
        : x(0.0f), y(0.0f), z(0.0f), _pad(0.0f)
#endif
    {}

    /// Broadcast scalar to all 3 components.
    explicit PULSE_FORCE_INLINE Vec3(float s) noexcept
#ifdef PULSE_SIMD_SSE42
        : m128(_mm_set_ps(0.0f, s, s, s))
#else
        : x(s), y(s), z(s), _pad(0.0f)
#endif
    {}

    /// Component-wise construction.
    PULSE_FORCE_INLINE Vec3(float x_, float y_, float z_) noexcept
#ifdef PULSE_SIMD_SSE42
        : m128(_mm_set_ps(0.0f, z_, y_, x_))
#else
        : x(x_), y(y_), z(z_), _pad(0.0f)
#endif
    {}

#ifdef PULSE_SIMD_SSE42
    /// Construct directly from an __m128 register. Caller must ensure w = 0.
    PULSE_FORCE_INLINE Vec3(__m128 v) noexcept : m128(simd::zeroW(v)) {}

    /// Construct from __m128 without zeroing w (trusted internal use).
    struct TrustedTag {};
    PULSE_FORCE_INLINE Vec3(__m128 v, TrustedTag) noexcept : m128(v) {}
#endif

    // ── Static factories ──────────────────────────────────────────────────

    [[nodiscard]] static PULSE_FORCE_INLINE Vec3 zero()  noexcept { return Vec3(0.0f, 0.0f, 0.0f); }
    [[nodiscard]] static PULSE_FORCE_INLINE Vec3 one()   noexcept { return Vec3(1.0f, 1.0f, 1.0f); }
    [[nodiscard]] static PULSE_FORCE_INLINE Vec3 unitX() noexcept { return Vec3(1.0f, 0.0f, 0.0f); }
    [[nodiscard]] static PULSE_FORCE_INLINE Vec3 unitY() noexcept { return Vec3(0.0f, 1.0f, 0.0f); }
    [[nodiscard]] static PULSE_FORCE_INLINE Vec3 unitZ() noexcept { return Vec3(0.0f, 0.0f, 1.0f); }
    [[nodiscard]] static PULSE_FORCE_INLINE Vec3 up()    noexcept { return Vec3(0.0f, 1.0f, 0.0f); }
    [[nodiscard]] static PULSE_FORCE_INLINE Vec3 right() noexcept { return Vec3(1.0f, 0.0f, 0.0f); }
    [[nodiscard]] static PULSE_FORCE_INLINE Vec3 forward() noexcept { return Vec3(0.0f, 0.0f, -1.0f); }

    // ── Element access ────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE float getX() const noexcept {
#ifdef PULSE_SIMD_SSE42
        return _mm_cvtss_f32(m128);
#else
        return x;
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE float getY() const noexcept {
#ifdef PULSE_SIMD_SSE42
        return _mm_cvtss_f32(_mm_shuffle_ps(m128, m128, _MM_SHUFFLE(1, 1, 1, 1)));
#else
        return y;
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE float getZ() const noexcept {
#ifdef PULSE_SIMD_SSE42
        return _mm_cvtss_f32(_mm_shuffle_ps(m128, m128, _MM_SHUFFLE(2, 2, 2, 2)));
#else
        return z;
#endif
    }

    PULSE_FORCE_INLINE void setX(float v) noexcept {
#ifdef PULSE_SIMD_SSE42
        m128 = _mm_blend_ps(m128, _mm_set_ss(v), 0b0001);
#else
        x = v;
#endif
    }

    PULSE_FORCE_INLINE void setY(float v) noexcept {
#ifdef PULSE_SIMD_SSE42
        __m128 tmp = _mm_set_ps(0.0f, 0.0f, v, 0.0f);
        m128 = _mm_blend_ps(m128, tmp, 0b0010);
#else
        y = v;
#endif
    }

    PULSE_FORCE_INLINE void setZ(float v) noexcept {
#ifdef PULSE_SIMD_SSE42
        __m128 tmp = _mm_set_ps(0.0f, v, 0.0f, 0.0f);
        m128 = _mm_blend_ps(m128, tmp, 0b0100);
#else
        z = v;
#endif
    }

    /// Array-style access (read-only, extracts from SIMD register).
    [[nodiscard]] PULSE_FORCE_INLINE float operator[](int i) const noexcept {
#ifdef PULSE_SIMD_SSE42
        PULSE_SIMD_ALIGN float tmp[4];
        _mm_store_ps(tmp, m128);
        return tmp[i];
#else
        return (&x)[i];
#endif
    }

    // ── Arithmetic operators ──────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE Vec3 operator+(Vec3 rhs) const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Vec3(_mm_add_ps(m128, rhs.m128), TrustedTag{});
#else
        return {x + rhs.x, y + rhs.y, z + rhs.z};
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec3 operator-(Vec3 rhs) const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Vec3(_mm_sub_ps(m128, rhs.m128), TrustedTag{});
#else
        return {x - rhs.x, y - rhs.y, z - rhs.z};
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec3 operator*(float s) const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Vec3(_mm_mul_ps(m128, _mm_set1_ps(s)), TrustedTag{});
#else
        return {x * s, y * s, z * s};
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec3 operator/(float s) const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Vec3(_mm_div_ps(m128, _mm_set1_ps(s)), TrustedTag{});
#else
        const float inv = 1.0f / s;
        return {x * inv, y * inv, z * inv};
#endif
    }

    /// Component-wise multiplication.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 operator*(Vec3 rhs) const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Vec3(_mm_mul_ps(m128, rhs.m128), TrustedTag{});
#else
        return {x * rhs.x, y * rhs.y, z * rhs.z};
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec3 operator-() const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Vec3(_mm_sub_ps(_mm_setzero_ps(), m128), TrustedTag{});
#else
        return {-x, -y, -z};
#endif
    }

    PULSE_FORCE_INLINE Vec3& operator+=(Vec3 rhs) noexcept {
#ifdef PULSE_SIMD_SSE42
        m128 = _mm_add_ps(m128, rhs.m128);
#else
        x += rhs.x; y += rhs.y; z += rhs.z;
#endif
        return *this;
    }

    PULSE_FORCE_INLINE Vec3& operator-=(Vec3 rhs) noexcept {
#ifdef PULSE_SIMD_SSE42
        m128 = _mm_sub_ps(m128, rhs.m128);
#else
        x -= rhs.x; y -= rhs.y; z -= rhs.z;
#endif
        return *this;
    }

    PULSE_FORCE_INLINE Vec3& operator*=(float s) noexcept {
#ifdef PULSE_SIMD_SSE42
        m128 = _mm_mul_ps(m128, _mm_set1_ps(s));
#else
        x *= s; y *= s; z *= s;
#endif
        return *this;
    }

    PULSE_FORCE_INLINE Vec3& operator/=(float s) noexcept {
#ifdef PULSE_SIMD_SSE42
        m128 = _mm_div_ps(m128, _mm_set1_ps(s));
#else
        const float inv = 1.0f / s;
        x *= inv; y *= inv; z *= inv;
#endif
        return *this;
    }

    // ── Comparison ────────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE bool operator==(Vec3 rhs) const noexcept {
#ifdef PULSE_SIMD_SSE42
        __m128 diff = _mm_sub_ps(m128, rhs.m128);
        // Absolute value of difference
        __m128 absDiff = _mm_andnot_ps(_mm_set1_ps(-0.0f), diff);
        __m128 eps = _mm_set1_ps(math::Epsilon);
        __m128 cmp = _mm_cmple_ps(absDiff, eps);
        // Check x, y, z (ignore w) — mask 0x7 = bits 0,1,2
        return (_mm_movemask_ps(cmp) & 0x7) == 0x7;
#else
        return math::approxEqual(x, rhs.x) && math::approxEqual(y, rhs.y) && math::approxEqual(z, rhs.z);
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE bool operator!=(Vec3 rhs) const noexcept {
        return !(*this == rhs);
    }

    // ── Length / Normalization ─────────────────────────────────────────────

    /// Squared length (no sqrt).
    [[nodiscard]] PULSE_FORCE_INLINE float lengthSq() const noexcept {
#ifdef PULSE_SIMD_SSE42
        return simd::dot3(m128, m128);
#else
        return x * x + y * y + z * z;
#endif
    }

    /// Euclidean length.
    [[nodiscard]] PULSE_FORCE_INLINE float length() const noexcept {
#ifdef PULSE_SIMD_SSE42
        return simd::length3(m128);
#else
        return math::fastSqrt(lengthSq());
#endif
    }

    /// Returns a normalized copy. Returns zero vector if length is near-zero.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 normalized() const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Vec3(simd::normalize3(m128), TrustedTag{});
#else
        const float lsq = lengthSq();
        if (lsq < math::Epsilon * math::Epsilon) return Vec3::zero();
        const float inv = math::fastInvSqrt(lsq);
        return {x * inv, y * inv, z * inv};
#endif
    }

    /// Normalizes in place.
    PULSE_FORCE_INLINE void normalize() noexcept {
        *this = normalized();
    }

    /// Check if approximately unit length.
    [[nodiscard]] PULSE_FORCE_INLINE bool isNormalized(float tolerance = 0.001f) const noexcept {
        return math::approxEqual(lengthSq(), 1.0f, tolerance);
    }

    // ── Dot / Cross ───────────────────────────────────────────────────────

    /// Dot product.
    [[nodiscard]] PULSE_FORCE_INLINE float dot(Vec3 rhs) const noexcept {
#ifdef PULSE_SIMD_SSE42
        return simd::dot3(m128, rhs.m128);
#else
        return x * rhs.x + y * rhs.y + z * rhs.z;
#endif
    }

    /// Cross product.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 cross(Vec3 rhs) const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Vec3(simd::cross3(m128, rhs.m128), TrustedTag{});
#else
        return {
            y * rhs.z - z * rhs.y,
            z * rhs.x - x * rhs.z,
            x * rhs.y - y * rhs.x
        };
#endif
    }

    // ── Utilities ─────────────────────────────────────────────────────────

    /// Distance to another point.
    [[nodiscard]] PULSE_FORCE_INLINE float distanceTo(Vec3 other) const noexcept {
        return (*this - other).length();
    }

    /// Squared distance to another point.
    [[nodiscard]] PULSE_FORCE_INLINE float distanceSqTo(Vec3 other) const noexcept {
        return (*this - other).lengthSq();
    }

    /// Reflect about a normal.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 reflect(Vec3 normal) const noexcept {
        return *this - normal * (2.0f * dot(normal));
    }

    /// Project this vector onto another vector.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 projectOnto(Vec3 other) const noexcept {
        const float d = other.dot(other);
        if (d < math::Epsilon) return Vec3::zero();
        return other * (dot(other) / d);
    }

    /// Linear interpolation towards another vector.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 lerp(Vec3 target, float t) const noexcept {
#ifdef PULSE_SIMD_SSE42
        __m128 vt = _mm_set1_ps(t);
        __m128 diff = _mm_sub_ps(target.m128, m128);
        return Vec3(_mm_add_ps(m128, _mm_mul_ps(diff, vt)), TrustedTag{});
#else
        return {
            math::lerp(x, target.x, t),
            math::lerp(y, target.y, t),
            math::lerp(z, target.z, t)
        };
#endif
    }

    /// Component-wise min.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 min(Vec3 rhs) const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Vec3(_mm_min_ps(m128, rhs.m128), TrustedTag{});
#else
        return {
            (x < rhs.x) ? x : rhs.x,
            (y < rhs.y) ? y : rhs.y,
            (z < rhs.z) ? z : rhs.z
        };
#endif
    }

    /// Component-wise max.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 max(Vec3 rhs) const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Vec3(_mm_max_ps(m128, rhs.m128), TrustedTag{});
#else
        return {
            (x > rhs.x) ? x : rhs.x,
            (y > rhs.y) ? y : rhs.y,
            (z > rhs.z) ? z : rhs.z
        };
#endif
    }

    /// Component-wise absolute value.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 abs() const noexcept {
#ifdef PULSE_SIMD_SSE42
        __m128 signMask = _mm_set1_ps(-0.0f);
        return Vec3(_mm_andnot_ps(signMask, m128), TrustedTag{});
#else
        return {math::fastAbs(x), math::fastAbs(y), math::fastAbs(z)};
#endif
    }

    /// Clamp each component between [lo, hi].
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 clamped(float lo, float hi) const noexcept {
#ifdef PULSE_SIMD_SSE42
        __m128 vlo = _mm_set1_ps(lo);
        __m128 vhi = _mm_set1_ps(hi);
        return Vec3(_mm_min_ps(_mm_max_ps(m128, vlo), vhi), TrustedTag{});
#else
        return {
            math::clamp(x, lo, hi),
            math::clamp(y, lo, hi),
            math::clamp(z, lo, hi)
        };
#endif
    }

    /// Triple scalar product: this . (b x c).
    [[nodiscard]] PULSE_FORCE_INLINE float tripleProduct(Vec3 b, Vec3 c) const noexcept {
        return dot(b.cross(c));
    }
};

// ── Free-function operators ───────────────────────────────────────────────────

[[nodiscard]] PULSE_FORCE_INLINE Vec3 operator*(float s, Vec3 v) noexcept {
    return v * s;
}

// ── Free-function utilities ───────────────────────────────────────────────────

namespace math {
    [[nodiscard]] PULSE_FORCE_INLINE float dot(Vec3 a, Vec3 b) noexcept {
        return a.dot(b);
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec3 cross(Vec3 a, Vec3 b) noexcept {
        return a.cross(b);
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec3 normalize(Vec3 v) noexcept {
        return v.normalized();
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec3 lerp(Vec3 a, Vec3 b, float t) noexcept {
        return a.lerp(b, t);
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec3 reflect(Vec3 v, Vec3 normal) noexcept {
        return v.reflect(normal);
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec3 project(Vec3 v, Vec3 onto) noexcept {
        return v.projectOnto(onto);
    }

    /// Compute an orthonormal basis from a single direction vector.
    /// Given 'n' (assumed normalized), computes two perpendicular unit vectors 't' and 'b'.
    PULSE_FORCE_INLINE void orthonormalBasis(Vec3 n, Vec3& t, Vec3& b) noexcept {
        // Frisvad's method (robust, branchless for most inputs)
        if (n.getZ() < -0.9999999f) {
            t = Vec3(0.0f, -1.0f, 0.0f);
            b = Vec3(-1.0f, 0.0f, 0.0f);
            return;
        }
        const float a = 1.0f / (1.0f + n.getZ());
        const float bv = -n.getX() * n.getY() * a;
        t = Vec3(1.0f - n.getX() * n.getX() * a, bv, -n.getX());
        b = Vec3(bv, 1.0f - n.getY() * n.getY() * a, -n.getY());
    }
} // namespace math

} // namespace pulse
