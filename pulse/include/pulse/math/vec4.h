/**
 * @file vec4.h
 * @brief 4D vector — SSE4.2 accelerated, 16-byte aligned.
 *
 * Full 4-component vector using __m128 storage. All four components (x, y, z, w)
 * are active and participate in all operations. Used for homogeneous coordinates,
 * color (RGBA), quaternion math internals, and Mat4 row operations.
 *
 * Memory layout: [x, y, z, w] — 16 bytes, 16-byte aligned.
 */

#pragma once

#include "math_common.h"
#include "vec3.h"

namespace pulse {

/**
 * @struct Vec4
 * @brief A 4-component floating-point vector backed by __m128.
 */
struct PULSE_SIMD_ALIGN Vec4 {
#ifdef PULSE_SIMD_SSE42
    __m128 m128;
#else
    float x, y, z, w;
#endif

    // ── Constructors ──────────────────────────────────────────────────────

    PULSE_FORCE_INLINE Vec4() noexcept
#ifdef PULSE_SIMD_SSE42
        : m128(_mm_setzero_ps())
#else
        : x(0.0f), y(0.0f), z(0.0f), w(0.0f)
#endif
    {}

    explicit PULSE_FORCE_INLINE Vec4(float s) noexcept
#ifdef PULSE_SIMD_SSE42
        : m128(_mm_set1_ps(s))
#else
        : x(s), y(s), z(s), w(s)
#endif
    {}

    PULSE_FORCE_INLINE Vec4(float x_, float y_, float z_, float w_) noexcept
#ifdef PULSE_SIMD_SSE42
        : m128(_mm_set_ps(w_, z_, y_, x_))
#else
        : x(x_), y(y_), z(z_), w(w_)
#endif
    {}

    /// Construct from Vec3 + w component.
    PULSE_FORCE_INLINE Vec4(Vec3 v, float w_) noexcept
#ifdef PULSE_SIMD_SSE42
        : m128(_mm_blend_ps(v.m128, _mm_set_ps(w_, 0.0f, 0.0f, 0.0f), 0b1000))
#else
        : x(v.getX()), y(v.getY()), z(v.getZ()), w(w_)
#endif
    {}

#ifdef PULSE_SIMD_SSE42
    PULSE_FORCE_INLINE Vec4(__m128 v) noexcept : m128(v) {}
#endif

    // ── Static factories ──────────────────────────────────────────────────

    [[nodiscard]] static PULSE_FORCE_INLINE Vec4 zero() noexcept { return Vec4(0.0f, 0.0f, 0.0f, 0.0f); }
    [[nodiscard]] static PULSE_FORCE_INLINE Vec4 one()  noexcept { return Vec4(1.0f, 1.0f, 1.0f, 1.0f); }
    [[nodiscard]] static PULSE_FORCE_INLINE Vec4 unitX() noexcept { return Vec4(1.0f, 0.0f, 0.0f, 0.0f); }
    [[nodiscard]] static PULSE_FORCE_INLINE Vec4 unitY() noexcept { return Vec4(0.0f, 1.0f, 0.0f, 0.0f); }
    [[nodiscard]] static PULSE_FORCE_INLINE Vec4 unitZ() noexcept { return Vec4(0.0f, 0.0f, 1.0f, 0.0f); }
    [[nodiscard]] static PULSE_FORCE_INLINE Vec4 unitW() noexcept { return Vec4(0.0f, 0.0f, 0.0f, 1.0f); }

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

    [[nodiscard]] PULSE_FORCE_INLINE float getW() const noexcept {
#ifdef PULSE_SIMD_SSE42
        return _mm_cvtss_f32(_mm_shuffle_ps(m128, m128, _MM_SHUFFLE(3, 3, 3, 3)));
#else
        return w;
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

    PULSE_FORCE_INLINE void setW(float v) noexcept {
#ifdef PULSE_SIMD_SSE42
        __m128 tmp = _mm_set_ps(v, 0.0f, 0.0f, 0.0f);
        m128 = _mm_blend_ps(m128, tmp, 0b1000);
#else
        w = v;
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE float operator[](int i) const noexcept {
#ifdef PULSE_SIMD_SSE42
        PULSE_SIMD_ALIGN float tmp[4];
        _mm_store_ps(tmp, m128);
        return tmp[i];
#else
        return (&x)[i];
#endif
    }

    /// Extract the xyz components as a Vec3.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 xyz() const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Vec3(simd::zeroW(m128), Vec3::TrustedTag{});
#else
        return Vec3(x, y, z);
#endif
    }

    // ── Arithmetic operators ──────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE Vec4 operator+(Vec4 rhs) const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Vec4(_mm_add_ps(m128, rhs.m128));
#else
        return {x + rhs.x, y + rhs.y, z + rhs.z, w + rhs.w};
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec4 operator-(Vec4 rhs) const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Vec4(_mm_sub_ps(m128, rhs.m128));
#else
        return {x - rhs.x, y - rhs.y, z - rhs.z, w - rhs.w};
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec4 operator*(float s) const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Vec4(_mm_mul_ps(m128, _mm_set1_ps(s)));
#else
        return {x * s, y * s, z * s, w * s};
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec4 operator/(float s) const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Vec4(_mm_div_ps(m128, _mm_set1_ps(s)));
#else
        const float inv = 1.0f / s;
        return {x * inv, y * inv, z * inv, w * inv};
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec4 operator*(Vec4 rhs) const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Vec4(_mm_mul_ps(m128, rhs.m128));
#else
        return {x * rhs.x, y * rhs.y, z * rhs.z, w * rhs.w};
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec4 operator-() const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Vec4(_mm_sub_ps(_mm_setzero_ps(), m128));
#else
        return {-x, -y, -z, -w};
#endif
    }

    PULSE_FORCE_INLINE Vec4& operator+=(Vec4 rhs) noexcept {
#ifdef PULSE_SIMD_SSE42
        m128 = _mm_add_ps(m128, rhs.m128);
#else
        x += rhs.x; y += rhs.y; z += rhs.z; w += rhs.w;
#endif
        return *this;
    }

    PULSE_FORCE_INLINE Vec4& operator-=(Vec4 rhs) noexcept {
#ifdef PULSE_SIMD_SSE42
        m128 = _mm_sub_ps(m128, rhs.m128);
#else
        x -= rhs.x; y -= rhs.y; z -= rhs.z; w -= rhs.w;
#endif
        return *this;
    }

    PULSE_FORCE_INLINE Vec4& operator*=(float s) noexcept {
#ifdef PULSE_SIMD_SSE42
        m128 = _mm_mul_ps(m128, _mm_set1_ps(s));
#else
        x *= s; y *= s; z *= s; w *= s;
#endif
        return *this;
    }

    PULSE_FORCE_INLINE Vec4& operator/=(float s) noexcept {
#ifdef PULSE_SIMD_SSE42
        m128 = _mm_div_ps(m128, _mm_set1_ps(s));
#else
        const float inv = 1.0f / s;
        x *= inv; y *= inv; z *= inv; w *= inv;
#endif
        return *this;
    }

    // ── Comparison ────────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE bool operator==(Vec4 rhs) const noexcept {
#ifdef PULSE_SIMD_SSE42
        __m128 diff = _mm_sub_ps(m128, rhs.m128);
        __m128 absDiff = _mm_andnot_ps(_mm_set1_ps(-0.0f), diff);
        __m128 eps = _mm_set1_ps(math::Epsilon);
        __m128 cmp = _mm_cmple_ps(absDiff, eps);
        return _mm_movemask_ps(cmp) == 0xF;
#else
        return math::approxEqual(x, rhs.x) && math::approxEqual(y, rhs.y) &&
               math::approxEqual(z, rhs.z) && math::approxEqual(w, rhs.w);
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE bool operator!=(Vec4 rhs) const noexcept {
        return !(*this == rhs);
    }

    // ── Length / Normalization ─────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE float lengthSq() const noexcept {
#ifdef PULSE_SIMD_SSE42
        return simd::dot4(m128, m128);
#else
        return x * x + y * y + z * z + w * w;
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE float length() const noexcept {
#ifdef PULSE_SIMD_SSE42
        return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(m128, m128, 0xF1)));
#else
        return math::fastSqrt(lengthSq());
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec4 normalized() const noexcept {
#ifdef PULSE_SIMD_SSE42
        __m128 dp = _mm_dp_ps(m128, m128, 0xFF);
        __m128 eps = _mm_set1_ps(1.0e-12f);
        __m128 mask = _mm_cmpgt_ps(dp, eps);
        __m128 inv_len = _mm_rsqrt_ps(dp);
        // Newton-Raphson refinement
        __m128 half = _mm_set1_ps(0.5f);
        __m128 three = _mm_set1_ps(3.0f);
        __m128 muls = _mm_mul_ps(_mm_mul_ps(dp, inv_len), inv_len);
        inv_len = _mm_mul_ps(_mm_mul_ps(half, inv_len), _mm_sub_ps(three, muls));
        __m128 result = _mm_mul_ps(m128, inv_len);
        return Vec4(_mm_and_ps(result, mask));
#else
        const float lsq = lengthSq();
        if (lsq < math::Epsilon * math::Epsilon) return Vec4::zero();
        const float inv = math::fastInvSqrt(lsq);
        return {x * inv, y * inv, z * inv, w * inv};
#endif
    }

    PULSE_FORCE_INLINE void normalize() noexcept {
        *this = normalized();
    }

    // ── Dot ───────────────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE float dot(Vec4 rhs) const noexcept {
#ifdef PULSE_SIMD_SSE42
        return simd::dot4(m128, rhs.m128);
#else
        return x * rhs.x + y * rhs.y + z * rhs.z + w * rhs.w;
#endif
    }

    // ── Utilities ─────────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE Vec4 lerp(Vec4 target, float t) const noexcept {
#ifdef PULSE_SIMD_SSE42
        __m128 vt = _mm_set1_ps(t);
        __m128 diff = _mm_sub_ps(target.m128, m128);
        return Vec4(_mm_add_ps(m128, _mm_mul_ps(diff, vt)));
#else
        return {
            math::lerp(x, target.x, t),
            math::lerp(y, target.y, t),
            math::lerp(z, target.z, t),
            math::lerp(w, target.w, t)
        };
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec4 min(Vec4 rhs) const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Vec4(_mm_min_ps(m128, rhs.m128));
#else
        return {
            (x < rhs.x) ? x : rhs.x, (y < rhs.y) ? y : rhs.y,
            (z < rhs.z) ? z : rhs.z, (w < rhs.w) ? w : rhs.w
        };
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec4 max(Vec4 rhs) const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Vec4(_mm_max_ps(m128, rhs.m128));
#else
        return {
            (x > rhs.x) ? x : rhs.x, (y > rhs.y) ? y : rhs.y,
            (z > rhs.z) ? z : rhs.z, (w > rhs.w) ? w : rhs.w
        };
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec4 abs() const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Vec4(_mm_andnot_ps(_mm_set1_ps(-0.0f), m128));
#else
        return {math::fastAbs(x), math::fastAbs(y), math::fastAbs(z), math::fastAbs(w)};
#endif
    }

    /// Perspective division: returns {x/w, y/w, z/w, 1.0f}.
    [[nodiscard]] PULSE_FORCE_INLINE Vec4 perspectiveDivide() const noexcept {
        const float invW = 1.0f / getW();
        return {getX() * invW, getY() * invW, getZ() * invW, 1.0f};
    }
};

// ── Free-function operators ───────────────────────────────────────────────────

[[nodiscard]] PULSE_FORCE_INLINE Vec4 operator*(float s, Vec4 v) noexcept {
    return v * s;
}

// ── Free-function utilities ───────────────────────────────────────────────────

namespace math {
    [[nodiscard]] PULSE_FORCE_INLINE float dot(Vec4 a, Vec4 b) noexcept {
        return a.dot(b);
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec4 normalize(Vec4 v) noexcept {
        return v.normalized();
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec4 lerp(Vec4 a, Vec4 b, float t) noexcept {
        return a.lerp(b, t);
    }
} // namespace math

} // namespace pulse
