/**
 * @file vec2.h
 * @brief 2D vector — lightweight scalar implementation.
 *
 * Vec2 is intentionally kept scalar (no SIMD). It's 8 bytes, naturally aligned,
 * and used primarily for UV coordinates, screen-space calculations, and 2D physics
 * projections where the overhead of SIMD register loads would dominate.
 */

#pragma once

#include "math_common.h"

namespace pulse {

/**
 * @struct Vec2
 * @brief A 2-component floating-point vector.
 *
 * Memory layout: [x, y] — 8 bytes, 4-byte aligned.
 * No SIMD — the register load/store overhead exceeds the benefit for 2 floats.
 */
struct Vec2 {
    float x, y;

    // ── Constructors ──────────────────────────────────────────────────────

    /// Default: zero-initialized.
    constexpr Vec2() noexcept : x(0.0f), y(0.0f) {}

    /// Broadcast scalar to both components.
    explicit constexpr Vec2(float s) noexcept : x(s), y(s) {}

    /// Component-wise construction.
    constexpr Vec2(float x_, float y_) noexcept : x(x_), y(y_) {}

    // ── Static factories ──────────────────────────────────────────────────

    [[nodiscard]] static constexpr Vec2 zero()  noexcept { return {0.0f, 0.0f}; }
    [[nodiscard]] static constexpr Vec2 one()   noexcept { return {1.0f, 1.0f}; }
    [[nodiscard]] static constexpr Vec2 unitX() noexcept { return {1.0f, 0.0f}; }
    [[nodiscard]] static constexpr Vec2 unitY() noexcept { return {0.0f, 1.0f}; }

    // ── Element access ────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE constexpr float  operator[](int i) const noexcept { return (&x)[i]; }
    [[nodiscard]] PULSE_FORCE_INLINE constexpr float& operator[](int i)       noexcept { return (&x)[i]; }

    // ── Arithmetic operators ──────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE constexpr Vec2 operator+(Vec2 rhs) const noexcept {
        return {x + rhs.x, y + rhs.y};
    }

    [[nodiscard]] PULSE_FORCE_INLINE constexpr Vec2 operator-(Vec2 rhs) const noexcept {
        return {x - rhs.x, y - rhs.y};
    }

    [[nodiscard]] PULSE_FORCE_INLINE constexpr Vec2 operator*(float s) const noexcept {
        return {x * s, y * s};
    }

    [[nodiscard]] PULSE_FORCE_INLINE constexpr Vec2 operator/(float s) const noexcept {
        const float inv = 1.0f / s;
        return {x * inv, y * inv};
    }

    [[nodiscard]] PULSE_FORCE_INLINE constexpr Vec2 operator*(Vec2 rhs) const noexcept {
        return {x * rhs.x, y * rhs.y};
    }

    [[nodiscard]] PULSE_FORCE_INLINE constexpr Vec2 operator-() const noexcept {
        return {-x, -y};
    }

    PULSE_FORCE_INLINE constexpr Vec2& operator+=(Vec2 rhs) noexcept {
        x += rhs.x; y += rhs.y; return *this;
    }

    PULSE_FORCE_INLINE constexpr Vec2& operator-=(Vec2 rhs) noexcept {
        x -= rhs.x; y -= rhs.y; return *this;
    }

    PULSE_FORCE_INLINE constexpr Vec2& operator*=(float s) noexcept {
        x *= s; y *= s; return *this;
    }

    PULSE_FORCE_INLINE constexpr Vec2& operator/=(float s) noexcept {
        const float inv = 1.0f / s;
        x *= inv; y *= inv; return *this;
    }

    // ── Comparison ────────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE constexpr bool operator==(Vec2 rhs) const noexcept {
        return math::approxEqual(x, rhs.x) && math::approxEqual(y, rhs.y);
    }

    [[nodiscard]] PULSE_FORCE_INLINE constexpr bool operator!=(Vec2 rhs) const noexcept {
        return !(*this == rhs);
    }

    // ── Length / Normalization ─────────────────────────────────────────────

    /// Squared length (avoids sqrt).
    [[nodiscard]] PULSE_FORCE_INLINE constexpr float lengthSq() const noexcept {
        return x * x + y * y;
    }

    /// Euclidean length.
    [[nodiscard]] PULSE_FORCE_INLINE float length() const noexcept {
        return math::fastSqrt(x * x + y * y);
    }

    /// Returns a normalized copy. Returns zero vector if length is near-zero.
    [[nodiscard]] PULSE_FORCE_INLINE Vec2 normalized() const noexcept {
        const float lsq = lengthSq();
        if (lsq < math::Epsilon * math::Epsilon) return Vec2::zero();
        const float inv = math::fastInvSqrt(lsq);
        return {x * inv, y * inv};
    }

    /// Normalizes in place.
    PULSE_FORCE_INLINE void normalize() noexcept {
        *this = normalized();
    }

    // ── Dot / Cross / Perpendicular ───────────────────────────────────────

    /// Dot product.
    [[nodiscard]] PULSE_FORCE_INLINE constexpr float dot(Vec2 rhs) const noexcept {
        return x * rhs.x + y * rhs.y;
    }

    /// 2D cross product (returns scalar: the z-component of the 3D cross product).
    [[nodiscard]] PULSE_FORCE_INLINE constexpr float cross(Vec2 rhs) const noexcept {
        return x * rhs.y - y * rhs.x;
    }

    /// Perpendicular vector (90° counter-clockwise rotation).
    [[nodiscard]] PULSE_FORCE_INLINE constexpr Vec2 perp() const noexcept {
        return {-y, x};
    }

    // ── Utilities ─────────────────────────────────────────────────────────

    /// Distance to another point.
    [[nodiscard]] PULSE_FORCE_INLINE float distanceTo(Vec2 other) const noexcept {
        return (*this - other).length();
    }

    /// Squared distance to another point.
    [[nodiscard]] PULSE_FORCE_INLINE constexpr float distanceSqTo(Vec2 other) const noexcept {
        return (*this - other).lengthSq();
    }

    /// Reflect this vector about a normal.
    [[nodiscard]] PULSE_FORCE_INLINE constexpr Vec2 reflect(Vec2 normal) const noexcept {
        return *this - normal * (2.0f * dot(normal));
    }

    /// Linear interpolation towards another vector.
    [[nodiscard]] PULSE_FORCE_INLINE constexpr Vec2 lerp(Vec2 target, float t) const noexcept {
        return {
            math::lerp(x, target.x, t),
            math::lerp(y, target.y, t)
        };
    }

    /// Component-wise min.
    [[nodiscard]] PULSE_FORCE_INLINE constexpr Vec2 min(Vec2 rhs) const noexcept {
        return {(x < rhs.x) ? x : rhs.x, (y < rhs.y) ? y : rhs.y};
    }

    /// Component-wise max.
    [[nodiscard]] PULSE_FORCE_INLINE constexpr Vec2 max(Vec2 rhs) const noexcept {
        return {(x > rhs.x) ? x : rhs.x, (y > rhs.y) ? y : rhs.y};
    }

    /// Component-wise absolute value.
    [[nodiscard]] PULSE_FORCE_INLINE Vec2 abs() const noexcept {
        return {math::fastAbs(x), math::fastAbs(y)};
    }

    /// Clamp each component between [lo, hi].
    [[nodiscard]] PULSE_FORCE_INLINE constexpr Vec2 clamped(float lo, float hi) const noexcept {
        return {math::clamp(x, lo, hi), math::clamp(y, lo, hi)};
    }
};

// ── Free-function operators ───────────────────────────────────────────────────

/// Scalar * Vec2.
[[nodiscard]] PULSE_FORCE_INLINE constexpr Vec2 operator*(float s, Vec2 v) noexcept {
    return v * s;
}

// ── Free-function utilities ───────────────────────────────────────────────────

namespace math {
    [[nodiscard]] PULSE_FORCE_INLINE constexpr float dot(Vec2 a, Vec2 b) noexcept {
        return a.dot(b);
    }

    [[nodiscard]] PULSE_FORCE_INLINE constexpr float cross(Vec2 a, Vec2 b) noexcept {
        return a.cross(b);
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec2 normalize(Vec2 v) noexcept {
        return v.normalized();
    }

    [[nodiscard]] PULSE_FORCE_INLINE constexpr Vec2 lerp(Vec2 a, Vec2 b, float t) noexcept {
        return a.lerp(b, t);
    }
} // namespace math

} // namespace pulse
