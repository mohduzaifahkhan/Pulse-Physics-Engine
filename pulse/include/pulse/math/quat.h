/**
 * @file quat.h
 * @brief Quaternion — SSE4.2 accelerated, 16-byte aligned.
 *
 * Stores (x, y, z, w) where w is the scalar part. This matches the convention
 * used by Bullet, PhysX, and glTF. Internally stored as __m128 [x, y, z, w].
 *
 * All quaternion operations (multiply, conjugate, inverse, slerp, rotate vector)
 * are implemented with SIMD intrinsics. The quaternion is always assumed to be
 * unit-length for rotation operations.
 *
 * Memory layout: [x, y, z, w] — 16 bytes, 16-byte aligned.
 */

#pragma once

#include "math_common.h"
#include "vec3.h"

namespace pulse {

/**
 * @struct Quat
 * @brief A unit quaternion for representing 3D rotations.
 *
 * Convention: q = w + xi + yj + zk, stored as [x, y, z, w] in memory.
 */
struct PULSE_SIMD_ALIGN Quat {
#ifdef PULSE_SIMD_SSE42
    __m128 m128; ///< [x, y, z, w]
#else
    float x, y, z, w;
#endif

    // ── Constructors ──────────────────────────────────────────────────────

    /// Default: identity quaternion (0, 0, 0, 1).
    PULSE_FORCE_INLINE Quat() noexcept
#ifdef PULSE_SIMD_SSE42
        : m128(_mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f))
#else
        : x(0.0f), y(0.0f), z(0.0f), w(1.0f)
#endif
    {}

    /// Direct component construction: (x, y, z, w).
    PULSE_FORCE_INLINE Quat(float x_, float y_, float z_, float w_) noexcept
#ifdef PULSE_SIMD_SSE42
        : m128(_mm_set_ps(w_, z_, y_, x_))
#else
        : x(x_), y(y_), z(z_), w(w_)
#endif
    {}

#ifdef PULSE_SIMD_SSE42
    PULSE_FORCE_INLINE Quat(__m128 v) noexcept : m128(v) {}
#endif

    // ── Static factories ──────────────────────────────────────────────────

    /// Identity quaternion (no rotation).
    [[nodiscard]] static PULSE_FORCE_INLINE Quat identity() noexcept {
        return Quat(0.0f, 0.0f, 0.0f, 1.0f);
    }

    /// Create from axis-angle representation.
    /// @param axis Normalized rotation axis.
    /// @param angleRad Rotation angle in radians.
    [[nodiscard]] static PULSE_FORCE_INLINE Quat fromAxisAngle(Vec3 axis, float angleRad) noexcept {
        const float halfAngle = angleRad * 0.5f;
        const float s = std::sin(halfAngle);
        const float c = std::cos(halfAngle);
#ifdef PULSE_SIMD_SSE42
        __m128 vsin = _mm_set1_ps(s);
        __m128 scaled = _mm_mul_ps(axis.m128, vsin);
        // Set w = c
        return Quat(_mm_blend_ps(scaled, _mm_set_ps(c, 0.0f, 0.0f, 0.0f), 0b1000));
#else
        return Quat(axis.getX() * s, axis.getY() * s, axis.getZ() * s, c);
#endif
    }

    /// Create from Euler angles (pitch, yaw, roll) in radians.
    /// Convention: ZYX intrinsic (roll around Z, yaw around Y, pitch around X).
    [[nodiscard]] static PULSE_FORCE_INLINE Quat fromEuler(float pitch, float yaw, float roll) noexcept {
        const float cy = std::cos(yaw * 0.5f);
        const float sy = std::sin(yaw * 0.5f);
        const float cp = std::cos(pitch * 0.5f);
        const float sp = std::sin(pitch * 0.5f);
        const float cr = std::cos(roll * 0.5f);
        const float sr = std::sin(roll * 0.5f);

        return Quat(
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy
        );
    }

    /// Create a quaternion that rotates from direction 'from' to direction 'to'.
    /// Both must be unit-length.
    [[nodiscard]] static PULSE_FORCE_INLINE Quat fromTwoVectors(Vec3 from, Vec3 to) noexcept {
        const float d = from.dot(to);
        if (d >= 1.0f - math::Epsilon) {
            // Vectors are parallel, same direction
            return Quat::identity();
        }
        if (d <= -1.0f + math::Epsilon) {
            // Vectors are anti-parallel — pick an arbitrary perpendicular axis
            Vec3 perp = Vec3::unitX().cross(from);
            if (perp.lengthSq() < math::Epsilon) {
                perp = Vec3::unitY().cross(from);
            }
            perp.normalize();
            return Quat(perp.getX(), perp.getY(), perp.getZ(), 0.0f);
        }
        Vec3 half = (from + to).normalized();
        Vec3 c = from.cross(half);
        return Quat(c.getX(), c.getY(), c.getZ(), from.dot(half));
    }

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

    // ── Quaternion multiplication ─────────────────────────────────────────

    /// Hamilton product: this * rhs.
    /// If both are unit quaternions, represents composing the rotations.
    [[nodiscard]] PULSE_FORCE_INLINE Quat operator*(Quat rhs) const noexcept {
#ifdef PULSE_SIMD_SSE42
        // Hamilton product using SIMD
        // Let a = this = [ax, ay, az, aw], b = rhs = [bx, by, bz, bw]
        // Result:
        //   w = aw*bw - ax*bx - ay*by - az*bz
        //   x = aw*bx + ax*bw + ay*bz - az*by
        //   y = aw*by - ax*bz + ay*bw + az*bx
        //   z = aw*bz + ax*by - ay*bx + az*bw

        // Splat each component of 'this'
        __m128 a_xxxx = _mm_shuffle_ps(m128, m128, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 a_yyyy = _mm_shuffle_ps(m128, m128, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 a_zzzz = _mm_shuffle_ps(m128, m128, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 a_wwww = _mm_shuffle_ps(m128, m128, _MM_SHUFFLE(3, 3, 3, 3));

        // b shuffled for each partial product
        // For x: bw, bz, -by, -bx -> swap + sign
        __m128 b_wzyx = _mm_shuffle_ps(rhs.m128, rhs.m128, _MM_SHUFFLE(0, 1, 2, 3));
        __m128 b_zwxy = _mm_shuffle_ps(rhs.m128, rhs.m128, _MM_SHUFFLE(1, 0, 3, 2));
        __m128 b_yxwz = _mm_shuffle_ps(rhs.m128, rhs.m128, _MM_SHUFFLE(2, 3, 0, 1));

        // Sign masks for Hamilton product
        // p0 = aw * [bx, by, bz, bw]
        __m128 p0 = _mm_mul_ps(a_wwww, rhs.m128);

        // p1 = ax * [bw, -bz, by, -bx]  (sign pattern: +, -, +, -)
        __m128 sign1 = _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f);
        __m128 p1 = _mm_mul_ps(a_xxxx, _mm_mul_ps(b_wzyx, sign1));

        // p2 = ay * [bz, bw, -bx, -by]  (sign pattern: +, +, -, -)
        __m128 sign2 = _mm_set_ps(-1.0f, -1.0f, 1.0f, 1.0f);
        __m128 p2 = _mm_mul_ps(a_yyyy, _mm_mul_ps(b_zwxy, sign2));

        // p3 = az * [-by, bx, bw, -bz]  (sign pattern: -, +, +, -)
        __m128 sign3 = _mm_set_ps(-1.0f, 1.0f, 1.0f, -1.0f);
        __m128 p3 = _mm_mul_ps(a_zzzz, _mm_mul_ps(b_yxwz, sign3));

        return Quat(_mm_add_ps(_mm_add_ps(p0, p1), _mm_add_ps(p2, p3)));
#else
        return Quat(
            w * rhs.x + x * rhs.w + y * rhs.z - z * rhs.y,
            w * rhs.y - x * rhs.z + y * rhs.w + z * rhs.x,
            w * rhs.z + x * rhs.y - y * rhs.x + z * rhs.w,
            w * rhs.w - x * rhs.x - y * rhs.y - z * rhs.z
        );
#endif
    }

    PULSE_FORCE_INLINE Quat& operator*=(Quat rhs) noexcept {
        *this = *this * rhs;
        return *this;
    }

    // ── Scalar multiply / add ─────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE Quat operator*(float s) const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Quat(_mm_mul_ps(m128, _mm_set1_ps(s)));
#else
        return {x * s, y * s, z * s, w * s};
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE Quat operator+(Quat rhs) const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Quat(_mm_add_ps(m128, rhs.m128));
#else
        return {x + rhs.x, y + rhs.y, z + rhs.z, w + rhs.w};
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE Quat operator-(Quat rhs) const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Quat(_mm_sub_ps(m128, rhs.m128));
#else
        return {x - rhs.x, y - rhs.y, z - rhs.z, w - rhs.w};
#endif
    }

    [[nodiscard]] PULSE_FORCE_INLINE Quat operator-() const noexcept {
#ifdef PULSE_SIMD_SSE42
        return Quat(_mm_sub_ps(_mm_setzero_ps(), m128));
#else
        return {-x, -y, -z, -w};
#endif
    }

    // ── Conjugate / Inverse ───────────────────────────────────────────────

    /// Conjugate: negate the vector part (x, y, z), keep w.
    [[nodiscard]] PULSE_FORCE_INLINE Quat conjugate() const noexcept {
#ifdef PULSE_SIMD_SSE42
        // Negate x, y, z; keep w. Sign mask: [-1, -1, -1, 1].
        __m128 sign = _mm_set_ps(1.0f, -1.0f, -1.0f, -1.0f);
        return Quat(_mm_mul_ps(m128, sign));
#else
        return {-x, -y, -z, w};
#endif
    }

    /// Inverse: for unit quaternions, this is the same as conjugate.
    [[nodiscard]] PULSE_FORCE_INLINE Quat inverse() const noexcept {
        return conjugate(); // Assumes unit quaternion
    }

    /// Length squared (should be 1.0 for unit quaternions).
    [[nodiscard]] PULSE_FORCE_INLINE float lengthSq() const noexcept {
#ifdef PULSE_SIMD_SSE42
        return simd::dot4(m128, m128);
#else
        return x * x + y * y + z * z + w * w;
#endif
    }

    /// Length.
    [[nodiscard]] PULSE_FORCE_INLINE float length() const noexcept {
#ifdef PULSE_SIMD_SSE42
        return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(m128, m128, 0xF1)));
#else
        return math::fastSqrt(lengthSq());
#endif
    }

    /// Normalize in place.
    PULSE_FORCE_INLINE void normalize() noexcept {
#ifdef PULSE_SIMD_SSE42
        __m128 dp = _mm_dp_ps(m128, m128, 0xFF);
        __m128 inv_len = _mm_rsqrt_ps(dp);
        // Newton-Raphson
        __m128 half = _mm_set1_ps(0.5f);
        __m128 three = _mm_set1_ps(3.0f);
        __m128 muls = _mm_mul_ps(_mm_mul_ps(dp, inv_len), inv_len);
        inv_len = _mm_mul_ps(_mm_mul_ps(half, inv_len), _mm_sub_ps(three, muls));
        m128 = _mm_mul_ps(m128, inv_len);
#else
        const float inv = math::fastInvSqrt(lengthSq());
        x *= inv; y *= inv; z *= inv; w *= inv;
#endif
    }

    /// Return normalized copy.
    [[nodiscard]] PULSE_FORCE_INLINE Quat normalized() const noexcept {
        Quat q = *this;
        q.normalize();
        return q;
    }

    // ── Dot product ───────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE float dot(Quat rhs) const noexcept {
#ifdef PULSE_SIMD_SSE42
        return simd::dot4(m128, rhs.m128);
#else
        return x * rhs.x + y * rhs.y + z * rhs.z + w * rhs.w;
#endif
    }

    // ── Rotate a vector ───────────────────────────────────────────────────

    /// Rotate a Vec3 by this quaternion: q * v * q^{-1}.
    /// Optimized formula: v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 rotate(Vec3 v) const noexcept {
#ifdef PULSE_SIMD_SSE42
        // Extract vector part of quaternion
        __m128 qv = simd::zeroW(m128); // (qx, qy, qz, 0)
        __m128 qw = _mm_shuffle_ps(m128, m128, _MM_SHUFFLE(3, 3, 3, 3)); // (qw, qw, qw, qw)

        // t = 2 * cross(q.xyz, v)
        __m128 t = simd::cross3(qv, v.m128);
        t = _mm_add_ps(t, t); // t *= 2

        // result = v + qw * t + cross(q.xyz, t)
        __m128 result = _mm_add_ps(v.m128, _mm_mul_ps(qw, t));
        result = _mm_add_ps(result, simd::cross3(qv, t));
        return Vec3(result, Vec3::TrustedTag{});
#else
        // v' = v + 2w * (q.xyz x v) + 2 * (q.xyz x (q.xyz x v))
        Vec3 qv(x, y, z);
        Vec3 t = qv.cross(v) * 2.0f;
        return v + t * w + qv.cross(t);
#endif
    }

    /// Inverse rotate: rotate by the conjugate quaternion.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 inverseRotate(Vec3 v) const noexcept {
        return conjugate().rotate(v);
    }

    // ── Interpolation ─────────────────────────────────────────────────────

    /// Spherical linear interpolation (SLERP).
    [[nodiscard]] PULSE_FORCE_INLINE Quat slerp(Quat target, float t) const noexcept {
        float cosHalfAngle = dot(target);

        // If dot is negative, negate one quaternion to take the shorter path
        Quat adjustedTarget = target;
        if (cosHalfAngle < 0.0f) {
            adjustedTarget = -target;
            cosHalfAngle = -cosHalfAngle;
        }

        // If quaternions are very close, use linear interpolation to avoid division by zero
        if (cosHalfAngle > 1.0f - math::Epsilon) {
            Quat result = *this * (1.0f - t) + adjustedTarget * t;
            result.normalize();
            return result;
        }

        const float halfAngle = std::acos(cosHalfAngle);
        const float sinHalfAngle = std::sin(halfAngle);
        const float invSin = 1.0f / sinHalfAngle;
        const float ratioA = std::sin((1.0f - t) * halfAngle) * invSin;
        const float ratioB = std::sin(t * halfAngle) * invSin;

        return *this * ratioA + adjustedTarget * ratioB;
    }

    /// Normalized linear interpolation (NLERP) — faster than SLERP, nearly identical for small angles.
    [[nodiscard]] PULSE_FORCE_INLINE Quat nlerp(Quat target, float t) const noexcept {
        float d = dot(target);
        Quat adjustedTarget = (d < 0.0f) ? -target : target;
        Quat result = *this * (1.0f - t) + adjustedTarget * t;
        result.normalize();
        return result;
    }

    // ── Conversion ────────────────────────────────────────────────────────

    /// Extract the rotation axis and angle from this unit quaternion.
    PULSE_FORCE_INLINE void toAxisAngle(Vec3& axis, float& angle) const noexcept {
        const float qw = getW();
        angle = 2.0f * std::acos(math::clamp(qw, -1.0f, 1.0f));
        const float sinHalf = std::sin(angle * 0.5f);
        if (math::fastAbs(sinHalf) > math::Epsilon) {
            const float inv = 1.0f / sinHalf;
            axis = Vec3(getX() * inv, getY() * inv, getZ() * inv);
        } else {
            axis = Vec3::unitX();
            angle = 0.0f;
        }
    }

    /// Extract Euler angles (pitch, yaw, roll) in radians.
    PULSE_FORCE_INLINE void toEuler(float& pitch, float& yaw, float& roll) const noexcept {
        const float qx = getX(), qy = getY(), qz = getZ(), qw = getW();

        // Roll (X-axis rotation)
        const float sinr_cosp = 2.0f * (qw * qx + qy * qz);
        const float cosr_cosp = 1.0f - 2.0f * (qx * qx + qy * qy);
        roll = std::atan2(sinr_cosp, cosr_cosp);

        // Pitch (Y-axis rotation)
        const float sinp = 2.0f * (qw * qy - qz * qx);
        if (math::fastAbs(sinp) >= 1.0f) {
            pitch = std::copysign(math::HalfPi, sinp);
        } else {
            pitch = std::asin(sinp);
        }

        // Yaw (Z-axis rotation)
        const float siny_cosp = 2.0f * (qw * qz + qx * qy);
        const float cosy_cosp = 1.0f - 2.0f * (qy * qy + qz * qz);
        yaw = std::atan2(siny_cosp, cosy_cosp);
    }

    /// Get the "forward" direction this quaternion points to (rotating -Z).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 getForward() const noexcept {
        return rotate(Vec3(0.0f, 0.0f, -1.0f));
    }

    /// Get the "up" direction this quaternion points to (rotating +Y).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 getUp() const noexcept {
        return rotate(Vec3(0.0f, 1.0f, 0.0f));
    }

    /// Get the "right" direction this quaternion points to (rotating +X).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 getRight() const noexcept {
        return rotate(Vec3(1.0f, 0.0f, 0.0f));
    }

    // ── Comparison ────────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE bool operator==(Quat rhs) const noexcept {
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

    [[nodiscard]] PULSE_FORCE_INLINE bool operator!=(Quat rhs) const noexcept {
        return !(*this == rhs);
    }
};

// ── Free-function utilities ───────────────────────────────────────────────────

namespace math {
    [[nodiscard]] PULSE_FORCE_INLINE Quat slerp(Quat a, Quat b, float t) noexcept {
        return a.slerp(b, t);
    }

    [[nodiscard]] PULSE_FORCE_INLINE Quat nlerp(Quat a, Quat b, float t) noexcept {
        return a.nlerp(b, t);
    }

    [[nodiscard]] PULSE_FORCE_INLINE float dot(Quat a, Quat b) noexcept {
        return a.dot(b);
    }

    [[nodiscard]] PULSE_FORCE_INLINE Quat normalize(Quat q) noexcept {
        return q.normalized();
    }

    [[nodiscard]] PULSE_FORCE_INLINE Quat conjugate(Quat q) noexcept {
        return q.conjugate();
    }

    [[nodiscard]] PULSE_FORCE_INLINE Quat inverse(Quat q) noexcept {
        return q.inverse();
    }
} // namespace math

} // namespace pulse
