/**
 * @file mat3.h
 * @brief 3×3 matrix — SSE4.2 accelerated, row-major layout.
 *
 * Stored as three __m128 rows (each with w = 0). This allows row-vector
 * transforms, SIMD dot products for each output component, and efficient
 * transpose using SSE shuffle instructions.
 *
 * Primary uses:
 * - Inertia tensors
 * - Rotation matrices
 * - Normal transforms (inverse-transpose)
 * - 3×3 linear algebra (determinant, inverse, eigen decomposition stubs)
 *
 * Memory layout: 3× __m128 = 48 bytes, 16-byte aligned.
 * Row-major: row[0] = [m00, m01, m02, 0], row[1] = [m10, m11, m12, 0], ...
 */

#pragma once

#include "math_common.h"
#include "vec3.h"
#include "quat.h"

namespace pulse {

/**
 * @struct Mat3
 * @brief A 3×3 matrix stored as 3 row vectors (__m128 each, w padded to 0).
 *
 * Row-major convention. Matrix * vector = row dot products.
 */
struct PULSE_SIMD_ALIGN Mat3 {
    Vec3 rows[3]; ///< Row-major storage. Each row is a Vec3 (16 bytes, w=0).

    // ── Constructors ──────────────────────────────────────────────────────

    /// Default: identity matrix.
    PULSE_FORCE_INLINE Mat3() noexcept
        : rows{Vec3(1.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f)}
    {}

    /// Construct from 3 row vectors.
    PULSE_FORCE_INLINE Mat3(Vec3 r0, Vec3 r1, Vec3 r2) noexcept
        : rows{r0, r1, r2}
    {}

    /// Construct from 9 scalars (row-major order).
    PULSE_FORCE_INLINE Mat3(
        float m00, float m01, float m02,
        float m10, float m11, float m12,
        float m20, float m21, float m22
    ) noexcept
        : rows{Vec3(m00, m01, m02), Vec3(m10, m11, m12), Vec3(m20, m21, m22)}
    {}

    // ── Static factories ──────────────────────────────────────────────────

    [[nodiscard]] static PULSE_FORCE_INLINE Mat3 identity() noexcept { return Mat3(); }

    [[nodiscard]] static PULSE_FORCE_INLINE Mat3 zero() noexcept {
        return Mat3(Vec3::zero(), Vec3::zero(), Vec3::zero());
    }

    /// Diagonal matrix.
    [[nodiscard]] static PULSE_FORCE_INLINE Mat3 diagonal(float d0, float d1, float d2) noexcept {
        return Mat3(
            d0, 0.0f, 0.0f,
            0.0f, d1, 0.0f,
            0.0f, 0.0f, d2
        );
    }

    /// Uniform scale.
    [[nodiscard]] static PULSE_FORCE_INLINE Mat3 scale(float s) noexcept {
        return diagonal(s, s, s);
    }

    /// Non-uniform scale.
    [[nodiscard]] static PULSE_FORCE_INLINE Mat3 scale(float sx, float sy, float sz) noexcept {
        return diagonal(sx, sy, sz);
    }

    /// Construct from a quaternion rotation.
    [[nodiscard]] static PULSE_FORCE_INLINE Mat3 fromQuat(Quat q) noexcept {
        const float qx = q.getX(), qy = q.getY(), qz = q.getZ(), qw = q.getW();
        const float x2 = qx + qx, y2 = qy + qy, z2 = qz + qz;
        const float xx = qx * x2, xy = qx * y2, xz = qx * z2;
        const float yy = qy * y2, yz = qy * z2, zz = qz * z2;
        const float wx = qw * x2, wy = qw * y2, wz = qw * z2;

        return Mat3(
            1.0f - (yy + zz), xy - wz,          xz + wy,
            xy + wz,          1.0f - (xx + zz),  yz - wx,
            xz - wy,          yz + wx,           1.0f - (xx + yy)
        );
    }

    /// Construct a skew-symmetric (cross-product) matrix from a vector.
    /// skew(v) * u == cross(v, u).
    [[nodiscard]] static PULSE_FORCE_INLINE Mat3 skewSymmetric(Vec3 v) noexcept {
        const float vx = v.getX(), vy = v.getY(), vz = v.getZ();
        return Mat3(
             0.0f, -vz,   vy,
             vz,    0.0f, -vx,
            -vy,    vx,    0.0f
        );
    }

    /// Outer product: a * b^T.
    [[nodiscard]] static PULSE_FORCE_INLINE Mat3 outerProduct(Vec3 a, Vec3 b) noexcept {
        return Mat3(
            a * b.getX(),
            a * b.getY(),
            a * b.getZ()
        );
    }

    // ── Element access ────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE Vec3&       operator[](int row)       noexcept { return rows[row]; }
    [[nodiscard]] PULSE_FORCE_INLINE const Vec3& operator[](int row) const noexcept { return rows[row]; }

    /// Access individual element (row, col).
    [[nodiscard]] PULSE_FORCE_INLINE float operator()(int row, int col) const noexcept {
        return rows[row][col];
    }

    /// Get a column vector.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 column(int col) const noexcept {
        return Vec3(rows[0][col], rows[1][col], rows[2][col]);
    }

    // ── Arithmetic ────────────────────────────────────────────────────────

    /// Matrix + matrix.
    [[nodiscard]] PULSE_FORCE_INLINE Mat3 operator+(const Mat3& rhs) const noexcept {
        return Mat3(rows[0] + rhs.rows[0], rows[1] + rhs.rows[1], rows[2] + rhs.rows[2]);
    }

    /// Matrix - matrix.
    [[nodiscard]] PULSE_FORCE_INLINE Mat3 operator-(const Mat3& rhs) const noexcept {
        return Mat3(rows[0] - rhs.rows[0], rows[1] - rhs.rows[1], rows[2] - rhs.rows[2]);
    }

    /// Matrix * scalar.
    [[nodiscard]] PULSE_FORCE_INLINE Mat3 operator*(float s) const noexcept {
        return Mat3(rows[0] * s, rows[1] * s, rows[2] * s);
    }

    /// Matrix * vector (transforms the vector).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 operator*(Vec3 v) const noexcept {
        return Vec3(
            rows[0].dot(v),
            rows[1].dot(v),
            rows[2].dot(v)
        );
    }

    /// Matrix * matrix.
    [[nodiscard]] PULSE_FORCE_INLINE Mat3 operator*(const Mat3& rhs) const noexcept {
        // Each element (i,j) of the result = dot(row_i, column_j of rhs).
        // Compute by transforming each row of 'this' by the transpose of rhs's columns.
        Mat3 result;
        for (int i = 0; i < 3; ++i) {
            float r0 = rows[i].dot(rhs.column(0));
            float r1 = rows[i].dot(rhs.column(1));
            float r2 = rows[i].dot(rhs.column(2));
            result.rows[i] = Vec3(r0, r1, r2);
        }
        return result;
    }

    PULSE_FORCE_INLINE Mat3& operator+=(const Mat3& rhs) noexcept {
        rows[0] += rhs.rows[0]; rows[1] += rhs.rows[1]; rows[2] += rhs.rows[2];
        return *this;
    }

    PULSE_FORCE_INLINE Mat3& operator-=(const Mat3& rhs) noexcept {
        rows[0] -= rhs.rows[0]; rows[1] -= rhs.rows[1]; rows[2] -= rhs.rows[2];
        return *this;
    }

    PULSE_FORCE_INLINE Mat3& operator*=(float s) noexcept {
        rows[0] *= s; rows[1] *= s; rows[2] *= s;
        return *this;
    }

    PULSE_FORCE_INLINE Mat3& operator*=(const Mat3& rhs) noexcept {
        *this = *this * rhs;
        return *this;
    }

    // ── Transpose ─────────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE Mat3 transposed() const noexcept {
#ifdef PULSE_SIMD_SSE42
        // Transpose 3×3 within the 4-wide __m128 registers
        // rows: r0 = [a b c 0], r1 = [d e f 0], r2 = [g h i 0]
        // result: [a d g 0], [b e h 0], [c f i 0]
        __m128 r0 = rows[0].m128;
        __m128 r1 = rows[1].m128;
        __m128 r2 = rows[2].m128;

        // t0 = [a, d, b, e]
        __m128 t0 = _mm_unpacklo_ps(r0, r1);
        // t1 = [c, f, 0, 0]
        __m128 t1 = _mm_unpackhi_ps(r0, r1);

        // col0 = [a, d, g, 0]
        __m128 col0 = _mm_movelh_ps(t0, r2);
        col0 = _mm_blend_ps(col0, _mm_setzero_ps(), 0b1000);
        // For col0: we need [a, d, g, 0]
        // t0 = [a, d, b, e], r2 = [g, h, i, 0]
        // _mm_movelh_ps(t0, r2) = [a, d, g, h] -> blend w=0
        col0 = _mm_blend_ps(_mm_movelh_ps(t0, r2), _mm_setzero_ps(), 0b1000);

        // col1 = [b, e, h, 0]
        // _mm_movehl_ps(r2, t0) = [b, e, h, i] in some order... let's use shuffle
        __m128 col1 = _mm_shuffle_ps(t0, r2, _MM_SHUFFLE(3, 1, 3, 2));
        col1 = _mm_blend_ps(col1, _mm_setzero_ps(), 0b1000);

        // col2 = [c, f, i, 0]
        __m128 col2 = _mm_shuffle_ps(t1, r2, _MM_SHUFFLE(3, 2, 1, 0));
        col2 = _mm_blend_ps(col2, _mm_setzero_ps(), 0b1000);

        return Mat3(
            Vec3(col0, Vec3::TrustedTag{}),
            Vec3(col1, Vec3::TrustedTag{}),
            Vec3(col2, Vec3::TrustedTag{})
        );
#else
        return Mat3(column(0), column(1), column(2));
#endif
    }

    // ── Determinant ───────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE float determinant() const noexcept {
        // det = row[0] . (row[1] x row[2])
        return rows[0].dot(rows[1].cross(rows[2]));
    }

    // ── Inverse ───────────────────────────────────────────────────────────

    /// Compute the inverse. Assumes the matrix is invertible.
    [[nodiscard]] PULSE_FORCE_INLINE Mat3 inversed() const noexcept {
        // Cofactor expansion
        Vec3 c0 = rows[1].cross(rows[2]);
        Vec3 c1 = rows[2].cross(rows[0]);
        Vec3 c2 = rows[0].cross(rows[1]);

        float det = rows[0].dot(c0);
        float invDet = 1.0f / det;

        // Transpose of cofactor matrix / det
        return Mat3(
            Vec3(c0.getX(), c1.getX(), c2.getX()) * invDet,
            Vec3(c0.getY(), c1.getY(), c2.getY()) * invDet,
            Vec3(c0.getZ(), c1.getZ(), c2.getZ()) * invDet
        );
    }

    // ── Trace ─────────────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE float trace() const noexcept {
        return rows[0].getX() + rows[1].getY() + rows[2].getZ();
    }

    // ── Convert to quaternion ─────────────────────────────────────────────

    /// Convert a rotation matrix to a unit quaternion.
    [[nodiscard]] PULSE_FORCE_INLINE Quat toQuat() const noexcept {
        const float tr = trace();
        float qx, qy, qz, qw;

        if (tr > 0.0f) {
            float s = math::fastSqrt(tr + 1.0f) * 2.0f; // s = 4*qw
            qw = 0.25f * s;
            float invS = 1.0f / s;
            qx = (rows[2].getY() - rows[1].getZ()) * invS;
            qy = (rows[0].getZ() - rows[2].getX()) * invS;
            qz = (rows[1].getX() - rows[0].getY()) * invS;
        } else if (rows[0].getX() > rows[1].getY() && rows[0].getX() > rows[2].getZ()) {
            float s = math::fastSqrt(1.0f + rows[0].getX() - rows[1].getY() - rows[2].getZ()) * 2.0f;
            float invS = 1.0f / s;
            qw = (rows[2].getY() - rows[1].getZ()) * invS;
            qx = 0.25f * s;
            qy = (rows[0].getY() + rows[1].getX()) * invS;
            qz = (rows[0].getZ() + rows[2].getX()) * invS;
        } else if (rows[1].getY() > rows[2].getZ()) {
            float s = math::fastSqrt(1.0f + rows[1].getY() - rows[0].getX() - rows[2].getZ()) * 2.0f;
            float invS = 1.0f / s;
            qw = (rows[0].getZ() - rows[2].getX()) * invS;
            qx = (rows[0].getY() + rows[1].getX()) * invS;
            qy = 0.25f * s;
            qz = (rows[1].getZ() + rows[2].getY()) * invS;
        } else {
            float s = math::fastSqrt(1.0f + rows[2].getZ() - rows[0].getX() - rows[1].getY()) * 2.0f;
            float invS = 1.0f / s;
            qw = (rows[1].getX() - rows[0].getY()) * invS;
            qx = (rows[0].getZ() + rows[2].getX()) * invS;
            qy = (rows[1].getZ() + rows[2].getY()) * invS;
            qz = 0.25f * s;
        }

        return Quat(qx, qy, qz, qw);
    }

    // ── Comparison ────────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE bool operator==(const Mat3& rhs) const noexcept {
        return rows[0] == rhs.rows[0] && rows[1] == rhs.rows[1] && rows[2] == rhs.rows[2];
    }

    [[nodiscard]] PULSE_FORCE_INLINE bool operator!=(const Mat3& rhs) const noexcept {
        return !(*this == rhs);
    }
};

// ── Free-function operators ───────────────────────────────────────────────────

[[nodiscard]] PULSE_FORCE_INLINE Mat3 operator*(float s, const Mat3& m) noexcept {
    return m * s;
}

// ── Free-function utilities ───────────────────────────────────────────────────

namespace math {
    [[nodiscard]] PULSE_FORCE_INLINE Mat3 transpose(const Mat3& m) noexcept {
        return m.transposed();
    }

    [[nodiscard]] PULSE_FORCE_INLINE Mat3 inverse(const Mat3& m) noexcept {
        return m.inversed();
    }

    [[nodiscard]] PULSE_FORCE_INLINE float determinant(const Mat3& m) noexcept {
        return m.determinant();
    }
} // namespace math

} // namespace pulse
