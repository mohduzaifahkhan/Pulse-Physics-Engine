/**
 * @file mat4.h
 * @brief 4×4 matrix — SSE4.2 accelerated, row-major layout.
 *
 * Stored as four __m128 rows. Row-major convention: M * v transforms a column
 * vector. The matrix multiply is fully SIMD-optimized using broadcast-multiply-add
 * chains, which is the fastest known approach on SSE.
 *
 * Primary uses:
 * - Model/View/Projection transforms
 * - Homogeneous coordinate transforms
 * - Extracting 3×3 rotation/scale sub-matrix
 *
 * Memory layout: 4× __m128 = 64 bytes = exactly one cache line. 16-byte aligned.
 */

#pragma once

#include "math_common.h"
#include "vec3.h"
#include "vec4.h"
#include "mat3.h"
#include "quat.h"

namespace pulse {

/**
 * @struct Mat4
 * @brief A 4×4 matrix stored as 4 row vectors (__m128 each).
 *
 * Row-major convention. Fits exactly in one 64-byte cache line.
 */
struct PULSE_SIMD_ALIGN Mat4 {
    Vec4 rows[4];

    // ── Constructors ──────────────────────────────────────────────────────

    /// Default: identity.
    PULSE_FORCE_INLINE Mat4() noexcept
        : rows{
            Vec4(1.0f, 0.0f, 0.0f, 0.0f),
            Vec4(0.0f, 1.0f, 0.0f, 0.0f),
            Vec4(0.0f, 0.0f, 1.0f, 0.0f),
            Vec4(0.0f, 0.0f, 0.0f, 1.0f)
        }
    {}

    /// Construct from 4 row vectors.
    PULSE_FORCE_INLINE Mat4(Vec4 r0, Vec4 r1, Vec4 r2, Vec4 r3) noexcept
        : rows{r0, r1, r2, r3}
    {}

    /// Construct from 16 scalars (row-major order).
    PULSE_FORCE_INLINE Mat4(
        float m00, float m01, float m02, float m03,
        float m10, float m11, float m12, float m13,
        float m20, float m21, float m22, float m23,
        float m30, float m31, float m32, float m33
    ) noexcept
        : rows{
            Vec4(m00, m01, m02, m03),
            Vec4(m10, m11, m12, m13),
            Vec4(m20, m21, m22, m23),
            Vec4(m30, m31, m32, m33)
        }
    {}

    // ── Static factories ──────────────────────────────────────────────────

    [[nodiscard]] static PULSE_FORCE_INLINE Mat4 identity() noexcept { return Mat4(); }

    [[nodiscard]] static PULSE_FORCE_INLINE Mat4 zero() noexcept {
        return Mat4(Vec4::zero(), Vec4::zero(), Vec4::zero(), Vec4::zero());
    }

    /// Translation matrix.
    [[nodiscard]] static PULSE_FORCE_INLINE Mat4 translation(Vec3 t) noexcept {
        return Mat4(
            1.0f, 0.0f, 0.0f, t.getX(),
            0.0f, 1.0f, 0.0f, t.getY(),
            0.0f, 0.0f, 1.0f, t.getZ(),
            0.0f, 0.0f, 0.0f, 1.0f
        );
    }

    /// Uniform scale matrix.
    [[nodiscard]] static PULSE_FORCE_INLINE Mat4 scale(float s) noexcept {
        return Mat4(
            s, 0.0f, 0.0f, 0.0f,
            0.0f, s, 0.0f, 0.0f,
            0.0f, 0.0f, s, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        );
    }

    /// Non-uniform scale matrix.
    [[nodiscard]] static PULSE_FORCE_INLINE Mat4 scale(Vec3 s) noexcept {
        return Mat4(
            s.getX(), 0.0f,     0.0f,     0.0f,
            0.0f,     s.getY(), 0.0f,     0.0f,
            0.0f,     0.0f,     s.getZ(), 0.0f,
            0.0f,     0.0f,     0.0f,     1.0f
        );
    }

    /// Rotation matrix from quaternion.
    [[nodiscard]] static PULSE_FORCE_INLINE Mat4 rotation(Quat q) noexcept {
        Mat3 r = Mat3::fromQuat(q);
        return Mat4(
            Vec4(r[0], 0.0f),
            Vec4(r[1], 0.0f),
            Vec4(r[2], 0.0f),
            Vec4(0.0f, 0.0f, 0.0f, 1.0f)
        );
    }

    /// Combined TRS matrix: Translation * Rotation * Scale.
    [[nodiscard]] static PULSE_FORCE_INLINE Mat4 trs(Vec3 pos, Quat rot, Vec3 scl) noexcept {
        Mat3 r = Mat3::fromQuat(rot);
        // Scale the rotation matrix columns
        Vec3 r0 = r[0] * scl.getX();
        Vec3 r1 = r[1] * scl.getY();
        Vec3 r2 = r[2] * scl.getZ();
        return Mat4(
            Vec4(r0, pos.getX()),
            Vec4(r1, pos.getY()),
            Vec4(r2, pos.getZ()),
            Vec4(0.0f, 0.0f, 0.0f, 1.0f)
        );
    }

    /// Look-at matrix (right-handed, for view transforms).
    [[nodiscard]] static PULSE_FORCE_INLINE Mat4 lookAt(Vec3 eye, Vec3 target, Vec3 worldUp) noexcept {
        Vec3 f = (target - eye).normalized(); // Forward
        Vec3 r = f.cross(worldUp).normalized(); // Right
        Vec3 u = r.cross(f); // True up

        return Mat4(
             r.getX(),  r.getY(),  r.getZ(), -r.dot(eye),
             u.getX(),  u.getY(),  u.getZ(), -u.dot(eye),
            -f.getX(), -f.getY(), -f.getZ(),  f.dot(eye),
             0.0f,      0.0f,      0.0f,      1.0f
        );
    }

    /// Perspective projection matrix (right-handed, depth [0, 1]).
    [[nodiscard]] static PULSE_FORCE_INLINE Mat4 perspective(float fovY, float aspect, float nearZ, float farZ) noexcept {
        const float tanHalfFov = std::tan(fovY * 0.5f);
        const float invRange = 1.0f / (nearZ - farZ);

        return Mat4(
            1.0f / (aspect * tanHalfFov), 0.0f,                  0.0f,                       0.0f,
            0.0f,                          1.0f / tanHalfFov,     0.0f,                       0.0f,
            0.0f,                          0.0f,                  farZ * invRange,            nearZ * farZ * invRange,
            0.0f,                          0.0f,                 -1.0f,                       0.0f
        );
    }

    /// Orthographic projection matrix.
    [[nodiscard]] static PULSE_FORCE_INLINE Mat4 ortho(float left, float right, float bottom, float top, float nearZ, float farZ) noexcept {
        const float invWidth  = 1.0f / (right - left);
        const float invHeight = 1.0f / (top - bottom);
        const float invDepth  = 1.0f / (nearZ - farZ);

        return Mat4(
            2.0f * invWidth, 0.0f,             0.0f,           -(right + left) * invWidth,
            0.0f,            2.0f * invHeight,  0.0f,           -(top + bottom) * invHeight,
            0.0f,            0.0f,              invDepth,        nearZ * invDepth,
            0.0f,            0.0f,              0.0f,            1.0f
        );
    }

    // ── Element access ────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE Vec4&       operator[](int row)       noexcept { return rows[row]; }
    [[nodiscard]] PULSE_FORCE_INLINE const Vec4& operator[](int row) const noexcept { return rows[row]; }

    [[nodiscard]] PULSE_FORCE_INLINE float operator()(int row, int col) const noexcept {
        return rows[row][col];
    }

    [[nodiscard]] PULSE_FORCE_INLINE Vec4 column(int col) const noexcept {
        return Vec4(rows[0][col], rows[1][col], rows[2][col], rows[3][col]);
    }

    /// Extract the upper-left 3×3 sub-matrix (rotation + scale).
    [[nodiscard]] PULSE_FORCE_INLINE Mat3 upperLeft3x3() const noexcept {
        return Mat3(
            rows[0].xyz(),
            rows[1].xyz(),
            rows[2].xyz()
        );
    }

    /// Extract the translation component.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 getTranslation() const noexcept {
        return Vec3(rows[0].getW(), rows[1].getW(), rows[2].getW());
    }

    // ── Arithmetic ────────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE Mat4 operator+(const Mat4& rhs) const noexcept {
        return Mat4(
            rows[0] + rhs.rows[0], rows[1] + rhs.rows[1],
            rows[2] + rhs.rows[2], rows[3] + rhs.rows[3]
        );
    }

    [[nodiscard]] PULSE_FORCE_INLINE Mat4 operator-(const Mat4& rhs) const noexcept {
        return Mat4(
            rows[0] - rhs.rows[0], rows[1] - rhs.rows[1],
            rows[2] - rhs.rows[2], rows[3] - rhs.rows[3]
        );
    }

    [[nodiscard]] PULSE_FORCE_INLINE Mat4 operator*(float s) const noexcept {
        return Mat4(rows[0] * s, rows[1] * s, rows[2] * s, rows[3] * s);
    }

    /// Matrix * Vec4.
    [[nodiscard]] PULSE_FORCE_INLINE Vec4 operator*(Vec4 v) const noexcept {
#ifdef PULSE_SIMD_SSE42
        // Row dot products using dp_ps
        __m128 r0 = _mm_dp_ps(rows[0].m128, v.m128, 0xF1);
        __m128 r1 = _mm_dp_ps(rows[1].m128, v.m128, 0xF2);
        __m128 r2 = _mm_dp_ps(rows[2].m128, v.m128, 0xF4);
        __m128 r3 = _mm_dp_ps(rows[3].m128, v.m128, 0xF8);
        return Vec4(_mm_or_ps(_mm_or_ps(r0, r1), _mm_or_ps(r2, r3)));
#else
        return Vec4(
            rows[0].dot(v), rows[1].dot(v),
            rows[2].dot(v), rows[3].dot(v)
        );
#endif
    }

    /// Matrix * Matrix — the most performance-critical operation.
    /// Uses broadcast-multiply-add: each row of result = sum of (this_row_element_j * rhs_row_j).
    [[nodiscard]] PULSE_FORCE_INLINE Mat4 operator*(const Mat4& rhs) const noexcept {
#ifdef PULSE_SIMD_SSE42
        Mat4 result;
        for (int i = 0; i < 4; ++i) {
            __m128 row = rows[i].m128;
            // Broadcast each element of 'row' and multiply with corresponding rhs row
            __m128 e0 = _mm_shuffle_ps(row, row, _MM_SHUFFLE(0, 0, 0, 0));
            __m128 e1 = _mm_shuffle_ps(row, row, _MM_SHUFFLE(1, 1, 1, 1));
            __m128 e2 = _mm_shuffle_ps(row, row, _MM_SHUFFLE(2, 2, 2, 2));
            __m128 e3 = _mm_shuffle_ps(row, row, _MM_SHUFFLE(3, 3, 3, 3));

            __m128 res = _mm_mul_ps(e0, rhs.rows[0].m128);
            res = _mm_add_ps(res, _mm_mul_ps(e1, rhs.rows[1].m128));
            res = _mm_add_ps(res, _mm_mul_ps(e2, rhs.rows[2].m128));
            res = _mm_add_ps(res, _mm_mul_ps(e3, rhs.rows[3].m128));

            result.rows[i] = Vec4(res);
        }
        return result;
#else
        Mat4 result;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    sum += rows[i][k] * rhs.rows[k][j];
                }
                result.rows[i] = Vec4(
                    (j == 0) ? sum : result.rows[i].getX(),
                    (j == 1) ? sum : result.rows[i].getY(),
                    (j == 2) ? sum : result.rows[i].getZ(),
                    (j == 3) ? sum : result.rows[i].getW()
                );
            }
        }
        return result;
#endif
    }

    /// Transform a Vec3 as a point (w=1): applies rotation + translation.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 transformPoint(Vec3 p) const noexcept {
        Vec4 v = *this * Vec4(p, 1.0f);
        return v.xyz();
    }

    /// Transform a Vec3 as a direction (w=0): applies rotation only, no translation.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 transformDirection(Vec3 d) const noexcept {
        Vec4 v = *this * Vec4(d, 0.0f);
        return v.xyz();
    }

    PULSE_FORCE_INLINE Mat4& operator*=(const Mat4& rhs) noexcept {
        *this = *this * rhs;
        return *this;
    }

    PULSE_FORCE_INLINE Mat4& operator*=(float s) noexcept {
        rows[0] *= s; rows[1] *= s; rows[2] *= s; rows[3] *= s;
        return *this;
    }

    // ── Transpose ─────────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE Mat4 transposed() const noexcept {
#ifdef PULSE_SIMD_SSE42
        __m128 r0 = rows[0].m128;
        __m128 r1 = rows[1].m128;
        __m128 r2 = rows[2].m128;
        __m128 r3 = rows[3].m128;

        // Standard 4×4 SSE transpose
        __m128 t0 = _mm_unpacklo_ps(r0, r1); // [a00, a10, a01, a11]
        __m128 t1 = _mm_unpackhi_ps(r0, r1); // [a02, a12, a03, a13]
        __m128 t2 = _mm_unpacklo_ps(r2, r3); // [a20, a30, a21, a31]
        __m128 t3 = _mm_unpackhi_ps(r2, r3); // [a22, a32, a23, a33]

        return Mat4(
            Vec4(_mm_movelh_ps(t0, t2)),  // [a00, a10, a20, a30]
            Vec4(_mm_movehl_ps(t2, t0)),  // [a01, a11, a21, a31]
            Vec4(_mm_movelh_ps(t1, t3)),  // [a02, a12, a22, a32]
            Vec4(_mm_movehl_ps(t3, t1))   // [a03, a13, a23, a33]
        );
#else
        return Mat4(column(0), column(1), column(2), column(3));
#endif
    }

    // ── Determinant ───────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE float determinant() const noexcept {
        // Laplace expansion along the first row
        const float a = rows[0][0], b = rows[0][1], c = rows[0][2], d = rows[0][3];

        // 3×3 sub-determinants (cofactors)
        float det00 = rows[1][1] * (rows[2][2] * rows[3][3] - rows[2][3] * rows[3][2])
                     - rows[1][2] * (rows[2][1] * rows[3][3] - rows[2][3] * rows[3][1])
                     + rows[1][3] * (rows[2][1] * rows[3][2] - rows[2][2] * rows[3][1]);

        float det01 = rows[1][0] * (rows[2][2] * rows[3][3] - rows[2][3] * rows[3][2])
                     - rows[1][2] * (rows[2][0] * rows[3][3] - rows[2][3] * rows[3][0])
                     + rows[1][3] * (rows[2][0] * rows[3][2] - rows[2][2] * rows[3][0]);

        float det02 = rows[1][0] * (rows[2][1] * rows[3][3] - rows[2][3] * rows[3][1])
                     - rows[1][1] * (rows[2][0] * rows[3][3] - rows[2][3] * rows[3][0])
                     + rows[1][3] * (rows[2][0] * rows[3][1] - rows[2][1] * rows[3][0]);

        float det03 = rows[1][0] * (rows[2][1] * rows[3][2] - rows[2][2] * rows[3][1])
                     - rows[1][1] * (rows[2][0] * rows[3][2] - rows[2][2] * rows[3][0])
                     + rows[1][2] * (rows[2][0] * rows[3][1] - rows[2][1] * rows[3][0]);

        return a * det00 - b * det01 + c * det02 - d * det03;
    }

    // ── Inverse ───────────────────────────────────────────────────────────

    /// Full 4×4 inverse via cofactor expansion.
    [[nodiscard]] PULSE_FORCE_INLINE Mat4 inversed() const noexcept {
        // Compute all 2×2 sub-determinants for rows 2-3
        float s0 = rows[0][0] * rows[1][1] - rows[1][0] * rows[0][1];
        float s1 = rows[0][0] * rows[1][2] - rows[1][0] * rows[0][2];
        float s2 = rows[0][0] * rows[1][3] - rows[1][0] * rows[0][3];
        float s3 = rows[0][1] * rows[1][2] - rows[1][1] * rows[0][2];
        float s4 = rows[0][1] * rows[1][3] - rows[1][1] * rows[0][3];
        float s5 = rows[0][2] * rows[1][3] - rows[1][2] * rows[0][3];

        float c5 = rows[2][2] * rows[3][3] - rows[3][2] * rows[2][3];
        float c4 = rows[2][1] * rows[3][3] - rows[3][1] * rows[2][3];
        float c3 = rows[2][1] * rows[3][2] - rows[3][1] * rows[2][2];
        float c2 = rows[2][0] * rows[3][3] - rows[3][0] * rows[2][3];
        float c1 = rows[2][0] * rows[3][2] - rows[3][0] * rows[2][2];
        float c0 = rows[2][0] * rows[3][1] - rows[3][0] * rows[2][1];

        float det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
        float invDet = 1.0f / det;

        Mat4 result;

        result.rows[0] = Vec4(
            ( rows[1][1] * c5 - rows[1][2] * c4 + rows[1][3] * c3) * invDet,
            (-rows[0][1] * c5 + rows[0][2] * c4 - rows[0][3] * c3) * invDet,
            ( rows[3][1] * s5 - rows[3][2] * s4 + rows[3][3] * s3) * invDet,
            (-rows[2][1] * s5 + rows[2][2] * s4 - rows[2][3] * s3) * invDet
        );

        result.rows[1] = Vec4(
            (-rows[1][0] * c5 + rows[1][2] * c2 - rows[1][3] * c1) * invDet,
            ( rows[0][0] * c5 - rows[0][2] * c2 + rows[0][3] * c1) * invDet,
            (-rows[3][0] * s5 + rows[3][2] * s2 - rows[3][3] * s1) * invDet,
            ( rows[2][0] * s5 - rows[2][2] * s2 + rows[2][3] * s1) * invDet
        );

        result.rows[2] = Vec4(
            ( rows[1][0] * c4 - rows[1][1] * c2 + rows[1][3] * c0) * invDet,
            (-rows[0][0] * c4 + rows[0][1] * c2 - rows[0][3] * c0) * invDet,
            ( rows[3][0] * s4 - rows[3][1] * s2 + rows[3][3] * s0) * invDet,
            (-rows[2][0] * s4 + rows[2][1] * s2 - rows[2][3] * s0) * invDet
        );

        result.rows[3] = Vec4(
            (-rows[1][0] * c3 + rows[1][1] * c1 - rows[1][2] * c0) * invDet,
            ( rows[0][0] * c3 - rows[0][1] * c1 + rows[0][2] * c0) * invDet,
            (-rows[3][0] * s3 + rows[3][1] * s1 - rows[3][2] * s0) * invDet,
            ( rows[2][0] * s3 - rows[2][1] * s1 + rows[2][2] * s0) * invDet
        );

        return result;
    }

    /// Fast inverse for affine matrices (rotation + translation, no perspective).
    /// Much cheaper than full inverse — assumes last row is [0, 0, 0, 1].
    [[nodiscard]] PULSE_FORCE_INLINE Mat4 affineInverse() const noexcept {
        Mat3 r = upperLeft3x3();
        Mat3 rInv = r.inversed();
        Vec3 t = getTranslation();
        Vec3 tInv = -(rInv * t);
        return Mat4(
            Vec4(rInv[0], tInv.getX()),
            Vec4(rInv[1], tInv.getY()),
            Vec4(rInv[2], tInv.getZ()),
            Vec4(0.0f, 0.0f, 0.0f, 1.0f)
        );
    }

    // ── Comparison ────────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE bool operator==(const Mat4& rhs) const noexcept {
        return rows[0] == rhs.rows[0] && rows[1] == rhs.rows[1] &&
               rows[2] == rhs.rows[2] && rows[3] == rhs.rows[3];
    }

    [[nodiscard]] PULSE_FORCE_INLINE bool operator!=(const Mat4& rhs) const noexcept {
        return !(*this == rhs);
    }
};

// ── Free-function operators ───────────────────────────────────────────────────

[[nodiscard]] PULSE_FORCE_INLINE Mat4 operator*(float s, const Mat4& m) noexcept {
    return m * s;
}

// ── Free-function utilities ───────────────────────────────────────────────────

namespace math {
    [[nodiscard]] PULSE_FORCE_INLINE Mat4 transpose(const Mat4& m) noexcept {
        return m.transposed();
    }

    [[nodiscard]] PULSE_FORCE_INLINE Mat4 inverse(const Mat4& m) noexcept {
        return m.inversed();
    }

    [[nodiscard]] PULSE_FORCE_INLINE Mat4 affineInverse(const Mat4& m) noexcept {
        return m.affineInverse();
    }

    [[nodiscard]] PULSE_FORCE_INLINE float determinant(const Mat4& m) noexcept {
        return m.determinant();
    }
} // namespace math

} // namespace pulse
