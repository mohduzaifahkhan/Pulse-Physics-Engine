/**
 * @file transform.h
 * @brief Rigid-body transform — position + rotation (quaternion).
 *
 * Represents a rigid-body transformation (no scale for physics — scale is
 * handled by shapes). This is the fundamental pose representation used throughout
 * the physics engine.
 *
 * Memory layout: Vec3 (16 bytes) + Quat (16 bytes) = 32 bytes, 16-byte aligned.
 * Two transforms fit exactly in one cache line (64 bytes).
 */

#pragma once

#include "math_common.h"
#include "vec3.h"
#include "quat.h"
#include "mat3.h"
#include "mat4.h"

namespace pulse {

/**
 * @struct Transform
 * @brief A rigid-body pose: position (Vec3) + orientation (Quat).
 *
 * No scale component — physics bodies use shapes with their own local scale.
 * 32 bytes, 16-byte aligned. Two transforms per cache line.
 */
struct PULSE_SIMD_ALIGN Transform {
    Vec3 position;   ///< World-space position.
    Quat rotation;   ///< World-space orientation (unit quaternion).

    // ── Constructors ──────────────────────────────────────────────────────

    /// Default: origin, identity rotation.
    PULSE_FORCE_INLINE Transform() noexcept
        : position(Vec3::zero()), rotation(Quat::identity())
    {}

    /// Construct from position and rotation.
    PULSE_FORCE_INLINE Transform(Vec3 pos, Quat rot) noexcept
        : position(pos), rotation(rot)
    {}

    /// Construct from position only (identity rotation).
    explicit PULSE_FORCE_INLINE Transform(Vec3 pos) noexcept
        : position(pos), rotation(Quat::identity())
    {}

    /// Construct from rotation only (origin position).
    explicit PULSE_FORCE_INLINE Transform(Quat rot) noexcept
        : position(Vec3::zero()), rotation(rot)
    {}

    // ── Static factories ──────────────────────────────────────────────────

    [[nodiscard]] static PULSE_FORCE_INLINE Transform identity() noexcept {
        return Transform();
    }

    // ── Transform operations ──────────────────────────────────────────────

    /// Transform a point from local space to world space.
    /// worldPos = rotation * localPos + position
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 transformPoint(Vec3 localPoint) const noexcept {
        return rotation.rotate(localPoint) + position;
    }

    /// Transform a direction from local space to world space (no translation).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 transformDirection(Vec3 localDir) const noexcept {
        return rotation.rotate(localDir);
    }

    /// Inverse-transform a point from world space to local space.
    /// localPos = rotation^{-1} * (worldPos - position)
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 inverseTransformPoint(Vec3 worldPoint) const noexcept {
        return rotation.inverseRotate(worldPoint - position);
    }

    /// Inverse-transform a direction from world space to local space.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 inverseTransformDirection(Vec3 worldDir) const noexcept {
        return rotation.inverseRotate(worldDir);
    }

    // ── Composition ───────────────────────────────────────────────────────

    /// Compose two transforms: result = this * rhs.
    /// Equivalent to applying rhs first, then this.
    [[nodiscard]] PULSE_FORCE_INLINE Transform operator*(Transform rhs) const noexcept {
        return Transform(
            transformPoint(rhs.position),
            rotation * rhs.rotation
        );
    }

    PULSE_FORCE_INLINE Transform& operator*=(Transform rhs) noexcept {
        *this = *this * rhs;
        return *this;
    }

    /// Compute the inverse transform.
    [[nodiscard]] PULSE_FORCE_INLINE Transform inversed() const noexcept {
        Quat invRot = rotation.conjugate();
        return Transform(invRot.rotate(-position), invRot);
    }

    // ── Interpolation ─────────────────────────────────────────────────────

    /// Linearly interpolate position, spherically interpolate rotation.
    [[nodiscard]] PULSE_FORCE_INLINE Transform lerp(Transform target, float t) const noexcept {
        return Transform(
            position.lerp(target.position, t),
            rotation.slerp(target.rotation, t)
        );
    }

    /// Faster interpolation using nlerp for rotation.
    [[nodiscard]] PULSE_FORCE_INLINE Transform nlerp(Transform target, float t) const noexcept {
        return Transform(
            position.lerp(target.position, t),
            rotation.nlerp(target.rotation, t)
        );
    }

    // ── Conversion ────────────────────────────────────────────────────────

    /// Convert to a 4×4 homogeneous matrix.
    [[nodiscard]] PULSE_FORCE_INLINE Mat4 toMat4() const noexcept {
        Mat3 r = Mat3::fromQuat(rotation);
        return Mat4(
            Vec4(r[0], position.getX()),
            Vec4(r[1], position.getY()),
            Vec4(r[2], position.getZ()),
            Vec4(0.0f, 0.0f, 0.0f, 1.0f)
        );
    }

    /// Convert to a 3×3 rotation matrix.
    [[nodiscard]] PULSE_FORCE_INLINE Mat3 toMat3() const noexcept {
        return Mat3::fromQuat(rotation);
    }

    // ── Getters ───────────────────────────────────────────────────────────

    /// Get the forward direction (-Z in local space).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 getForward() const noexcept {
        return rotation.getForward();
    }

    /// Get the up direction (+Y in local space).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 getUp() const noexcept {
        return rotation.getUp();
    }

    /// Get the right direction (+X in local space).
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 getRight() const noexcept {
        return rotation.getRight();
    }

    // ── Comparison ────────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE bool operator==(const Transform& rhs) const noexcept {
        return position == rhs.position && rotation == rhs.rotation;
    }

    [[nodiscard]] PULSE_FORCE_INLINE bool operator!=(const Transform& rhs) const noexcept {
        return !(*this == rhs);
    }
};

// ── Free-function utilities ───────────────────────────────────────────────────

namespace math {
    [[nodiscard]] PULSE_FORCE_INLINE Transform lerp(Transform a, Transform b, float t) noexcept {
        return a.lerp(b, t);
    }

    [[nodiscard]] PULSE_FORCE_INLINE Transform inverse(Transform t) noexcept {
        return t.inversed();
    }
} // namespace math

} // namespace pulse
