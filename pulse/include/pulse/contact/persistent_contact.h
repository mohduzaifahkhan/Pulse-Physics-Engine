/**
 * @file persistent_contact.h
 * @brief Extended contact point with accumulated solver impulses.
 *
 * Augments the narrow-phase ContactPoint with the state needed for
 * warm-starting the constraint solver: accumulated normal and tangent
 * impulses, position correction terms, and a tangent basis derived
 * from the contact normal.
 *
 * Design: The tangent basis is lazily computed via computeTangents().
 * Two orthonormal tangent vectors are built from the contact normal
 * using a Frisvad-style construction (numerically robust for all
 * normal orientations).
 */

#pragma once

#include "contact_common.h"
#include <pulse/narrowphase/narrowphase_common.h>
#include <pulse/math/vec3.h>
#include <pulse/math/math_common.h>

namespace pulse {

/**
 * @struct PersistentContact
 * @brief A contact point that persists across frames with solver state.
 *
 * Extends the geometric ContactPoint data with accumulated impulses
 * for warm-starting, a tangent basis for friction, and lifecycle
 * metadata for the contact cache.
 */
struct PersistentContact {
    // ── Geometric data (mirrored from ContactPoint) ──────────────────────

    Vec3     positionOnA;   ///< Contact point on body A (world space).
    Vec3     positionOnB;   ///< Contact point on body B (world space).
    Vec3     normal;        ///< Contact normal (world), from B toward A.
    float    penetration;   ///< Penetration depth (positive = overlap).
    uint32_t featureIdA;    ///< Feature ID on shape A (for matching).
    uint32_t featureIdB;    ///< Feature ID on shape B (for matching).

    // ── Solver state ─────────────────────────────────────────────────────

    float normalImpulse;    ///< Accumulated normal impulse (warm start).
    float tangentImpulse0;  ///< Accumulated friction impulse (tangent 0).
    float tangentImpulse1;  ///< Accumulated friction impulse (tangent 1).
    float positionCorrection; ///< Baumgarte / split-impulse correction.

    // ── Tangent basis ────────────────────────────────────────────────────

    Vec3 tangent0;          ///< First tangent direction (perpendicular to normal).
    Vec3 tangent1;          ///< Second tangent direction (perpendicular to normal and tangent0).

    // ── Lifecycle ────────────────────────────────────────────────────────

    uint16_t     contactAge; ///< Number of frames this contact has persisted.
    ContactFlags flags;      ///< Lifecycle flags (New, Persisted, etc.).
    uint8_t      _pad;       ///< Padding.

    // ── Constructors ─────────────────────────────────────────────────────

    /// Default: zero everything.
    PULSE_FORCE_INLINE PersistentContact() noexcept
        : positionOnA(Vec3::zero()),
          positionOnB(Vec3::zero()),
          normal(Vec3::zero()),
          penetration(0.0f),
          featureIdA(0xFFFFFFFFu),
          featureIdB(0xFFFFFFFFu),
          normalImpulse(0.0f),
          tangentImpulse0(0.0f),
          tangentImpulse1(0.0f),
          positionCorrection(0.0f),
          tangent0(Vec3::zero()),
          tangent1(Vec3::zero()),
          contactAge(0),
          flags(ContactFlags::New),
          _pad(0)
    {}

    /// Construct from a narrow-phase ContactPoint.
    PULSE_FORCE_INLINE explicit PersistentContact(const ContactPoint& cp) noexcept
        : positionOnA(cp.positionOnA),
          positionOnB(cp.positionOnB),
          normal(cp.normal),
          penetration(cp.penetration),
          featureIdA(cp.featureIdA),
          featureIdB(cp.featureIdB),
          normalImpulse(0.0f),
          tangentImpulse0(0.0f),
          tangentImpulse1(0.0f),
          positionCorrection(0.0f),
          tangent0(Vec3::zero()),
          tangent1(Vec3::zero()),
          contactAge(0),
          flags(ContactFlags::New),
          _pad(0)
    {
        computeTangents();
    }

    // ── Tangent basis computation ────────────────────────────────────────

    /**
     * @brief Compute an orthonormal tangent basis from the contact normal.
     *
     * Uses Frisvad's method for robust perpendicular construction that
     * avoids singularities when the normal is close to any cardinal axis.
     */
    PULSE_FORCE_INLINE void computeTangents() noexcept {
        float nx = normal.getX();
        float ny = normal.getY();
        float nz = normal.getZ();

        // Frisvad-style: pick the axis least aligned with the normal
        if (math::fastAbs(nx) < 0.9f) {
            // normal is not near ±X — cross with X axis
            Vec3 t = normal.cross(Vec3(1.0f, 0.0f, 0.0f));
            float len = t.length();
            tangent0 = (len > math::Epsilon) ? t * (1.0f / len) : Vec3(0.0f, 1.0f, 0.0f);
        } else {
            // normal is near ±X — cross with Y axis
            Vec3 t = normal.cross(Vec3(0.0f, 1.0f, 0.0f));
            float len = t.length();
            tangent0 = (len > math::Epsilon) ? t * (1.0f / len) : Vec3(0.0f, 0.0f, 1.0f);
        }

        tangent1 = normal.cross(tangent0);
        float len1 = tangent1.length();
        if (len1 > math::Epsilon) {
            tangent1 = tangent1 * (1.0f / len1);
        }
    }

    // ── Update geometry from a new ContactPoint ──────────────────────────

    /**
     * @brief Update the geometric data from a fresh narrow-phase result.
     *
     * Preserves accumulated impulses (the whole point of warm starting)
     * while updating positions, normal, and penetration depth.
     */
    PULSE_FORCE_INLINE void updateGeometry(const ContactPoint& cp) noexcept {
        positionOnA  = cp.positionOnA;
        positionOnB  = cp.positionOnB;
        normal       = cp.normal;
        penetration  = cp.penetration;
        featureIdA   = cp.featureIdA;
        featureIdB   = cp.featureIdB;
        computeTangents();
    }

    // ── Impulse management ───────────────────────────────────────────────

    /// Scale all accumulated impulses by a factor.
    PULSE_FORCE_INLINE void scaleImpulses(float factor) noexcept {
        normalImpulse   *= factor;
        tangentImpulse0 *= factor;
        tangentImpulse1 *= factor;
    }

    /// Clear all accumulated impulses to zero.
    PULSE_FORCE_INLINE void clearImpulses() noexcept {
        normalImpulse   = 0.0f;
        tangentImpulse0 = 0.0f;
        tangentImpulse1 = 0.0f;
    }

    /// Check if this contact carries meaningful impulse data.
    [[nodiscard]] PULSE_FORCE_INLINE bool hasImpulseData() const noexcept {
        return hasFlag(flags, ContactFlags::HasImpulse) &&
               (math::fastAbs(normalImpulse) > math::Epsilon ||
                math::fastAbs(tangentImpulse0) > math::Epsilon ||
                math::fastAbs(tangentImpulse1) > math::Epsilon);
    }
};

} // namespace pulse
