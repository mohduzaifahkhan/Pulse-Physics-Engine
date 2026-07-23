/**
 * @file ccd.h
 * @brief Continuous Collision Detection (CCD) — prevents tunneling.
 *
 * Detects collisions between frames for fast-moving objects using
 * Conservative Advancement: iteratively advances the shapes along their
 * linear trajectories, computing closest distance via GJK at each step,
 * and advancing by (distance / relative_velocity). Converges when the
 * distance is below a tolerance or the time exceeds 1.0.
 *
 * Handles linear motion only (translational interpolation between start
 * and end transforms). For rotating bodies, the angular motion bound
 * extends the safe advancement distance.
 */

#pragma once

#include "narrowphase_common.h"
#include "gjk.h"

namespace pulse {

// ── CCD Conservative Advancement ─────────────────────────────────────────────

/**
 * @brief Compute the time of impact between two sweeping convex shapes.
 *
 * Uses conservative advancement: at each step, compute the closest distance
 * via GJK, then advance time by (distance / max_relative_speed).
 *
 * @tparam ShapeA  Shape type with supportWorld().
 * @tparam ShapeB  Shape type with supportWorld().
 * @param shapeA       First shape.
 * @param txA_start    Start transform of shape A.
 * @param txA_end      End transform of shape A.
 * @param shapeB       Second shape.
 * @param txB_start    Start transform of shape B.
 * @param txB_end      End transform of shape B.
 * @param result       Output: time of impact, contact point, normal.
 * @param maxIter      Maximum CCD iterations (default 32).
 * @param tolerance    Distance tolerance for convergence (default 1e-4).
 * @return True if a collision was detected (TOI < 1.0).
 */
template <typename ShapeA, typename ShapeB>
static inline bool ccdQuery(
    const ShapeA& shapeA,
    const Transform& txA_start, const Transform& txA_end,
    const ShapeB& shapeB,
    const Transform& txB_start, const Transform& txB_end,
    CcdResult& result,
    uint32_t maxIter = 32,
    float tolerance = 1.0e-4f) noexcept
{
    result = CcdResult();

    // Compute maximum relative displacement
    Vec3 dispA = txA_end.position - txA_start.position;
    Vec3 dispB = txB_end.position - txB_start.position;
    Vec3 relDisp = dispA - dispB;
    float relSpeed = relDisp.length();

    if (relSpeed < math::Epsilon) {
        // No relative motion — do a static overlap test at t=0
        GjkResult gjkResult;
        gjkQuery(shapeA, txA_start, shapeB, txB_start, gjkResult);
        if (gjkResult.status == GjkStatus::Overlapping) {
            result.timeOfImpact = 0.0f;
            result.point = (txA_start.position + txB_start.position) * 0.5f;
            result.normal = (txA_start.position - txB_start.position);
            float len = result.normal.length();
            if (len > math::Epsilon) result.normal = result.normal * (1.0f / len);
            else result.normal = Vec3(1.0f, 0.0f, 0.0f);
            result.hit = true;
            return true;
        }
        return false;
    }

    // Also account for angular motion as a radial bound
    // For simplicity, estimate max angular displacement per shape
    // This adds a conservative bound to the advancement step
    float angularBoundA = 0.0f;
    float angularBoundB = 0.0f;
    {
        // Approximate angular displacement from quaternion difference
        Quat dqA = txA_end.rotation * txA_start.rotation.conjugate();
        Quat dqB = txB_end.rotation * txB_start.rotation.conjugate();
        // Angle ≈ 2 * acos(|w|), bound the arc ≈ angle * radius
        // We use a conservative estimate: the 'w' component deviation from 1.0
        float wA = math::fastAbs(dqA.getW());
        float wB = math::fastAbs(dqB.getW());
        angularBoundA = (wA < 1.0f - math::Epsilon) ? 2.0f * std::acos(math::clamp(wA, -1.0f, 1.0f)) : 0.0f;
        angularBoundB = (wB < 1.0f - math::Epsilon) ? 2.0f * std::acos(math::clamp(wB, -1.0f, 1.0f)) : 0.0f;
    }

    // Conservative advancement loop
    float t = 0.0f;

    for (uint32_t iter = 0; iter < maxIter; ++iter) {
        // Interpolate transforms at time t
        Transform txA_t = txA_start.lerp(txA_end, t);
        Transform txB_t = txB_start.lerp(txB_end, t);

        // GJK distance query
        GjkResult gjkResult;
        gjkQuery(shapeA, txA_t, shapeB, txB_t, gjkResult);

        if (gjkResult.status == GjkStatus::Overlapping) {
            // Shapes overlap at time t
            result.timeOfImpact = t;
            result.point = (gjkResult.closestOnA + gjkResult.closestOnB) * 0.5f;
            // Use the last known separation direction
            Vec3 n = gjkResult.closestOnA - gjkResult.closestOnB;
            float len = n.length();
            if (len > math::Epsilon) {
                result.normal = n * (1.0f / len);
            } else {
                result.normal = relDisp * (-1.0f / relSpeed);
            }
            result.hit = true;
            return true;
        }

        float dist = gjkResult.distance;

        if (dist < tolerance) {
            // Close enough — report as impact
            result.timeOfImpact = t;
            result.point = (gjkResult.closestOnA + gjkResult.closestOnB) * 0.5f;
            result.normal = gjkResult.closestOnA - gjkResult.closestOnB;
            float len = result.normal.length();
            if (len > math::Epsilon) {
                result.normal = result.normal * (1.0f / len);
            } else {
                result.normal = relDisp * (-1.0f / relSpeed);
            }
            result.hit = true;
            return true;
        }

        // Compute safe advancement step
        // The maximum relative speed includes linear + angular contributions.
        // For angular: angular_speed * max_radius ≈ gives a linear velocity bound.
        // We use a simplified estimate here.
        float totalRelSpeed = relSpeed + angularBoundA + angularBoundB;
        if (totalRelSpeed < math::Epsilon) break;

        float dt = dist / totalRelSpeed;

        // Advance time
        t += dt;

        if (t >= 1.0f) {
            // No collision in the time interval
            break;
        }
    }

    // No collision detected
    result.hit = false;
    result.timeOfImpact = 1.0f;
    return false;
}

} // namespace pulse
