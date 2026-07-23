/**
 * @file mpr.h
 * @brief Minkowski Portal Refinement (MPR) — alternative overlap + penetration test.
 *
 * MPR is a simpler alternative to GJK+EPA that finds both overlap and penetration
 * info in a single algorithm. It works by:
 *
 *   Phase 1 (Discovery): Find an interior point of the Minkowski difference and
 *   build a portal (triangle) from support points.
 *
 *   Phase 2 (Refinement): Iteratively refine the portal until it faces the
 *   origin or the origin is proven to be outside.
 *
 *   Phase 3 (Penetration): Extract penetration depth and normal from the
 *   final portal.
 *
 * Advantages over GJK+EPA: simpler code, single-pass, no polytope expansion.
 * Disadvantages: slightly less precise, cannot compute closest distance when
 * shapes are separated (only overlap/penetration).
 *
 * Reference: Gary Snethen, "XenoCollide: Complex Collision Made Simple" (GDC 2008).
 */

#pragma once

#include "narrowphase_common.h"
#include <pulse/shapes/sphere.h>
#include <pulse/shapes/box.h>
#include <pulse/shapes/capsule.h>
#include <pulse/shapes/cylinder.h>
#include <pulse/shapes/convex_hull.h>

namespace pulse {

// ── MPR query ────────────────────────────────────────────────────────────────

/**
 * @brief Run the MPR algorithm to detect overlap and penetration.
 *
 * @tparam ShapeA  Shape type with supportWorld(Vec3 dir, const Transform&).
 * @tparam ShapeB  Shape type with supportWorld(Vec3 dir, const Transform&).
 * @param shapeA   First shape.
 * @param txA      Transform of shape A.
 * @param shapeB   Second shape.
 * @param txB      Transform of shape B.
 * @param result   Output: overlap status, penetration depth/normal, contact point.
 * @param maxIter  Maximum iterations (default 64).
 * @param tolerance  Convergence tolerance (default 1e-4).
 * @return True if shapes overlap.
 */
template <typename ShapeA, typename ShapeB>
static inline bool mprQuery(
    const ShapeA& shapeA, const Transform& txA,
    const ShapeB& shapeB, const Transform& txB,
    MprResult& result,
    uint32_t maxIter = 64,
    float tolerance = 1.0e-4f) noexcept
{
    result = MprResult();

    // ── Phase 0: Interior point (v0) ──
    // Use center of the Minkowski difference (centerA - centerB).
    // For most shapes, the center is at the transform position.
    Vec3 v0 = txA.position - txB.position;

    if (v0.lengthSq() < math::Epsilon * math::Epsilon) {
        v0 = Vec3(math::Epsilon, 0.0f, 0.0f);
    }

    // ── Phase 1: Discover portal ──
    // v1: support in the direction of v0 → origin (i.e., -v0)
    Vec3 searchDir = -v0;
    if (searchDir.lengthSq() < math::Epsilon * math::Epsilon) {
        searchDir = Vec3(1.0f, 0.0f, 0.0f);
    }

    Vec3 v1a = shapeA.supportWorld(searchDir, txA);
    Vec3 v1b = shapeB.supportWorld(-searchDir, txB);
    Vec3 v1 = v1a - v1b;

    // Check if v1 crossed the origin in the search direction
    if (v1.dot(searchDir) < 0.0f) {
        // Origin is not in the Minkowski difference — shapes separated
        result.overlapping = false;
        return false;
    }

    // v2: support perpendicular to the v0-v1 line, toward origin
    Vec3 v0v1 = v1 - v0;
    searchDir = v0v1.cross(-v0).cross(v0v1);
    if (searchDir.lengthSq() < math::Epsilon * math::Epsilon) {
        // v0 and v1 are collinear — pick an arbitrary perpendicular
        if (math::fastAbs(v0v1.getX()) < 0.9f)
            searchDir = v0v1.cross(Vec3(1.0f, 0.0f, 0.0f));
        else
            searchDir = v0v1.cross(Vec3(0.0f, 1.0f, 0.0f));
    }

    Vec3 v2a = shapeA.supportWorld(searchDir, txA);
    Vec3 v2b = shapeB.supportWorld(-searchDir, txB);
    Vec3 v2 = v2a - v2b;

    if (v2.dot(searchDir) < 0.0f) {
        result.overlapping = false;
        return false;
    }

    // v3: support in the direction of the portal normal (v1-v0) × (v2-v0)
    searchDir = (v1 - v0).cross(v2 - v0);
    // Make sure the normal points toward the origin
    if (searchDir.dot(-v0) < 0.0f) {
        // Swap v1 and v2 to flip the portal
        Vec3 tmp;
        tmp = v1; v1 = v2; v2 = tmp;
        tmp = v1a; v1a = v2a; v2a = tmp;
        tmp = v1b; v1b = v2b; v2b = tmp;
        searchDir = -searchDir;
    }

    Vec3 v3a = shapeA.supportWorld(searchDir, txA);
    Vec3 v3b = shapeB.supportWorld(-searchDir, txB);
    Vec3 v3 = v3a - v3b;

    if (v3.dot(searchDir) < 0.0f) {
        result.overlapping = false;
        return false;
    }

    // ── Phase 2: Portal refinement ──
    // We have v0 (interior), v1, v2, v3 (portal triangle).
    // Refine until the portal faces the origin.

    for (uint32_t iter = 0; iter < maxIter; ++iter) {
        // Portal normal
        Vec3 portalNormal = (v2 - v1).cross(v3 - v1);
        float portalLen = portalNormal.length();
        if (portalLen < math::Epsilon) {
            // Degenerate portal
            break;
        }
        portalNormal = portalNormal * (1.0f / portalLen);

        // Check if origin is on the inside of the portal
        // (i.e., the portal's normal dot with (origin - v1) is ≤ 0)
        float originDist = portalNormal.dot(v1);

        if (originDist >= 0.0f) {
            // Origin is behind or on the portal — shapes overlap
            result.overlapping = true;
            result.normal = portalNormal;
            result.penetration = originDist;

            // Compute contact point via barycentric coordinates on the portal
            Vec3 ab = v2 - v1;
            Vec3 ac = v3 - v1;
            Vec3 ap = -v1; // origin - v1

            float d00 = ab.dot(ab);
            float d01 = ab.dot(ac);
            float d11 = ac.dot(ac);
            float d20 = ap.dot(ab);
            float d21 = ap.dot(ac);
            float denom = d00 * d11 - d01 * d01;

            float bv = (d11 * d20 - d01 * d21) / (denom + math::Epsilon);
            float bw = (d00 * d21 - d01 * d20) / (denom + math::Epsilon);
            float bu = 1.0f - bv - bw;

            bu = math::clamp(bu, 0.0f, 1.0f);
            bv = math::clamp(bv, 0.0f, 1.0f);
            bw = math::clamp(bw, 0.0f, 1.0f);
            float sum = bu + bv + bw;
            if (sum > math::Epsilon) { bu /= sum; bv /= sum; bw /= sum; }

            result.pointOnA = v1a * bu + v2a * bv + v3a * bw;
            result.pointOnB = v1b * bu + v2b * bv + v3b * bw;
            return true;
        }

        // Get new support point beyond the portal
        Vec3 v4a = shapeA.supportWorld(portalNormal, txA);
        Vec3 v4b = shapeB.supportWorld(-portalNormal, txB);
        Vec3 v4 = v4a - v4b;

        // Check convergence
        float advancement = v4.dot(portalNormal) - originDist;
        if (advancement < tolerance) {
            // Portal converged — check final state
            if (originDist >= -tolerance) {
                result.overlapping = true;
                result.normal = portalNormal;
                result.penetration = math::fastMax(0.0f, originDist);
                result.pointOnA = (v1a + v2a + v3a) * (1.0f / 3.0f);
                result.pointOnB = (v1b + v2b + v3b) * (1.0f / 3.0f);
                return true;
            }
            result.overlapping = false;
            return false;
        }

        // Determine which edge of the portal to replace.
        // The origin, v0, and v4 define which side of each portal edge the
        // origin lies on. Replace the portal vertex on the opposite side.

        Vec3 cross1 = (v4 - v0).cross(v1 - v0);
        Vec3 cross2 = (v4 - v0).cross(v2 - v0);
        Vec3 cross3 = (v4 - v0).cross(v3 - v0);

        if (cross1.dot(-v0) < 0.0f) {
            if (cross2.dot(-v0) < 0.0f) {
                // Replace v1
                v1 = v4; v1a = v4a; v1b = v4b;
            } else {
                // Replace v3
                v3 = v4; v3a = v4a; v3b = v4b;
            }
        } else {
            if (cross3.dot(-v0) < 0.0f) {
                // Replace v2
                v2 = v4; v2a = v4a; v2b = v4b;
            } else {
                // Replace v1
                v1 = v4; v1a = v4a; v1b = v4b;
            }
        }
    }

    // Max iterations reached — assume no overlap
    result.overlapping = false;
    return false;
}

} // namespace pulse
