/**
 * @file gjk.h
 * @brief Gilbert-Johnson-Keerthi (GJK) algorithm — closest distance / overlap test.
 *
 * Iterative algorithm that finds the closest point on the Minkowski difference
 * (A ⊖ B) to the origin. If the origin is inside the Minkowski difference,
 * the shapes overlap and the simplex is passed to EPA for penetration depth.
 *
 * Works with any convex shape via support functions. The implementation uses
 * Johnson's algorithm for the nearest-point on sub-simplices (Voronoi region
 * classification of the 1-simplex, 2-simplex, and 3-simplex).
 *
 * Template-based for zero-overhead dispatch when shape types are known at
 * compile time. A type-erased dispatch path is provided via collision_dispatch.h.
 */

#pragma once

#include "narrowphase_common.h"
#include <pulse/shapes/sphere.h>
#include <pulse/shapes/box.h>
#include <pulse/shapes/capsule.h>
#include <pulse/shapes/cylinder.h>
#include <pulse/shapes/convex_hull.h>

namespace pulse {

// ── GJK support function adapter ─────────────────────────────────────────────

namespace gjk_detail {

    /// Compute a Minkowski-difference support point: support(A, d) - support(B, -d).
    template <typename ShapeA, typename ShapeB>
    [[nodiscard]] PULSE_FORCE_INLINE SupportPoint minkowskiSupport(
        const ShapeA& shapeA, const Transform& txA,
        const ShapeB& shapeB, const Transform& txB,
        Vec3 direction) noexcept
    {
        Vec3 pA = shapeA.supportWorld(direction, txA);
        Vec3 pB = shapeB.supportWorld(-direction, txB);
        return SupportPoint(pA, pB);
    }

    // ── Simplex nearest-point (Johnson's algorithm) ──────────────────────

    /// Process a 1-simplex (line segment). Returns the new search direction.
    /// Updates the simplex to the nearest feature.
    static PULSE_FORCE_INLINE Vec3 doSimplex2(Simplex& simplex) noexcept {
        const Vec3& a = simplex[1].point; // Most recently added
        const Vec3& b = simplex[0].point;

        Vec3 ab = b - a;
        Vec3 ao = -a;

        if (ab.dot(ao) > 0.0f) {
            // Origin is between A and B — closest to the edge
            // Direction: perpendicular to AB toward origin
            Vec3 dir = ab.cross(ao).cross(ab);
            if (dir.lengthSq() < math::Epsilon * math::Epsilon) {
                // Origin is on the line — use a perpendicular direction
                // Pick an arbitrary perpendicular
                if (math::fastAbs(ab.getX()) < 0.9f)
                    dir = ab.cross(Vec3(1.0f, 0.0f, 0.0f));
                else
                    dir = ab.cross(Vec3(0.0f, 1.0f, 0.0f));
            }
            simplex.set(simplex[0], simplex[1]);
            return dir;
        } else {
            // Origin is closest to A — reduce to point
            simplex.set(simplex[1]);
            return ao;
        }
    }

    /// Process a 2-simplex (triangle). Returns the new search direction.
    static PULSE_FORCE_INLINE Vec3 doSimplex3(Simplex& simplex) noexcept {
        const Vec3& a = simplex[2].point; // Most recently added
        const Vec3& b = simplex[1].point;
        const Vec3& c = simplex[0].point;

        Vec3 ab = b - a;
        Vec3 ac = c - a;
        Vec3 ao = -a;
        Vec3 abc = ab.cross(ac); // Triangle normal

        // Test edge AB region
        Vec3 abPerp = ab.cross(abc);
        if (abPerp.dot(ao) > 0.0f) {
            if (ab.dot(ao) > 0.0f) {
                simplex.set(simplex[1], simplex[2]);
                return ab.cross(ao).cross(ab);
            } else {
                simplex.set(simplex[2]);
                return ao;
            }
        }

        // Test edge AC region
        Vec3 acPerp = abc.cross(ac);
        if (acPerp.dot(ao) > 0.0f) {
            if (ac.dot(ao) > 0.0f) {
                simplex.set(simplex[0], simplex[2]);
                return ac.cross(ao).cross(ac);
            } else {
                simplex.set(simplex[2]);
                return ao;
            }
        }

        // Origin is above or below the triangle
        if (abc.dot(ao) > 0.0f) {
            // Above — keep winding, search above
            simplex.set(simplex[0], simplex[1], simplex[2]);
            return abc;
        } else {
            // Below — reverse winding, search below
            simplex.set(simplex[1], simplex[0], simplex[2]);
            return -abc;
        }
    }

    /// Process a 3-simplex (tetrahedron). Returns true if origin is inside.
    /// If false, reduces the simplex and updates the search direction.
    static PULSE_FORCE_INLINE bool doSimplex4(Simplex& simplex, Vec3& direction) noexcept {
        const Vec3& a = simplex[3].point; // Most recently added
        const Vec3& b = simplex[2].point;
        const Vec3& c = simplex[1].point;
        const Vec3& d = simplex[0].point;

        Vec3 ab = b - a;
        Vec3 ac = c - a;
        Vec3 ad = d - a;
        Vec3 ao = -a;

        Vec3 abc = ab.cross(ac);
        Vec3 acd = ac.cross(ad);
        Vec3 adb = ad.cross(ab);

        // Check each face — if the origin is outside any face, reduce to that triangle
        if (abc.dot(ao) > 0.0f) {
            simplex.set(simplex[1], simplex[2], simplex[3]);
            direction = doSimplex3(simplex);
            return false;
        }

        if (acd.dot(ao) > 0.0f) {
            simplex.set(simplex[0], simplex[1], simplex[3]);
            direction = doSimplex3(simplex);
            return false;
        }

        if (adb.dot(ao) > 0.0f) {
            simplex.set(simplex[0], simplex[2], simplex[3]);
            direction = doSimplex3(simplex);
            return false;
        }

        // Origin is inside the tetrahedron → shapes overlap
        return true;
    }

} // namespace gjk_detail

// ── GJK query ────────────────────────────────────────────────────────────────

/**
 * @brief Run the GJK algorithm to determine overlap or closest distance.
 *
 * @tparam ShapeA  Shape type with supportWorld(Vec3 dir, const Transform&) method.
 * @tparam ShapeB  Shape type with supportWorld(Vec3 dir, const Transform&) method.
 * @param shapeA   First shape.
 * @param txA      Transform of shape A.
 * @param shapeB   Second shape.
 * @param txB      Transform of shape B.
 * @param result   Output: closest points, distance, simplex, status.
 * @param maxIter  Maximum iterations (default 64).
 */
template <typename ShapeA, typename ShapeB>
static inline void gjkQuery(
    const ShapeA& shapeA, const Transform& txA,
    const ShapeB& shapeB, const Transform& txB,
    GjkResult& result,
    uint32_t maxIter = 64) noexcept
{
    result = GjkResult();

    // Initial search direction: from center of B to center of A
    Vec3 direction = txA.position - txB.position;
    if (direction.lengthSq() < math::Epsilon * math::Epsilon) {
        direction = Vec3(1.0f, 0.0f, 0.0f); // Arbitrary if shapes are at same position
    }

    // Get first support point
    SupportPoint sp = gjk_detail::minkowskiSupport(shapeA, txA, shapeB, txB, direction);
    Simplex& simplex = result.simplex;
    simplex.addVertex(sp);

    // New search direction: toward the origin from the first support point
    direction = -sp.point;

    if (direction.lengthSq() < math::Epsilon * math::Epsilon) {
        // First support point is at the origin — shapes overlap at their centers
        result.status = GjkStatus::Overlapping;
        result.distance = 0.0f;
        result.iterations = 0;
        return;
    }

    for (uint32_t iter = 0; iter < maxIter; ++iter) {
        result.iterations = iter + 1;

        // Get new support point
        sp = gjk_detail::minkowskiSupport(shapeA, txA, shapeB, txB, direction);

        // Check if we passed the origin
        float progress = sp.point.dot(direction);
        if (progress < 0.0f) {
            // Did not pass the origin — shapes are separated
            result.status = GjkStatus::Separated;

            // Extract closest points from the simplex
            // Use the last search direction as the separation axis
            float dist = direction.length();
            if (dist > math::Epsilon) {
                Vec3 normalizedDir = direction * (1.0f / dist);

                // Compute closest points based on simplex size
                if (simplex.size == 1) {
                    result.closestOnA = simplex[0].pointA;
                    result.closestOnB = simplex[0].pointB;
                } else if (simplex.size == 2) {
                    // Closest point on line segment to origin
                    Vec3 a = simplex[1].point;
                    Vec3 b = simplex[0].point;
                    Vec3 ab = b - a;
                    float t = math::clamp(-a.dot(ab) / (ab.lengthSq() + math::Epsilon), 0.0f, 1.0f);
                    result.closestOnA = simplex[1].pointA + (simplex[0].pointA - simplex[1].pointA) * t;
                    result.closestOnB = simplex[1].pointB + (simplex[0].pointB - simplex[1].pointB) * t;
                } else {
                    // Triangle — barycentric coordinates
                    Vec3 a = simplex[2].point;
                    Vec3 b = simplex[1].point;
                    Vec3 c = simplex[0].point;
                    Vec3 ab = b - a;
                    Vec3 ac = c - a;
                    Vec3 ap = -a;

                    float d00 = ab.dot(ab);
                    float d01 = ab.dot(ac);
                    float d11 = ac.dot(ac);
                    float d20 = ap.dot(ab);
                    float d21 = ap.dot(ac);
                    float denom = d00 * d11 - d01 * d01;

                    float v = (d11 * d20 - d01 * d21) / (denom + math::Epsilon);
                    float w = (d00 * d21 - d01 * d20) / (denom + math::Epsilon);
                    float u = 1.0f - v - w;

                    v = math::clamp(v, 0.0f, 1.0f);
                    w = math::clamp(w, 0.0f, 1.0f);
                    u = math::clamp(u, 0.0f, 1.0f);
                    float sum = u + v + w;
                    if (sum > math::Epsilon) { u /= sum; v /= sum; w /= sum; }

                    result.closestOnA = simplex[2].pointA * u + simplex[1].pointA * v + simplex[0].pointA * w;
                    result.closestOnB = simplex[2].pointB * u + simplex[1].pointB * v + simplex[0].pointB * w;
                }

                result.distance = (result.closestOnA - result.closestOnB).length();
            }
            return;
        }

        // Add the new point to the simplex
        simplex.addVertex(sp);

        // Evolve the simplex
        switch (simplex.size) {
        case 2:
            direction = gjk_detail::doSimplex2(simplex);
            break;
        case 3:
            direction = gjk_detail::doSimplex3(simplex);
            break;
        case 4:
            if (gjk_detail::doSimplex4(simplex, direction)) {
                // Origin is inside the tetrahedron — overlap
                result.status = GjkStatus::Overlapping;
                result.distance = 0.0f;
                return;
            }
            break;
        default:
            break;
        }

        // Check for convergence (direction is near-zero)
        if (direction.lengthSq() < math::Epsilon * math::Epsilon) {
            result.status = GjkStatus::Overlapping;
            result.distance = 0.0f;
            return;
        }
    }

    // Failed to converge
    result.status = GjkStatus::MaxIterations;
}

// ── GJK distance query (convenience wrapper) ─────────────────────────────────

/**
 * @brief Compute the closest distance between two convex shapes.
 *
 * @return The distance (0 if overlapping or failed to converge).
 */
template <typename ShapeA, typename ShapeB>
[[nodiscard]] static inline float gjkDistance(
    const ShapeA& shapeA, const Transform& txA,
    const ShapeB& shapeB, const Transform& txB,
    uint32_t maxIter = 64) noexcept
{
    GjkResult result;
    gjkQuery(shapeA, txA, shapeB, txB, result, maxIter);
    return result.distance;
}

} // namespace pulse
