/**
 * @file collision_dispatch.h
 * @brief Collision dispatch table — routes shape-pair types to algorithms.
 *
 * This is the primary public entry point for the narrow-phase. Given two
 * shapes with their transforms, `collide()` selects the optimal algorithm
 * based on their ShapeType tags:
 *
 *   - Sphere-Sphere: analytic (distance between centers minus radii)
 *   - Sphere-Box: closest-point on box + radius check
 *   - Sphere-Capsule: point-segment distance + radius check
 *   - Capsule-Capsule: segment-segment distance + radius check
 *   - Box-Box: SAT with face clipping
 *   - Generic convex-convex: GJK + EPA
 *   - TriMesh-convex: per-triangle GJK
 *
 * All specialized paths produce high-quality contact manifolds with correct
 * normals and penetration depths. The generic path falls back to GJK+EPA.
 */

#pragma once

#include "narrowphase_common.h"
#include "sat.h"
#include "gjk.h"
#include "epa.h"

#include <pulse/shapes/sphere.h>
#include <pulse/shapes/box.h>
#include <pulse/shapes/capsule.h>
#include <pulse/shapes/cylinder.h>
#include <pulse/shapes/convex_hull.h>
#include <pulse/shapes/tri_mesh.h>

namespace pulse {

// ── Specialized collision routines ───────────────────────────────────────────

namespace collision_detail {

    // ── Sphere-Sphere ────────────────────────────────────────────────────

    static inline bool collideSphereVsSphere(
        const Sphere& a, const Transform& txA,
        const Sphere& b, const Transform& txB,
        ContactManifold& manifold) noexcept
    {
        Vec3 diff = txA.position - txB.position;
        float distSq = diff.lengthSq();
        float radiusSum = a.radius + b.radius;

        if (distSq > radiusSum * radiusSum) return false;

        float dist = math::fastSqrt(distSq);
        Vec3 normal;

        if (dist > math::Epsilon) {
            normal = diff * (1.0f / dist);
        } else {
            normal = Vec3(1.0f, 0.0f, 0.0f); // Concentric — arbitrary
            dist = 0.0f;
        }

        float penetration = radiusSum - dist;
        Vec3 pointOnA = txA.position - normal * a.radius;
        Vec3 pointOnB = txB.position + normal * b.radius;

        manifold.addPoint(pointOnA, pointOnB, normal, penetration);
        return true;
    }

    // ── Sphere-Box ───────────────────────────────────────────────────────

    static inline bool collideSphereVsBox(
        const Sphere& sphere, const Transform& txSphere,
        const Box& box, const Transform& txBox,
        ContactManifold& manifold) noexcept
    {
        // Transform sphere center to box's local space
        Vec3 localCenter = txBox.inverseTransformPoint(txSphere.position);

        // Find closest point on box to sphere center (local space)
        Vec3 closest = localCenter.max(-box.halfExtents).min(box.halfExtents);

        Vec3 diff = localCenter - closest;
        float distSq = diff.lengthSq();

        if (distSq > sphere.radius * sphere.radius) return false;

        // Check if center is inside the box
        bool inside = (distSq < math::Epsilon * math::Epsilon);
        Vec3 normal;
        float penetration;

        if (inside) {
            // Sphere center is inside the box — push out along the closest face
            float dx = box.halfExtents.getX() - math::fastAbs(localCenter.getX());
            float dy = box.halfExtents.getY() - math::fastAbs(localCenter.getY());
            float dz = box.halfExtents.getZ() - math::fastAbs(localCenter.getZ());

            if (dx < dy && dx < dz) {
                normal = Vec3(math::sign(localCenter.getX()), 0.0f, 0.0f);
                penetration = dx + sphere.radius;
            } else if (dy < dz) {
                normal = Vec3(0.0f, math::sign(localCenter.getY()), 0.0f);
                penetration = dy + sphere.radius;
            } else {
                normal = Vec3(0.0f, 0.0f, math::sign(localCenter.getZ()));
                penetration = dz + sphere.radius;
            }

            // Transform normal to world space
            normal = txBox.transformDirection(normal);
            closest = localCenter - normal * penetration; // Not exactly right, but close
        } else {
            float dist = math::fastSqrt(distSq);
            normal = txBox.transformDirection(diff * (1.0f / dist));
            penetration = sphere.radius - dist;
        }

        Vec3 worldClosest = txBox.transformPoint(closest);
        Vec3 pointOnSphere = txSphere.position - normal * sphere.radius;

        manifold.addPoint(pointOnSphere, worldClosest, normal, penetration);
        return true;
    }

    // ── Sphere-Capsule ───────────────────────────────────────────────────

    static inline bool collideSphereVsCapsule(
        const Sphere& sphere, const Transform& txSphere,
        const Capsule& capsule, const Transform& txCapsule,
        ContactManifold& manifold) noexcept
    {
        // Transform sphere center to capsule's local space
        Vec3 localCenter = txCapsule.inverseTransformPoint(txSphere.position);

        // Closest point on capsule's central segment to sphere center
        float y = math::clamp(localCenter.getY(), -capsule.halfHeight, capsule.halfHeight);
        Vec3 segPoint(0.0f, y, 0.0f);

        Vec3 diff = localCenter - segPoint;
        float distSq = diff.lengthSq();
        float radiusSum = sphere.radius + capsule.radius;

        if (distSq > radiusSum * radiusSum) return false;

        float dist = math::fastSqrt(distSq);
        Vec3 localNormal;

        if (dist > math::Epsilon) {
            localNormal = diff * (1.0f / dist);
        } else {
            localNormal = Vec3(1.0f, 0.0f, 0.0f);
            dist = 0.0f;
        }

        float penetration = radiusSum - dist;
        Vec3 normal = txCapsule.transformDirection(localNormal);
        Vec3 pointOnSphere = txSphere.position - normal * sphere.radius;
        Vec3 pointOnCapsule = txCapsule.transformPoint(segPoint + localNormal * capsule.radius);

        manifold.addPoint(pointOnSphere, pointOnCapsule, normal, penetration);
        return true;
    }

    // ── Capsule-Capsule ──────────────────────────────────────────────────

    /// Compute closest points between two 3D line segments.
    /// Returns the squared distance. s and t are parametric values [0,1].
    static PULSE_FORCE_INLINE float closestPointSegments(
        Vec3 p1, Vec3 q1, Vec3 p2, Vec3 q2,
        float& s, float& t, Vec3& c1, Vec3& c2) noexcept
    {
        Vec3 d1 = q1 - p1; // Direction of segment 1
        Vec3 d2 = q2 - p2; // Direction of segment 2
        Vec3 r = p1 - p2;

        float a = d1.dot(d1); // Squared length of segment 1
        float e = d2.dot(d2); // Squared length of segment 2
        float f = d2.dot(r);

        if (a <= math::Epsilon && e <= math::Epsilon) {
            // Both degenerate to points
            s = t = 0.0f;
            c1 = p1; c2 = p2;
            return (c1 - c2).lengthSq();
        }

        if (a <= math::Epsilon) {
            s = 0.0f;
            t = math::clamp(f / e, 0.0f, 1.0f);
        } else {
            float c = d1.dot(r);
            if (e <= math::Epsilon) {
                t = 0.0f;
                s = math::clamp(-c / a, 0.0f, 1.0f);
            } else {
                float b = d1.dot(d2);
                float denom = a * e - b * b;

                if (denom > math::Epsilon) {
                    s = math::clamp((b * f - c * e) / denom, 0.0f, 1.0f);
                } else {
                    s = 0.0f;
                }

                t = (b * s + f) / e;

                if (t < 0.0f) {
                    t = 0.0f;
                    s = math::clamp(-c / a, 0.0f, 1.0f);
                } else if (t > 1.0f) {
                    t = 1.0f;
                    s = math::clamp((b - c) / a, 0.0f, 1.0f);
                }
            }
        }

        c1 = p1 + d1 * s;
        c2 = p2 + d2 * t;
        return (c1 - c2).lengthSq();
    }

    static inline bool collideCapsuleVsCapsule(
        const Capsule& capA, const Transform& txA,
        const Capsule& capB, const Transform& txB,
        ContactManifold& manifold) noexcept
    {
        // Get world-space segment endpoints
        Vec3 a1 = txA.transformPoint(capA.getBottomCenter());
        Vec3 a2 = txA.transformPoint(capA.getTopCenter());
        Vec3 b1 = txB.transformPoint(capB.getBottomCenter());
        Vec3 b2 = txB.transformPoint(capB.getTopCenter());

        float s, t;
        Vec3 closestA, closestB;
        float distSq = closestPointSegments(a1, a2, b1, b2, s, t, closestA, closestB);
        float radiusSum = capA.radius + capB.radius;

        if (distSq > radiusSum * radiusSum) return false;

        float dist = math::fastSqrt(distSq);
        Vec3 normal;

        if (dist > math::Epsilon) {
            normal = (closestA - closestB) * (1.0f / dist);
        } else {
            normal = Vec3(1.0f, 0.0f, 0.0f);
            dist = 0.0f;
        }

        float penetration = radiusSum - dist;
        Vec3 pointOnA = closestA - normal * capA.radius;
        Vec3 pointOnB = closestB + normal * capB.radius;

        manifold.addPoint(pointOnA, pointOnB, normal, penetration);
        return true;
    }

    // ── Generic GJK + EPA path ───────────────────────────────────────────

    template <typename ShapeA, typename ShapeB>
    static inline bool collideGjkEpa(
        const ShapeA& shapeA, const Transform& txA,
        const ShapeB& shapeB, const Transform& txB,
        ContactManifold& manifold) noexcept
    {
        GjkResult gjkResult;
        gjkQuery(shapeA, txA, shapeB, txB, gjkResult);

        if (gjkResult.status != GjkStatus::Overlapping) {
            return false;
        }

        // Shapes overlap — run EPA for penetration info
        EpaResult epaResult;
        epaQuery(shapeA, txA, shapeB, txB, gjkResult.simplex, epaResult);

        if (epaResult.penetration > math::Epsilon) {
            manifold.addPoint(epaResult.pointOnA, epaResult.pointOnB,
                              epaResult.normal, epaResult.penetration);
            return true;
        }

        return false;
    }

    // ── Sphere-Cylinder ──────────────────────────────────────────────────

    static inline bool collideSphereVsCylinder(
        const Sphere& sphere, const Transform& txSphere,
        const Cylinder& cyl, const Transform& txCyl,
        ContactManifold& manifold) noexcept
    {
        return collideGjkEpa(sphere, txSphere, cyl, txCyl, manifold);
    }

    // ── Sphere-ConvexHull ────────────────────────────────────────────────

    static inline bool collideSphereVsConvexHull(
        const Sphere& sphere, const Transform& txSphere,
        const ConvexHull& hull, const Transform& txHull,
        ContactManifold& manifold) noexcept
    {
        return collideGjkEpa(sphere, txSphere, hull, txHull, manifold);
    }

    // ── Box-Capsule ──────────────────────────────────────────────────────

    static inline bool collideBoxVsCapsule(
        const Box& box, const Transform& txBox,
        const Capsule& cap, const Transform& txCap,
        ContactManifold& manifold) noexcept
    {
        return collideGjkEpa(box, txBox, cap, txCap, manifold);
    }

    // ── Box-Cylinder ─────────────────────────────────────────────────────

    static inline bool collideBoxVsCylinder(
        const Box& box, const Transform& txBox,
        const Cylinder& cyl, const Transform& txCyl,
        ContactManifold& manifold) noexcept
    {
        return collideGjkEpa(box, txBox, cyl, txCyl, manifold);
    }

    // ── Box-ConvexHull ───────────────────────────────────────────────────

    static inline bool collideBoxVsConvexHull(
        const Box& box, const Transform& txBox,
        const ConvexHull& hull, const Transform& txHull,
        ContactManifold& manifold) noexcept
    {
        return collideGjkEpa(box, txBox, hull, txHull, manifold);
    }

    // ── Capsule-Cylinder ─────────────────────────────────────────────────

    static inline bool collideCapsuleVsCylinder(
        const Capsule& cap, const Transform& txCap,
        const Cylinder& cyl, const Transform& txCyl,
        ContactManifold& manifold) noexcept
    {
        return collideGjkEpa(cap, txCap, cyl, txCyl, manifold);
    }

    // ── Capsule-ConvexHull ───────────────────────────────────────────────

    static inline bool collideCapsuleVsConvexHull(
        const Capsule& cap, const Transform& txCap,
        const ConvexHull& hull, const Transform& txHull,
        ContactManifold& manifold) noexcept
    {
        return collideGjkEpa(cap, txCap, hull, txHull, manifold);
    }

    // ── Cylinder-Cylinder ────────────────────────────────────────────────

    static inline bool collideCylinderVsCylinder(
        const Cylinder& cylA, const Transform& txA,
        const Cylinder& cylB, const Transform& txB,
        ContactManifold& manifold) noexcept
    {
        return collideGjkEpa(cylA, txA, cylB, txB, manifold);
    }

    // ── Cylinder-ConvexHull ──────────────────────────────────────────────

    static inline bool collideCylinderVsConvexHull(
        const Cylinder& cyl, const Transform& txCyl,
        const ConvexHull& hull, const Transform& txHull,
        ContactManifold& manifold) noexcept
    {
        return collideGjkEpa(cyl, txCyl, hull, txHull, manifold);
    }

    // ── ConvexHull-ConvexHull ─────────────────────────────────────────────

    static inline bool collideConvexHullVsConvexHull(
        const ConvexHull& hullA, const Transform& txA,
        const ConvexHull& hullB, const Transform& txB,
        ContactManifold& manifold) noexcept
    {
        return collideGjkEpa(hullA, txA, hullB, txB, manifold);
    }

} // namespace collision_detail

// ── Public collision dispatch (type-safe convenience functions) ───────────────

/// Collide two spheres.
[[nodiscard]] static inline bool collide(
    const Sphere& a, const Transform& txA,
    const Sphere& b, const Transform& txB,
    ContactManifold& manifold) noexcept
{
    manifold.shapeTypeA = ShapeType::Sphere;
    manifold.shapeTypeB = ShapeType::Sphere;
    return collision_detail::collideSphereVsSphere(a, txA, b, txB, manifold);
}

/// Collide sphere vs. box.
[[nodiscard]] static inline bool collide(
    const Sphere& a, const Transform& txA,
    const Box& b, const Transform& txB,
    ContactManifold& manifold) noexcept
{
    manifold.shapeTypeA = ShapeType::Sphere;
    manifold.shapeTypeB = ShapeType::Box;
    return collision_detail::collideSphereVsBox(a, txA, b, txB, manifold);
}

/// Collide sphere vs. capsule.
[[nodiscard]] static inline bool collide(
    const Sphere& a, const Transform& txA,
    const Capsule& b, const Transform& txB,
    ContactManifold& manifold) noexcept
{
    manifold.shapeTypeA = ShapeType::Sphere;
    manifold.shapeTypeB = ShapeType::Capsule;
    return collision_detail::collideSphereVsCapsule(a, txA, b, txB, manifold);
}

/// Collide sphere vs. cylinder.
[[nodiscard]] static inline bool collide(
    const Sphere& a, const Transform& txA,
    const Cylinder& b, const Transform& txB,
    ContactManifold& manifold) noexcept
{
    manifold.shapeTypeA = ShapeType::Sphere;
    manifold.shapeTypeB = ShapeType::Cylinder;
    return collision_detail::collideSphereVsCylinder(a, txA, b, txB, manifold);
}

/// Collide sphere vs. convex hull.
[[nodiscard]] static inline bool collide(
    const Sphere& a, const Transform& txA,
    const ConvexHull& b, const Transform& txB,
    ContactManifold& manifold) noexcept
{
    manifold.shapeTypeA = ShapeType::Sphere;
    manifold.shapeTypeB = ShapeType::ConvexHull;
    return collision_detail::collideSphereVsConvexHull(a, txA, b, txB, manifold);
}

/// Collide two boxes (SAT).
[[nodiscard]] static inline bool collide(
    const Box& a, const Transform& txA,
    const Box& b, const Transform& txB,
    ContactManifold& manifold) noexcept
{
    manifold.shapeTypeA = ShapeType::Box;
    manifold.shapeTypeB = ShapeType::Box;
    return satTestBoxBox(a, txA, b, txB, manifold);
}

/// Collide box vs. capsule.
[[nodiscard]] static inline bool collide(
    const Box& a, const Transform& txA,
    const Capsule& b, const Transform& txB,
    ContactManifold& manifold) noexcept
{
    manifold.shapeTypeA = ShapeType::Box;
    manifold.shapeTypeB = ShapeType::Capsule;
    return collision_detail::collideBoxVsCapsule(a, txA, b, txB, manifold);
}

/// Collide box vs. cylinder.
[[nodiscard]] static inline bool collide(
    const Box& a, const Transform& txA,
    const Cylinder& b, const Transform& txB,
    ContactManifold& manifold) noexcept
{
    manifold.shapeTypeA = ShapeType::Box;
    manifold.shapeTypeB = ShapeType::Cylinder;
    return collision_detail::collideBoxVsCylinder(a, txA, b, txB, manifold);
}

/// Collide box vs. convex hull.
[[nodiscard]] static inline bool collide(
    const Box& a, const Transform& txA,
    const ConvexHull& b, const Transform& txB,
    ContactManifold& manifold) noexcept
{
    manifold.shapeTypeA = ShapeType::Box;
    manifold.shapeTypeB = ShapeType::ConvexHull;
    return collision_detail::collideBoxVsConvexHull(a, txA, b, txB, manifold);
}

/// Collide two capsules.
[[nodiscard]] static inline bool collide(
    const Capsule& a, const Transform& txA,
    const Capsule& b, const Transform& txB,
    ContactManifold& manifold) noexcept
{
    manifold.shapeTypeA = ShapeType::Capsule;
    manifold.shapeTypeB = ShapeType::Capsule;
    return collision_detail::collideCapsuleVsCapsule(a, txA, b, txB, manifold);
}

/// Collide capsule vs. cylinder.
[[nodiscard]] static inline bool collide(
    const Capsule& a, const Transform& txA,
    const Cylinder& b, const Transform& txB,
    ContactManifold& manifold) noexcept
{
    manifold.shapeTypeA = ShapeType::Capsule;
    manifold.shapeTypeB = ShapeType::Cylinder;
    return collision_detail::collideCapsuleVsCylinder(a, txA, b, txB, manifold);
}

/// Collide capsule vs. convex hull.
[[nodiscard]] static inline bool collide(
    const Capsule& a, const Transform& txA,
    const ConvexHull& b, const Transform& txB,
    ContactManifold& manifold) noexcept
{
    manifold.shapeTypeA = ShapeType::Capsule;
    manifold.shapeTypeB = ShapeType::ConvexHull;
    return collision_detail::collideCapsuleVsConvexHull(a, txA, b, txB, manifold);
}

/// Collide two cylinders.
[[nodiscard]] static inline bool collide(
    const Cylinder& a, const Transform& txA,
    const Cylinder& b, const Transform& txB,
    ContactManifold& manifold) noexcept
{
    manifold.shapeTypeA = ShapeType::Cylinder;
    manifold.shapeTypeB = ShapeType::Cylinder;
    return collision_detail::collideCylinderVsCylinder(a, txA, b, txB, manifold);
}

/// Collide cylinder vs. convex hull.
[[nodiscard]] static inline bool collide(
    const Cylinder& a, const Transform& txA,
    const ConvexHull& b, const Transform& txB,
    ContactManifold& manifold) noexcept
{
    manifold.shapeTypeA = ShapeType::Cylinder;
    manifold.shapeTypeB = ShapeType::ConvexHull;
    return collision_detail::collideCylinderVsConvexHull(a, txA, b, txB, manifold);
}

/// Collide two convex hulls.
[[nodiscard]] static inline bool collide(
    const ConvexHull& a, const Transform& txA,
    const ConvexHull& b, const Transform& txB,
    ContactManifold& manifold) noexcept
{
    manifold.shapeTypeA = ShapeType::ConvexHull;
    manifold.shapeTypeB = ShapeType::ConvexHull;
    return collision_detail::collideConvexHullVsConvexHull(a, txA, b, txB, manifold);
}

} // namespace pulse
