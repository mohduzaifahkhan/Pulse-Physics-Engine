/**
 * @file test_shapes.cpp
 * @brief Comprehensive unit tests for the Pulse collision shapes module.
 *
 * Tests all 6 shape types: Sphere, Box, Capsule, Cylinder, ConvexHull, TriMesh.
 * Each shape is tested for construction, AABB, mass properties, support function,
 * point containment, closest point, and ray intersection.
 */

#include <pulse/shapes/shape_common.h>
#include <pulse/shapes/sphere.h>
#include <pulse/shapes/box.h>
#include <pulse/shapes/capsule.h>
#include <pulse/shapes/cylinder.h>
#include <pulse/shapes/convex_hull.h>
#include <pulse/shapes/tri_mesh.h>

#include <pulse/math/math_common.h>
#include <pulse/math/transform.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

// ── Minimal test framework ────────────────────────────────────────────────────

static int g_totalTests = 0;
static int g_passedTests = 0;
static int g_failedTests = 0;

#define TEST_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            std::printf("  FAIL: %s (line %d): %s\n", __func__, __LINE__, #expr); \
            return false; \
        } \
    } while(0)

#define RUN_TEST(func) \
    do { \
        g_totalTests++; \
        if (func()) { \
            g_passedTests++; \
        } else { \
            g_failedTests++; \
            std::printf("FAILED: %s\n", #func); \
        } \
    } while(0)

using namespace pulse;

// ── Sphere Tests ──────────────────────────────────────────────────────────────

bool test_sphere_construction() {
    Sphere s;
    TEST_ASSERT(math::approxEqual(s.radius, 1.0f));
    TEST_ASSERT(s.Type == ShapeType::Sphere);

    Sphere s2(2.5f);
    TEST_ASSERT(math::approxEqual(s2.radius, 2.5f));
    return true;
}

bool test_sphere_aabb() {
    Sphere s(2.0f);
    Transform tx(Vec3(1.0f, 2.0f, 3.0f));

    AABB aabb = s.computeAABB(tx);
    TEST_ASSERT(math::approxEqual(aabb.min.getX(), -1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(aabb.min.getY(),  0.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(aabb.min.getZ(),  1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(aabb.max.getX(),  3.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(aabb.max.getY(),  4.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(aabb.max.getZ(),  5.0f, 0.01f));

    AABB localAABB = s.computeLocalAABB();
    TEST_ASSERT(math::approxEqual(localAABB.min.getX(), -2.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(localAABB.max.getX(),  2.0f, 0.01f));
    return true;
}

bool test_sphere_mass() {
    Sphere s(1.0f);
    MassProperties mp = s.computeMass(1.0f);

    float expectedVolume = (4.0f / 3.0f) * math::Pi;
    TEST_ASSERT(math::approxEqual(mp.mass, expectedVolume, 0.01f));

    // Inertia should be (2/5)mr² = (2/5) * volume * 1
    float expectedI = (2.0f / 5.0f) * expectedVolume;
    TEST_ASSERT(math::approxEqual(mp.inertiaTensor[0].getX(), expectedI, 0.01f));
    TEST_ASSERT(math::approxEqual(mp.inertiaTensor[1].getY(), expectedI, 0.01f));
    TEST_ASSERT(math::approxEqual(mp.inertiaTensor[2].getZ(), expectedI, 0.01f));

    // CoM at origin
    TEST_ASSERT(math::approxEqual(mp.centerOfMass.getX(), 0.0f));
    return true;
}

bool test_sphere_support() {
    Sphere s(3.0f);

    Vec3 sup = s.support(Vec3::unitX());
    TEST_ASSERT(math::approxEqual(sup.getX(), 3.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(sup.getY(), 0.0f, 0.01f));

    Vec3 sup2 = s.support(Vec3(0.0f, -1.0f, 0.0f));
    TEST_ASSERT(math::approxEqual(sup2.getY(), -3.0f, 0.01f));

    Vec3 sup3 = s.support(Vec3(1.0f, 1.0f, 0.0f));
    float len = sup3.length();
    TEST_ASSERT(math::approxEqual(len, 3.0f, 0.01f));
    return true;
}

bool test_sphere_contains_point() {
    Sphere s(2.0f);

    TEST_ASSERT(s.containsPoint(Vec3::zero()));
    TEST_ASSERT(s.containsPoint(Vec3(1.0f, 0.0f, 0.0f)));
    TEST_ASSERT(s.containsPoint(Vec3(1.9f, 0.0f, 0.0f)));
    TEST_ASSERT(!s.containsPoint(Vec3(2.1f, 0.0f, 0.0f)));
    TEST_ASSERT(!s.containsPoint(Vec3(3.0f, 3.0f, 3.0f)));
    return true;
}

bool test_sphere_closest_point() {
    Sphere s(2.0f);

    Vec3 cp = s.closestPoint(Vec3(4.0f, 0.0f, 0.0f));
    TEST_ASSERT(math::approxEqual(cp.getX(), 2.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(cp.getY(), 0.0f, 0.01f));

    Vec3 cp2 = s.closestPoint(Vec3(0.0f, -5.0f, 0.0f));
    TEST_ASSERT(math::approxEqual(cp2.getY(), -2.0f, 0.01f));
    return true;
}

bool test_sphere_ray_intersection() {
    Sphere s(1.0f);

    // Ray from (0, 0, -5) toward +Z should hit at t=4
    ShapeRayResult r = s.rayIntersect(Vec3(0.0f, 0.0f, -5.0f), Vec3(0.0f, 0.0f, 1.0f));
    TEST_ASSERT(r.hit);
    TEST_ASSERT(math::approxEqual(r.t, 4.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(r.normal.getZ(), -1.0f, 0.01f));

    // Ray missing the sphere
    ShapeRayResult r2 = s.rayIntersect(Vec3(0.0f, 5.0f, -5.0f), Vec3(0.0f, 0.0f, 1.0f));
    TEST_ASSERT(!r2.hit);

    // Ray from inside
    ShapeRayResult r3 = s.rayIntersect(Vec3::zero(), Vec3(1.0f, 0.0f, 0.0f));
    TEST_ASSERT(r3.hit);
    TEST_ASSERT(math::approxEqual(r3.t, 1.0f, 0.01f));
    return true;
}

// ── Box Tests ─────────────────────────────────────────────────────────────────

bool test_box_construction() {
    Box b;
    TEST_ASSERT(math::approxEqual(b.halfExtents.getX(), 0.5f));
    TEST_ASSERT(b.Type == ShapeType::Box);

    Box b2(1.0f, 2.0f, 3.0f);
    TEST_ASSERT(math::approxEqual(b2.halfExtents.getX(), 1.0f));
    TEST_ASSERT(math::approxEqual(b2.halfExtents.getY(), 2.0f));
    TEST_ASSERT(math::approxEqual(b2.halfExtents.getZ(), 3.0f));
    return true;
}

bool test_box_aabb_identity() {
    Box b(1.0f, 2.0f, 3.0f);
    Transform tx(Vec3(10.0f, 0.0f, 0.0f));

    AABB aabb = b.computeAABB(tx);
    TEST_ASSERT(math::approxEqual(aabb.min.getX(), 9.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(aabb.max.getX(), 11.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(aabb.min.getY(), -2.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(aabb.max.getY(), 2.0f, 0.01f));
    return true;
}

bool test_box_aabb_rotated() {
    Box b(1.0f, 0.0f, 0.0f); // Thin box along X
    // Rotate 90 degrees around Z → box now along Y
    Quat rot = Quat::fromAxisAngle(Vec3::unitZ(), math::HalfPi);
    Transform tx(Vec3::zero(), rot);

    AABB aabb = b.computeAABB(tx);
    // After rotation, the X extent should be ~0 and Y extent should be ~1
    TEST_ASSERT(aabb.max.getX() < 0.1f);
    TEST_ASSERT(aabb.max.getY() > 0.9f);
    return true;
}

bool test_box_mass() {
    Box b(1.0f, 1.0f, 1.0f); // 2×2×2 cube
    MassProperties mp = b.computeMass(1.0f);

    float expectedMass = 8.0f; // Volume = 8
    TEST_ASSERT(math::approxEqual(mp.mass, expectedMass, 0.01f));

    // For a cube: I = m/12 * (h² + h²) = 8/12 * (4+4) = 16/3
    float expectedI = 8.0f / 12.0f * (4.0f + 4.0f);
    TEST_ASSERT(math::approxEqual(mp.inertiaTensor[0].getX(), expectedI, 0.1f));
    return true;
}

bool test_box_support() {
    Box b(2.0f, 3.0f, 4.0f);

    Vec3 sup = b.support(Vec3(1.0f, 0.0f, 0.0f));
    TEST_ASSERT(math::approxEqual(sup.getX(), 2.0f));

    Vec3 sup2 = b.support(Vec3(-1.0f, -1.0f, -1.0f));
    TEST_ASSERT(math::approxEqual(sup2.getX(), -2.0f));
    TEST_ASSERT(math::approxEqual(sup2.getY(), -3.0f));
    TEST_ASSERT(math::approxEqual(sup2.getZ(), -4.0f));
    return true;
}

bool test_box_contains_point() {
    Box b(1.0f, 2.0f, 3.0f);

    TEST_ASSERT(b.containsPoint(Vec3::zero()));
    TEST_ASSERT(b.containsPoint(Vec3(0.5f, 1.0f, 2.0f)));
    TEST_ASSERT(!b.containsPoint(Vec3(1.5f, 0.0f, 0.0f)));
    TEST_ASSERT(!b.containsPoint(Vec3(0.0f, 2.5f, 0.0f)));
    return true;
}

bool test_box_closest_point() {
    Box b(1.0f, 1.0f, 1.0f);

    Vec3 cp = b.closestPoint(Vec3(5.0f, 0.0f, 0.0f));
    TEST_ASSERT(math::approxEqual(cp.getX(), 1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(cp.getY(), 0.0f, 0.01f));

    // Point inside should clamp to itself
    Vec3 cp2 = b.closestPoint(Vec3(0.5f, 0.5f, 0.5f));
    TEST_ASSERT(math::approxEqual(cp2.getX(), 0.5f, 0.01f));
    return true;
}

bool test_box_ray_intersection() {
    Box b(1.0f, 1.0f, 1.0f);

    // Ray from -5 on X axis toward +X
    ShapeRayResult r = b.rayIntersect(Vec3(-5.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f));
    TEST_ASSERT(r.hit);
    TEST_ASSERT(math::approxEqual(r.t, 4.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(r.normal.getX(), -1.0f, 0.01f));

    // Ray missing
    ShapeRayResult r2 = b.rayIntersect(Vec3(-5.0f, 5.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f));
    TEST_ASSERT(!r2.hit);
    return true;
}

bool test_box_vertices() {
    Box b(1.0f, 2.0f, 3.0f);
    Vec3 verts[8];
    b.getVertices(verts);

    // Check that all 8 vertices have correct absolute values
    for (int i = 0; i < 8; ++i) {
        TEST_ASSERT(math::approxEqual(math::fastAbs(verts[i].getX()), 1.0f));
        TEST_ASSERT(math::approxEqual(math::fastAbs(verts[i].getY()), 2.0f));
        TEST_ASSERT(math::approxEqual(math::fastAbs(verts[i].getZ()), 3.0f));
    }
    return true;
}

// ── Capsule Tests ─────────────────────────────────────────────────────────────

bool test_capsule_construction() {
    Capsule c;
    TEST_ASSERT(math::approxEqual(c.radius, 0.5f));
    TEST_ASSERT(math::approxEqual(c.halfHeight, 0.5f));
    TEST_ASSERT(c.Type == ShapeType::Capsule);

    Capsule c2(1.0f, 2.0f);
    TEST_ASSERT(math::approxEqual(c2.totalHeight(), 6.0f, 0.01f)); // 2*(2+1)
    return true;
}

bool test_capsule_aabb() {
    Capsule c(1.0f, 2.0f);
    Transform tx;

    AABB aabb = c.computeAABB(tx);
    // Total height = 2*(2+1) = 6, width = 2*radius = 2
    TEST_ASSERT(math::approxEqual(aabb.min.getY(), -3.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(aabb.max.getY(),  3.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(aabb.min.getX(), -1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(aabb.max.getX(),  1.0f, 0.01f));
    return true;
}

bool test_capsule_mass() {
    Capsule c(1.0f, 1.0f);
    MassProperties mp = c.computeMass(1.0f);

    // Volume = cylinder(πr²*2h) + sphere(4/3πr³)
    float cylVol = math::Pi * 1.0f * 2.0f; // π*1*2
    float sphVol = (4.0f / 3.0f) * math::Pi;
    float expectedMass = cylVol + sphVol;
    TEST_ASSERT(math::approxEqual(mp.mass, expectedMass, 0.1f));
    TEST_ASSERT(mp.mass > 0.0f);
    return true;
}

bool test_capsule_support() {
    Capsule c(1.0f, 2.0f);

    // Upward direction: should be top center + radius upward
    Vec3 sup = c.support(Vec3(0.0f, 1.0f, 0.0f));
    TEST_ASSERT(math::approxEqual(sup.getY(), 3.0f, 0.01f)); // halfHeight + radius

    // Downward: bottom center - radius
    Vec3 sup2 = c.support(Vec3(0.0f, -1.0f, 0.0f));
    TEST_ASSERT(math::approxEqual(sup2.getY(), -3.0f, 0.01f));

    // Sideways: from one endpoint + radius in that direction
    Vec3 sup3 = c.support(Vec3(1.0f, 0.0f, 0.0f));
    TEST_ASSERT(math::approxEqual(sup3.getX(), 1.0f, 0.01f));
    return true;
}

bool test_capsule_contains_point() {
    Capsule c(1.0f, 2.0f);

    TEST_ASSERT(c.containsPoint(Vec3::zero()));
    TEST_ASSERT(c.containsPoint(Vec3(0.0f, 2.5f, 0.0f))); // Inside top hemisphere
    TEST_ASSERT(c.containsPoint(Vec3(0.5f, 0.0f, 0.0f))); // Inside cylinder
    TEST_ASSERT(!c.containsPoint(Vec3(1.5f, 0.0f, 0.0f))); // Outside
    TEST_ASSERT(!c.containsPoint(Vec3(0.0f, 3.5f, 0.0f))); // Above top hemisphere
    return true;
}

bool test_capsule_closest_point() {
    Capsule c(1.0f, 2.0f);

    // Point far on X axis → closest on cylinder surface
    Vec3 cp = c.closestPoint(Vec3(5.0f, 0.0f, 0.0f));
    TEST_ASSERT(math::approxEqual(cp.getX(), 1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(cp.getY(), 0.0f, 0.01f));
    return true;
}

// ── Cylinder Tests ────────────────────────────────────────────────────────────

bool test_cylinder_construction() {
    Cylinder c;
    TEST_ASSERT(math::approxEqual(c.radius, 0.5f));
    TEST_ASSERT(math::approxEqual(c.halfHeight, 0.5f));
    TEST_ASSERT(c.Type == ShapeType::Cylinder);

    Cylinder c2(2.0f, 3.0f);
    TEST_ASSERT(math::approxEqual(c2.radius, 2.0f));
    TEST_ASSERT(math::approxEqual(c2.halfHeight, 3.0f));
    return true;
}

bool test_cylinder_aabb() {
    Cylinder c(1.0f, 2.0f);
    Transform tx;

    AABB aabb = c.computeAABB(tx);
    TEST_ASSERT(math::approxEqual(aabb.min.getX(), -1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(aabb.max.getX(),  1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(aabb.min.getY(), -2.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(aabb.max.getY(),  2.0f, 0.01f));
    return true;
}

bool test_cylinder_mass() {
    Cylinder c(1.0f, 1.0f);
    MassProperties mp = c.computeMass(1.0f);

    float expectedMass = math::Pi * 1.0f * 2.0f; // πr²h = π*1*2
    TEST_ASSERT(math::approxEqual(mp.mass, expectedMass, 0.1f));

    // Iyy (symmetry axis) = (1/2)mr²
    float expectedIyy = 0.5f * expectedMass * 1.0f;
    TEST_ASSERT(math::approxEqual(mp.inertiaTensor[1].getY(), expectedIyy, 0.1f));
    return true;
}

bool test_cylinder_support() {
    Cylinder c(2.0f, 3.0f);

    // Straight up
    Vec3 sup = c.support(Vec3(0.0f, 1.0f, 0.0f));
    TEST_ASSERT(math::approxEqual(sup.getY(), 3.0f, 0.01f));

    // Straight right
    Vec3 sup2 = c.support(Vec3(1.0f, 0.0f, 0.0f));
    TEST_ASSERT(math::approxEqual(sup2.getX(), 2.0f, 0.01f));

    // Diagonal
    Vec3 sup3 = c.support(Vec3(1.0f, 1.0f, 0.0f));
    TEST_ASSERT(math::approxEqual(sup3.getY(), 3.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(sup3.getX(), 2.0f, 0.01f));
    return true;
}

bool test_cylinder_contains_point() {
    Cylinder c(1.0f, 2.0f);

    TEST_ASSERT(c.containsPoint(Vec3::zero()));
    TEST_ASSERT(c.containsPoint(Vec3(0.5f, 1.0f, 0.0f)));
    TEST_ASSERT(!c.containsPoint(Vec3(1.5f, 0.0f, 0.0f))); // Outside radius
    TEST_ASSERT(!c.containsPoint(Vec3(0.0f, 2.5f, 0.0f))); // Above cap
    return true;
}

bool test_cylinder_ray_intersection() {
    Cylinder c(1.0f, 2.0f);

    // Ray from -5 on X axis toward +X, hitting the cylinder body
    ShapeRayResult r = c.rayIntersect(Vec3(-5.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f));
    TEST_ASSERT(r.hit);
    TEST_ASSERT(math::approxEqual(r.t, 4.0f, 0.01f));

    // Ray from above, hitting the top cap
    ShapeRayResult r2 = c.rayIntersect(Vec3(0.0f, 5.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f));
    TEST_ASSERT(r2.hit);
    TEST_ASSERT(math::approxEqual(r2.t, 3.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(r2.normal.getY(), 1.0f, 0.01f));

    // Ray missing entirely
    ShapeRayResult r3 = c.rayIntersect(Vec3(-5.0f, 5.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f));
    TEST_ASSERT(!r3.hit);
    return true;
}

// ── ConvexHull Tests ──────────────────────────────────────────────────────────

// Helper: create a unit cube convex hull
static Vec3 g_cubeVerts[8] = {
    Vec3(-1.0f, -1.0f, -1.0f),
    Vec3( 1.0f, -1.0f, -1.0f),
    Vec3(-1.0f,  1.0f, -1.0f),
    Vec3( 1.0f,  1.0f, -1.0f),
    Vec3(-1.0f, -1.0f,  1.0f),
    Vec3( 1.0f, -1.0f,  1.0f),
    Vec3(-1.0f,  1.0f,  1.0f),
    Vec3( 1.0f,  1.0f,  1.0f),
};

bool test_convex_hull_construction() {
    ConvexHull hull(g_cubeVerts, 8);
    TEST_ASSERT(hull.vertexCount == 8);
    TEST_ASSERT(hull.vertices == g_cubeVerts);
    TEST_ASSERT(hull.Type == ShapeType::ConvexHull);
    return true;
}

bool test_convex_hull_aabb() {
    ConvexHull hull(g_cubeVerts, 8);
    Transform tx;

    AABB aabb = hull.computeAABB(tx);
    TEST_ASSERT(math::approxEqual(aabb.min.getX(), -1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(aabb.max.getX(),  1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(aabb.min.getY(), -1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(aabb.max.getY(),  1.0f, 0.01f));

    AABB localAABB = hull.computeLocalAABB();
    TEST_ASSERT(math::approxEqual(localAABB.min.getZ(), -1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(localAABB.max.getZ(),  1.0f, 0.01f));
    return true;
}

bool test_convex_hull_support() {
    ConvexHull hull(g_cubeVerts, 8);

    Vec3 sup = hull.support(Vec3(1.0f, 0.0f, 0.0f));
    TEST_ASSERT(math::approxEqual(sup.getX(), 1.0f, 0.01f));

    Vec3 sup2 = hull.support(Vec3(1.0f, 1.0f, 1.0f));
    TEST_ASSERT(math::approxEqual(sup2.getX(), 1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(sup2.getY(), 1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(sup2.getZ(), 1.0f, 0.01f));
    return true;
}

bool test_convex_hull_contains_point() {
    ConvexHull hull(g_cubeVerts, 8);

    TEST_ASSERT(hull.containsPoint(Vec3::zero()));
    TEST_ASSERT(hull.containsPoint(Vec3(0.5f, 0.5f, 0.5f)));
    TEST_ASSERT(!hull.containsPoint(Vec3(2.0f, 0.0f, 0.0f)));
    return true;
}

bool test_convex_hull_mass() {
    ConvexHull hull(g_cubeVerts, 8);
    MassProperties mp = hull.computeMass(1.0f);

    TEST_ASSERT(mp.mass > 0.0f);
    // Volume of 2x2x2 cube = 8
    TEST_ASSERT(math::approxEqual(mp.mass, 8.0f, 0.1f));
    return true;
}

bool test_convex_hull_closest_vertex() {
    ConvexHull hull(g_cubeVerts, 8);

    Vec3 cv = hull.closestVertex(Vec3(5.0f, 5.0f, 5.0f));
    TEST_ASSERT(math::approxEqual(cv.getX(), 1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(cv.getY(), 1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(cv.getZ(), 1.0f, 0.01f));
    return true;
}

bool test_small_convex_hull() {
    SmallConvexHull<8> small;
    TEST_ASSERT(small.count == 0);

    for (int i = 0; i < 8; ++i) {
        TEST_ASSERT(small.addVertex(g_cubeVerts[i]));
    }
    TEST_ASSERT(small.count == 8);
    TEST_ASSERT(!small.addVertex(Vec3::zero())); // Should be full

    ConvexHull hull = small.toHull();
    TEST_ASSERT(hull.vertexCount == 8);

    Vec3 sup = hull.support(Vec3::unitX());
    TEST_ASSERT(math::approxEqual(sup.getX(), 1.0f, 0.01f));
    return true;
}

// ── TriMesh Tests ─────────────────────────────────────────────────────────────

// Helper: simple quad (two triangles)
static Vec3 g_quadVerts[4] = {
    Vec3(-1.0f, 0.0f, -1.0f), // 0
    Vec3( 1.0f, 0.0f, -1.0f), // 1
    Vec3( 1.0f, 0.0f,  1.0f), // 2
    Vec3(-1.0f, 0.0f,  1.0f), // 3
};
static uint32_t g_quadIndices[6] = { 0, 1, 2,  0, 2, 3 };

bool test_tri_mesh_construction() {
    TriMesh mesh(g_quadVerts, 4, g_quadIndices, 2);
    TEST_ASSERT(mesh.vertexCount == 4);
    TEST_ASSERT(mesh.triangleCount == 2);
    TEST_ASSERT(mesh.Type == ShapeType::TriMesh);
    return true;
}

bool test_tri_mesh_aabb() {
    TriMesh mesh(g_quadVerts, 4, g_quadIndices, 2);

    AABB aabb = mesh.computeLocalAABB();
    TEST_ASSERT(math::approxEqual(aabb.min.getX(), -1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(aabb.max.getX(),  1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(aabb.min.getY(),  0.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(aabb.max.getY(),  0.0f, 0.01f));
    return true;
}

bool test_tri_mesh_ray_intersection() {
    TriMesh mesh(g_quadVerts, 4, g_quadIndices, 2);

    // Ray from above, straight down onto the quad
    ShapeRayResult r = mesh.rayIntersect(Vec3(0.0f, 5.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f));
    TEST_ASSERT(r.hit);
    TEST_ASSERT(math::approxEqual(r.t, 5.0f, 0.01f));

    // Ray from below pointing up
    ShapeRayResult r2 = mesh.rayIntersect(Vec3(0.0f, -3.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
    TEST_ASSERT(r2.hit);
    TEST_ASSERT(math::approxEqual(r2.t, 3.0f, 0.01f));

    // Ray missing the quad entirely
    ShapeRayResult r3 = mesh.rayIntersect(Vec3(5.0f, 5.0f, 5.0f), Vec3(0.0f, -1.0f, 0.0f));
    TEST_ASSERT(!r3.hit);
    return true;
}

bool test_tri_mesh_closest_point() {
    TriMesh mesh(g_quadVerts, 4, g_quadIndices, 2);

    Vec3 cp = mesh.closestPoint(Vec3(0.0f, 3.0f, 0.0f));
    TEST_ASSERT(math::approxEqual(cp.getY(), 0.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(cp.getX(), 0.0f, 0.1f));
    return true;
}

bool test_tri_mesh_contains_point() {
    TriMesh mesh(g_quadVerts, 4, g_quadIndices, 2);

    // TriMesh containsPoint always returns false (surface-only)
    TEST_ASSERT(!mesh.containsPoint(Vec3::zero()));
    return true;
}

bool test_tri_mesh_triangle_normal() {
    TriMesh mesh(g_quadVerts, 4, g_quadIndices, 2);

    Vec3 normal = mesh.getTriangleNormal(0).normalized();
    // The quad lies in the XZ plane, so normal should be along Y
    TEST_ASSERT(math::approxEqual(math::fastAbs(normal.getY()), 1.0f, 0.01f));
    return true;
}

// ── ShapeType / MassProperties Tests ──────────────────────────────────────────

bool test_shape_types() {
    TEST_ASSERT(static_cast<uint8_t>(ShapeType::Sphere) == 0);
    TEST_ASSERT(static_cast<uint8_t>(ShapeType::Box) == 1);
    TEST_ASSERT(static_cast<uint8_t>(ShapeType::Capsule) == 2);
    TEST_ASSERT(static_cast<uint8_t>(ShapeType::Cylinder) == 3);
    TEST_ASSERT(static_cast<uint8_t>(ShapeType::ConvexHull) == 4);
    TEST_ASSERT(static_cast<uint8_t>(ShapeType::TriMesh) == 5);
    TEST_ASSERT(static_cast<uint8_t>(ShapeType::Count) == 6);
    return true;
}

bool test_mass_properties_default() {
    MassProperties mp;
    TEST_ASSERT(math::approxEqual(mp.mass, 0.0f));
    TEST_ASSERT(mp.centerOfMass == Vec3::zero());
    return true;
}

bool test_shape_ray_result() {
    ShapeRayResult miss = ShapeRayResult::miss();
    TEST_ASSERT(!miss.hit);
    TEST_ASSERT(miss.t < 0.0f);

    ShapeRayResult hit(3.14f, Vec3::unitY());
    TEST_ASSERT(hit.hit);
    TEST_ASSERT(math::approxEqual(hit.t, 3.14f, 0.001f));
    return true;
}

// ── Cross-shape: world-space support ──────────────────────────────────────────

bool test_sphere_support_world() {
    Sphere s(2.0f);
    Transform tx(Vec3(10.0f, 0.0f, 0.0f));

    Vec3 sup = s.supportWorld(Vec3::unitX(), tx);
    TEST_ASSERT(math::approxEqual(sup.getX(), 12.0f, 0.01f));
    return true;
}

bool test_box_support_world() {
    Box b(1.0f, 1.0f, 1.0f);
    Transform tx(Vec3(5.0f, 0.0f, 0.0f));

    Vec3 sup = b.supportWorld(Vec3::unitX(), tx);
    TEST_ASSERT(math::approxEqual(sup.getX(), 6.0f, 0.01f));
    return true;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    std::printf("═══════════════════════════════════════════════════════\n");
    std::printf("  Pulse Physics Engine — Shapes Module Tests\n");
    std::printf("═══════════════════════════════════════════════════════\n\n");

    // Common types
    std::printf("── ShapeType / MassProperties ──\n");
    RUN_TEST(test_shape_types);
    RUN_TEST(test_mass_properties_default);
    RUN_TEST(test_shape_ray_result);

    // Sphere
    std::printf("\n── Sphere ──\n");
    RUN_TEST(test_sphere_construction);
    RUN_TEST(test_sphere_aabb);
    RUN_TEST(test_sphere_mass);
    RUN_TEST(test_sphere_support);
    RUN_TEST(test_sphere_contains_point);
    RUN_TEST(test_sphere_closest_point);
    RUN_TEST(test_sphere_ray_intersection);
    RUN_TEST(test_sphere_support_world);

    // Box
    std::printf("\n── Box ──\n");
    RUN_TEST(test_box_construction);
    RUN_TEST(test_box_aabb_identity);
    RUN_TEST(test_box_aabb_rotated);
    RUN_TEST(test_box_mass);
    RUN_TEST(test_box_support);
    RUN_TEST(test_box_contains_point);
    RUN_TEST(test_box_closest_point);
    RUN_TEST(test_box_ray_intersection);
    RUN_TEST(test_box_vertices);
    RUN_TEST(test_box_support_world);

    // Capsule
    std::printf("\n── Capsule ──\n");
    RUN_TEST(test_capsule_construction);
    RUN_TEST(test_capsule_aabb);
    RUN_TEST(test_capsule_mass);
    RUN_TEST(test_capsule_support);
    RUN_TEST(test_capsule_contains_point);
    RUN_TEST(test_capsule_closest_point);

    // Cylinder
    std::printf("\n── Cylinder ──\n");
    RUN_TEST(test_cylinder_construction);
    RUN_TEST(test_cylinder_aabb);
    RUN_TEST(test_cylinder_mass);
    RUN_TEST(test_cylinder_support);
    RUN_TEST(test_cylinder_contains_point);
    RUN_TEST(test_cylinder_ray_intersection);

    // ConvexHull
    std::printf("\n── ConvexHull ──\n");
    RUN_TEST(test_convex_hull_construction);
    RUN_TEST(test_convex_hull_aabb);
    RUN_TEST(test_convex_hull_support);
    RUN_TEST(test_convex_hull_contains_point);
    RUN_TEST(test_convex_hull_mass);
    RUN_TEST(test_convex_hull_closest_vertex);
    RUN_TEST(test_small_convex_hull);

    // TriMesh
    std::printf("\n── TriMesh ──\n");
    RUN_TEST(test_tri_mesh_construction);
    RUN_TEST(test_tri_mesh_aabb);
    RUN_TEST(test_tri_mesh_ray_intersection);
    RUN_TEST(test_tri_mesh_closest_point);
    RUN_TEST(test_tri_mesh_contains_point);
    RUN_TEST(test_tri_mesh_triangle_normal);

    // Summary
    std::printf("\n═══════════════════════════════════════════════════════\n");
    std::printf("  Results: %d/%d passed", g_passedTests, g_totalTests);
    if (g_failedTests > 0) {
        std::printf(" (%d FAILED)", g_failedTests);
    }
    std::printf("\n═══════════════════════════════════════════════════════\n");

    return g_failedTests > 0 ? 1 : 0;
}
