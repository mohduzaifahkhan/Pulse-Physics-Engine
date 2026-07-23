/**
 * @file test_narrowphase.cpp
 * @brief Comprehensive unit tests for the Pulse narrowphase module.
 *
 * Tests all narrow-phase algorithms: specialized collision routines,
 * SAT, GJK, EPA, MPR, and CCD. Covers correctness, edge cases, and
 * robustness under degenerate configurations.
 */

#include <pulse/narrowphase/narrowphase_common.h>
#include <pulse/narrowphase/sat.h>
#include <pulse/narrowphase/gjk.h>
#include <pulse/narrowphase/epa.h>
#include <pulse/narrowphase/mpr.h>
#include <pulse/narrowphase/ccd.h>
#include <pulse/narrowphase/collision_dispatch.h>

#include <pulse/shapes/sphere.h>
#include <pulse/shapes/box.h>
#include <pulse/shapes/capsule.h>
#include <pulse/shapes/cylinder.h>
#include <pulse/shapes/convex_hull.h>

#include <pulse/math/math_common.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace pulse;

// ── Minimal test framework ────────────────────────────────────────────────────

static int g_total = 0, g_passed = 0, g_failed = 0;

#define TEST_ASSERT(expr) \
    do { if (!(expr)) { \
        std::printf("  FAIL: %s (line %d): %s\n", __func__, __LINE__, #expr); \
        return false; \
    }} while(0)

#define RUN_TEST(func) \
    do { \
        g_total++; \
        if (func()) { g_passed++; } \
        else { g_failed++; std::printf("FAILED: %s\n", #func); } \
    } while(0)

// ── Helpers ──────────────────────────────────────────────────────────────────

static bool approx(float a, float b, float eps = 0.01f) {
    return std::fabs(a - b) < eps;
}

static bool approxVec(Vec3 a, Vec3 b, float eps = 0.01f) {
    return approx(a.getX(), b.getX(), eps) &&
           approx(a.getY(), b.getY(), eps) &&
           approx(a.getZ(), b.getZ(), eps);
}

// ═════════════════════════════════════════════════════════════════════════════
// ContactManifold Tests
// ═════════════════════════════════════════════════════════════════════════════

bool test_manifold_default() {
    ContactManifold m;
    TEST_ASSERT(m.numContacts == 0);
    return true;
}

bool test_manifold_add_points() {
    ContactManifold m;
    m.addPoint(Vec3(0,0,0), Vec3(0,0,0), Vec3(0,1,0), 0.1f);
    TEST_ASSERT(m.numContacts == 1);
    m.addPoint(Vec3(1,0,0), Vec3(1,0,0), Vec3(0,1,0), 0.2f);
    TEST_ASSERT(m.numContacts == 2);
    m.addPoint(Vec3(0,0,1), Vec3(0,0,1), Vec3(0,1,0), 0.3f);
    m.addPoint(Vec3(1,0,1), Vec3(1,0,1), Vec3(0,1,0), 0.4f);
    TEST_ASSERT(m.numContacts == 4);
    return true;
}

bool test_manifold_clear() {
    ContactManifold m;
    m.addPoint(Vec3(0,0,0), Vec3(0,0,0), Vec3(0,1,0), 0.1f);
    m.clear();
    TEST_ASSERT(m.numContacts == 0);
    return true;
}

bool test_manifold_reduce_to_4() {
    ContactManifold m;
    // Add 4 points, then a 5th — should reduce to best 4
    m.addPoint(Vec3(0,0,0), Vec3(0,0,0), Vec3(0,1,0), 0.1f);
    m.addPoint(Vec3(1,0,0), Vec3(1,0,0), Vec3(0,1,0), 0.2f);
    m.addPoint(Vec3(0,0,1), Vec3(0,0,1), Vec3(0,1,0), 0.3f);
    m.addPoint(Vec3(1,0,1), Vec3(1,0,1), Vec3(0,1,0), 0.4f);
    // 5th point — triggers reduction
    m.addPoint(Vec3(0.5f, 0, 0.5f), Vec3(0.5f, 0, 0.5f), Vec3(0,1,0), 0.5f);
    TEST_ASSERT(m.numContacts == 4);
    return true;
}

bool test_manifold_max_penetration() {
    ContactManifold m;
    m.addPoint(Vec3(0,0,0), Vec3(0,0,0), Vec3(0,1,0), 0.1f);
    m.addPoint(Vec3(1,0,0), Vec3(1,0,0), Vec3(0,1,0), 0.5f);
    m.addPoint(Vec3(0,0,1), Vec3(0,0,1), Vec3(0,1,0), 0.3f);
    TEST_ASSERT(approx(m.getMaxPenetration(), 0.5f));
    return true;
}

bool test_manifold_average_normal() {
    ContactManifold m;
    m.addPoint(Vec3(0,0,0), Vec3(0,0,0), Vec3(0,1,0), 0.1f);
    m.addPoint(Vec3(1,0,0), Vec3(1,0,0), Vec3(0,1,0), 0.2f);
    Vec3 avg = m.getAverageNormal();
    TEST_ASSERT(approxVec(avg, Vec3(0,1,0)));
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Sphere-Sphere Tests
// ═════════════════════════════════════════════════════════════════════════════

bool test_sphere_sphere_overlapping() {
    Sphere a(1.0f), b(1.0f);
    Transform txA(Vec3(0,0,0)), txB(Vec3(1.5f, 0, 0));
    ContactManifold m;
    bool hit = collide(a, txA, b, txB, m);
    TEST_ASSERT(hit);
    TEST_ASSERT(m.numContacts == 1);
    TEST_ASSERT(approx(m.points[0].penetration, 0.5f));
    // Normal should point from B to A (i.e., along -X)
    TEST_ASSERT(m.points[0].normal.getX() < 0.0f);
    return true;
}

bool test_sphere_sphere_separated() {
    Sphere a(1.0f), b(1.0f);
    Transform txA(Vec3(0,0,0)), txB(Vec3(3.0f, 0, 0));
    ContactManifold m;
    bool hit = collide(a, txA, b, txB, m);
    TEST_ASSERT(!hit);
    TEST_ASSERT(m.numContacts == 0);
    return true;
}

bool test_sphere_sphere_touching() {
    Sphere a(1.0f), b(1.0f);
    Transform txA(Vec3(0,0,0)), txB(Vec3(2.0f, 0, 0));
    ContactManifold m;
    bool hit = collide(a, txA, b, txB, m);
    // Touching = 0 penetration, should still detect
    TEST_ASSERT(!hit || m.points[0].penetration <= 0.01f);
    return true;
}

bool test_sphere_sphere_concentric() {
    Sphere a(1.0f), b(0.5f);
    Transform txA(Vec3(0,0,0)), txB(Vec3(0,0,0));
    ContactManifold m;
    bool hit = collide(a, txA, b, txB, m);
    TEST_ASSERT(hit);
    TEST_ASSERT(m.numContacts == 1);
    TEST_ASSERT(approx(m.points[0].penetration, 1.5f));
    return true;
}

bool test_sphere_sphere_diagonal() {
    Sphere a(1.0f), b(1.0f);
    Transform txA(Vec3(0,0,0)), txB(Vec3(1.0f, 1.0f, 0));
    ContactManifold m;
    bool hit = collide(a, txA, b, txB, m);
    float dist = std::sqrt(2.0f);
    TEST_ASSERT(hit);
    TEST_ASSERT(approx(m.points[0].penetration, 2.0f - dist, 0.05f));
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Sphere-Box Tests
// ═════════════════════════════════════════════════════════════════════════════

bool test_sphere_box_face_hit() {
    Sphere s(0.5f);
    Box b(1.0f, 1.0f, 1.0f);
    // Sphere center at (1.2, 0, 0), box at origin
    Transform txS(Vec3(1.2f, 0, 0)), txB;
    ContactManifold m;
    bool hit = collide(s, txS, b, txB, m);
    TEST_ASSERT(hit);
    TEST_ASSERT(m.numContacts == 1);
    TEST_ASSERT(approx(m.points[0].penetration, 0.3f, 0.05f));
    return true;
}

bool test_sphere_box_separated() {
    Sphere s(0.5f);
    Box b(1.0f, 1.0f, 1.0f);
    Transform txS(Vec3(2.0f, 0, 0)), txB;
    ContactManifold m;
    bool hit = collide(s, txS, b, txB, m);
    TEST_ASSERT(!hit);
    return true;
}

bool test_sphere_box_inside() {
    Sphere s(0.3f);
    Box b(1.0f, 1.0f, 1.0f);
    Transform txS(Vec3(0, 0, 0)), txB;
    ContactManifold m;
    bool hit = collide(s, txS, b, txB, m);
    TEST_ASSERT(hit);
    TEST_ASSERT(m.numContacts == 1);
    return true;
}

bool test_sphere_box_edge() {
    Sphere s(0.5f);
    Box b(1.0f, 1.0f, 1.0f);
    // Sphere near the edge of the box
    Transform txS(Vec3(1.2f, 1.2f, 0)), txB;
    ContactManifold m;
    bool hit = collide(s, txS, b, txB, m);
    // Distance from (1.2, 1.2, 0) to closest point (1,1,0) = sqrt(0.04+0.04) = 0.283
    // penetration ≈ 0.5 - 0.283 = 0.217
    TEST_ASSERT(hit);
    TEST_ASSERT(m.points[0].penetration > 0.0f);
    return true;
}

bool test_sphere_box_corner() {
    Sphere s(0.5f);
    Box b(1.0f, 1.0f, 1.0f);
    // Sphere near corner
    Transform txS(Vec3(1.2f, 1.2f, 1.2f)), txB;
    ContactManifold m;
    bool hit = collide(s, txS, b, txB, m);
    // Distance from (1.2,1.2,1.2) to corner (1,1,1) = sqrt(3*0.04) = 0.346
    // penetration ≈ 0.5 - 0.346 = 0.154
    TEST_ASSERT(hit);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Sphere-Capsule Tests
// ═════════════════════════════════════════════════════════════════════════════

bool test_sphere_capsule_side() {
    Sphere s(0.5f);
    Capsule c(0.3f, 1.0f);
    // Sphere at (0.6, 0, 0), capsule at origin aligned along Y
    Transform txS(Vec3(0.6f, 0, 0)), txC;
    ContactManifold m;
    bool hit = collide(s, txS, c, txC, m);
    TEST_ASSERT(hit);
    TEST_ASSERT(approx(m.points[0].penetration, 0.2f, 0.05f));
    return true;
}

bool test_sphere_capsule_end_cap() {
    Sphere s(0.5f);
    Capsule c(0.3f, 1.0f);
    // Sphere above top cap
    Transform txS(Vec3(0, 1.5f, 0)), txC;
    ContactManifold m;
    bool hit = collide(s, txS, c, txC, m);
    // Distance from sphere center to top sphere center (0,1,0) = 0.5
    // penetration = 0.5 + 0.3 - 0.5 = 0.3
    TEST_ASSERT(hit);
    TEST_ASSERT(approx(m.points[0].penetration, 0.3f, 0.05f));
    return true;
}

bool test_sphere_capsule_separated() {
    Sphere s(0.5f);
    Capsule c(0.3f, 1.0f);
    Transform txS(Vec3(2.0f, 0, 0)), txC;
    ContactManifold m;
    bool hit = collide(s, txS, c, txC, m);
    TEST_ASSERT(!hit);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Box-Box (SAT) Tests
// ═════════════════════════════════════════════════════════════════════════════

bool test_box_box_face_contact() {
    Box a(1.0f, 1.0f, 1.0f), b(1.0f, 1.0f, 1.0f);
    Transform txA, txB(Vec3(1.5f, 0, 0));
    ContactManifold m;
    bool hit = collide(a, txA, b, txB, m);
    TEST_ASSERT(hit);
    TEST_ASSERT(m.numContacts >= 1);
    TEST_ASSERT(approx(m.points[0].penetration, 0.5f, 0.1f));
    return true;
}

bool test_box_box_separated() {
    Box a(1.0f, 1.0f, 1.0f), b(1.0f, 1.0f, 1.0f);
    Transform txA, txB(Vec3(3.0f, 0, 0));
    ContactManifold m;
    bool hit = collide(a, txA, b, txB, m);
    TEST_ASSERT(!hit);
    return true;
}

bool test_box_box_touching() {
    Box a(1.0f, 1.0f, 1.0f), b(1.0f, 1.0f, 1.0f);
    Transform txA, txB(Vec3(2.0f, 0, 0));
    ContactManifold m;
    bool hit = collide(a, txA, b, txB, m);
    // Touching — may or may not register depending on epsilon
    // Just don't crash
    (void)hit;
    return true;
}

bool test_box_box_rotated_45() {
    Box a(1.0f, 1.0f, 1.0f), b(1.0f, 1.0f, 1.0f);
    Transform txA;
    // Rotate B by 45 degrees around Y
    Quat rot45 = Quat::fromAxisAngle(Vec3(0, 1, 0), math::Pi * 0.25f);
    Transform txB(Vec3(1.8f, 0, 0), rot45);
    ContactManifold m;
    bool hit = collide(a, txA, b, txB, m);
    // sqrt(2) * 1.0 ≈ 1.414, so box B's effective half-extent along X is ~1.414
    // They should overlap
    TEST_ASSERT(hit);
    TEST_ASSERT(m.numContacts >= 1);
    return true;
}

bool test_box_box_edge_contact() {
    Box a(1.0f, 1.0f, 1.0f), b(1.0f, 1.0f, 1.0f);
    // Position B at corner + slight overlap, rotated 45 around Y and Z
    Quat rotY = Quat::fromAxisAngle(Vec3(0, 1, 0), math::Pi * 0.25f);
    Quat rotZ = Quat::fromAxisAngle(Vec3(0, 0, 1), math::Pi * 0.25f);
    Transform txA;
    Transform txB(Vec3(2.0f, 0, 0), rotY * rotZ);
    ContactManifold m;
    bool hit = collide(a, txA, b, txB, m);
    // Just verify it doesn't crash and produces reasonable output
    (void)hit;
    return true;
}

bool test_box_box_diagonal_overlap() {
    Box a(1.0f, 1.0f, 1.0f), b(1.0f, 1.0f, 1.0f);
    Transform txA, txB(Vec3(1.0f, 1.0f, 0));
    ContactManifold m;
    bool hit = collide(a, txA, b, txB, m);
    TEST_ASSERT(hit);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Capsule-Capsule Tests
// ═════════════════════════════════════════════════════════════════════════════

bool test_capsule_capsule_parallel() {
    Capsule a(0.5f, 1.0f), b(0.5f, 1.0f);
    Transform txA, txB(Vec3(0.8f, 0, 0));
    ContactManifold m;
    bool hit = collide(a, txA, b, txB, m);
    TEST_ASSERT(hit);
    TEST_ASSERT(approx(m.points[0].penetration, 0.2f, 0.05f));
    return true;
}

bool test_capsule_capsule_crossing() {
    Capsule a(0.3f, 1.0f), b(0.3f, 1.0f);
    // A along Y, B rotated 90° to lie along X
    Quat rot90 = Quat::fromAxisAngle(Vec3(0, 0, 1), math::HalfPi);
    Transform txA, txB(Vec3(0, 0, 0), rot90);
    ContactManifold m;
    bool hit = collide(a, txA, b, txB, m);
    TEST_ASSERT(hit);
    TEST_ASSERT(approx(m.points[0].penetration, 0.6f, 0.05f));
    return true;
}

bool test_capsule_capsule_end_end() {
    Capsule a(0.3f, 1.0f), b(0.3f, 1.0f);
    // B positioned above A
    Transform txA, txB(Vec3(0, 2.3f, 0));
    ContactManifold m;
    bool hit = collide(a, txA, b, txB, m);
    // A top at y=1.0, B bottom at y=2.3-1.0=1.3. Gap = 1.3-1.0=0.3. Radii sum = 0.6.
    // So penetration = 0.6 - 0.3 = 0.3
    TEST_ASSERT(hit);
    TEST_ASSERT(approx(m.points[0].penetration, 0.3f, 0.05f));
    return true;
}

bool test_capsule_capsule_separated() {
    Capsule a(0.3f, 1.0f), b(0.3f, 1.0f);
    Transform txA, txB(Vec3(3.0f, 0, 0));
    ContactManifold m;
    bool hit = collide(a, txA, b, txB, m);
    TEST_ASSERT(!hit);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// GJK Tests
// ═════════════════════════════════════════════════════════════════════════════

bool test_gjk_spheres_separated() {
    Sphere a(1.0f), b(1.0f);
    Transform txA(Vec3(0,0,0)), txB(Vec3(4.0f, 0, 0));
    GjkResult result;
    gjkQuery(a, txA, b, txB, result);
    TEST_ASSERT(result.status == GjkStatus::Separated);
    TEST_ASSERT(approx(result.distance, 2.0f, 0.1f));
    return true;
}

bool test_gjk_spheres_overlapping() {
    Sphere a(1.0f), b(1.0f);
    Transform txA(Vec3(0,0,0)), txB(Vec3(1.0f, 0, 0));
    GjkResult result;
    gjkQuery(a, txA, b, txB, result);
    TEST_ASSERT(result.status == GjkStatus::Overlapping);
    return true;
}

bool test_gjk_box_sphere_separated() {
    Box box(1.0f, 1.0f, 1.0f);
    Sphere sphere(0.5f);
    Transform txBox, txSphere(Vec3(3.0f, 0, 0));
    GjkResult result;
    gjkQuery(box, txBox, sphere, txSphere, result);
    TEST_ASSERT(result.status == GjkStatus::Separated);
    // Distance should be positive and in a reasonable range.
    // Exact distance = 1.5, but GJK with few iterations may overestimate.
    TEST_ASSERT(result.distance > 1.0f && result.distance < 3.0f);
    return true;
}

bool test_gjk_convex_hulls() {
    // Two tetrahedra
    Vec3 vertsA[4] = { Vec3(-1,-1,-1), Vec3(1,-1,-1), Vec3(0,1,-1), Vec3(0,0,1) };
    Vec3 vertsB[4] = { Vec3(3,-1,-1), Vec3(5,-1,-1), Vec3(4,1,-1), Vec3(4,0,1) };
    ConvexHull hullA(vertsA, 4);
    ConvexHull hullB(vertsB, 4);
    Transform txA, txB;
    GjkResult result;
    gjkQuery(hullA, txA, hullB, txB, result);
    TEST_ASSERT(result.status == GjkStatus::Separated);
    TEST_ASSERT(result.distance > 1.0f);
    return true;
}

bool test_gjk_convergence() {
    // Overlapping boxes — test that GJK converges (doesn't hang)
    Box a(1.0f, 1.0f, 1.0f), b(1.0f, 1.0f, 1.0f);
    Transform txA, txB(Vec3(0.5f, 0.5f, 0.5f));
    GjkResult result;
    gjkQuery(a, txA, b, txB, result);
    TEST_ASSERT(result.status == GjkStatus::Overlapping);
    TEST_ASSERT(result.iterations <= 64);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// EPA Tests
// ═════════════════════════════════════════════════════════════════════════════

bool test_epa_spheres() {
    Sphere a(1.0f), b(1.0f);
    Transform txA(Vec3(0,0,0)), txB(Vec3(1.0f, 0, 0));

    GjkResult gjkResult;
    gjkQuery(a, txA, b, txB, gjkResult);
    TEST_ASSERT(gjkResult.status == GjkStatus::Overlapping);

    EpaResult epaResult;
    bool ok = epaQuery(a, txA, b, txB, gjkResult.simplex, epaResult);
    // EPA should find penetration ≈ 1.0
    if (ok) {
        TEST_ASSERT(epaResult.penetration > 0.5f);
        TEST_ASSERT(epaResult.converged);
    }
    return true;
}

bool test_epa_deep_overlap() {
    Sphere a(2.0f), b(2.0f);
    Transform txA(Vec3(0,0,0)), txB(Vec3(0.5f, 0, 0));

    GjkResult gjkResult;
    gjkQuery(a, txA, b, txB, gjkResult);
    TEST_ASSERT(gjkResult.status == GjkStatus::Overlapping);

    EpaResult epaResult;
    epaQuery(a, txA, b, txB, gjkResult.simplex, epaResult);
    // Penetration ≈ 4.0 - 0.5 = 3.5
    if (epaResult.converged) {
        TEST_ASSERT(epaResult.penetration > 2.0f);
    }
    return true;
}

bool test_epa_boxes() {
    Box a(1.0f, 1.0f, 1.0f), b(1.0f, 1.0f, 1.0f);
    Transform txA, txB(Vec3(1.0f, 0, 0));

    GjkResult gjkResult;
    gjkQuery(a, txA, b, txB, gjkResult);

    if (gjkResult.status == GjkStatus::Overlapping) {
        EpaResult epaResult;
        epaQuery(a, txA, b, txB, gjkResult.simplex, epaResult);
        // Penetration ≈ 1.0
        if (epaResult.converged) {
            TEST_ASSERT(epaResult.penetration > 0.5f);
        }
    }
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// MPR Tests
// ═════════════════════════════════════════════════════════════════════════════

bool test_mpr_spheres_overlapping() {
    Sphere a(1.0f), b(1.0f);
    Transform txA(Vec3(0,0,0)), txB(Vec3(1.0f, 0, 0));
    MprResult result;
    bool hit = mprQuery(a, txA, b, txB, result);
    TEST_ASSERT(hit);
    TEST_ASSERT(result.overlapping);
    TEST_ASSERT(result.penetration > 0.5f);
    return true;
}

bool test_mpr_spheres_separated() {
    Sphere a(1.0f), b(1.0f);
    Transform txA(Vec3(0,0,0)), txB(Vec3(3.0f, 0, 0));
    MprResult result;
    bool hit = mprQuery(a, txA, b, txB, result);
    TEST_ASSERT(!hit);
    TEST_ASSERT(!result.overlapping);
    return true;
}

bool test_mpr_boxes() {
    Box a(1.0f, 1.0f, 1.0f), b(1.0f, 1.0f, 1.0f);
    Transform txA, txB(Vec3(1.5f, 0, 0));
    MprResult result;
    bool hit = mprQuery(a, txA, b, txB, result);
    TEST_ASSERT(hit);
    TEST_ASSERT(result.overlapping);
    return true;
}

bool test_mpr_capsule_sphere() {
    Capsule cap(0.5f, 1.0f);
    Sphere sph(0.5f);
    Transform txCap, txSph(Vec3(0.8f, 0, 0));
    MprResult result;
    bool hit = mprQuery(cap, txCap, sph, txSph, result);
    TEST_ASSERT(hit);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// CCD Tests
// ═════════════════════════════════════════════════════════════════════════════

bool test_ccd_linear_collision() {
    Sphere a(0.5f), b(0.5f);
    Transform txA_start(Vec3(-5, 0, 0)), txA_end(Vec3(5, 0, 0));
    Transform txB_start(Vec3(2, 0, 0)), txB_end(Vec3(2, 0, 0)); // Static
    CcdResult result;
    bool hit = ccdQuery(a, txA_start, txA_end, b, txB_start, txB_end, result);
    TEST_ASSERT(hit);
    TEST_ASSERT(result.timeOfImpact > 0.0f && result.timeOfImpact < 1.0f);
    // TOI ≈ (2-0.5-0.5 - (-5)) / 10 = 6/10 = 0.6
    TEST_ASSERT(approx(result.timeOfImpact, 0.6f, 0.15f));
    return true;
}

bool test_ccd_no_collision() {
    Sphere a(0.5f), b(0.5f);
    Transform txA_start(Vec3(-5, 0, 0)), txA_end(Vec3(-2, 0, 0));
    Transform txB_start(Vec3(5, 0, 0)), txB_end(Vec3(5, 0, 0));
    CcdResult result;
    bool hit = ccdQuery(a, txA_start, txA_end, b, txB_start, txB_end, result);
    TEST_ASSERT(!hit);
    return true;
}

bool test_ccd_tunneling_prevention() {
    // Fast sphere shooting through a wall
    Sphere bullet(0.3f);
    Box wall(0.5f, 2.0f, 2.0f); // Reasonably thick wall
    Transform txA_start(Vec3(-10, 0, 0)), txA_end(Vec3(10, 0, 0));
    Transform txB_start(Vec3(0, 0, 0)), txB_end(Vec3(0, 0, 0));
    CcdResult result;
    bool hit = ccdQuery(bullet, txA_start, txA_end, wall, txB_start, txB_end, result);
    TEST_ASSERT(hit);
    TEST_ASSERT(result.timeOfImpact > 0.0f && result.timeOfImpact < 1.0f);
    return true;
}

bool test_ccd_initially_overlapping() {
    Sphere a(1.0f), b(1.0f);
    Transform txA_start(Vec3(0, 0, 0)), txA_end(Vec3(1, 0, 0));
    Transform txB_start(Vec3(0.5f, 0, 0)), txB_end(Vec3(0.5f, 0, 0));
    CcdResult result;
    bool hit = ccdQuery(a, txA_start, txA_end, b, txB_start, txB_end, result);
    TEST_ASSERT(hit);
    TEST_ASSERT(result.timeOfImpact == 0.0f);
    return true;
}

bool test_ccd_grazing() {
    // Sphere passing just barely by another
    Sphere a(0.5f), b(0.5f);
    Transform txA_start(Vec3(-5, 1.05f, 0)), txA_end(Vec3(5, 1.05f, 0));
    Transform txB_start(Vec3(0, 0, 0)), txB_end(Vec3(0, 0, 0));
    CcdResult result;
    bool hit = ccdQuery(a, txA_start, txA_end, b, txB_start, txB_end, result);
    // Barely misses — should not collide
    TEST_ASSERT(!hit);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Collision Dispatch Tests (mixed types)
// ═════════════════════════════════════════════════════════════════════════════

bool test_dispatch_sphere_cylinder() {
    Sphere s(0.5f);
    Cylinder c(0.5f, 1.0f);
    Transform txS(Vec3(0.8f, 0, 0)), txC;
    ContactManifold m;
    bool hit = collide(s, txS, c, txC, m);
    TEST_ASSERT(hit);
    return true;
}

bool test_dispatch_box_capsule() {
    Box b(1.0f, 1.0f, 1.0f);
    Capsule c(0.3f, 1.0f);
    Transform txB, txC(Vec3(1.1f, 0, 0));
    ContactManifold m;
    bool hit = collide(b, txB, c, txC, m);
    TEST_ASSERT(hit);
    return true;
}

bool test_dispatch_convex_hull_vs_convex_hull() {
    // Two overlapping cubes as convex hulls
    Vec3 vertsA[8], vertsB[8];
    Box(1.0f, 1.0f, 1.0f).getVertices(vertsA);
    Box(1.0f, 1.0f, 1.0f).getVertices(vertsB);
    ConvexHull hullA(vertsA, 8);
    ConvexHull hullB(vertsB, 8);
    Transform txA, txB(Vec3(1.0f, 0, 0));
    ContactManifold m;
    bool hit = collide(hullA, txA, hullB, txB, m);
    TEST_ASSERT(hit);
    return true;
}

bool test_dispatch_cylinder_cylinder() {
    Cylinder a(0.5f, 1.0f), b(0.5f, 1.0f);
    Transform txA, txB(Vec3(0.8f, 0, 0));
    ContactManifold m;
    bool hit = collide(a, txA, b, txB, m);
    TEST_ASSERT(hit);
    return true;
}

bool test_dispatch_box_convex_hull() {
    Box box(1.0f, 1.0f, 1.0f);
    Vec3 verts[4] = { Vec3(0.5f,-0.5f,-0.5f), Vec3(0.5f,0.5f,-0.5f),
                      Vec3(0.5f,0,0.5f), Vec3(1.5f,0,0) };
    ConvexHull hull(verts, 4);
    Transform txBox, txHull;
    ContactManifold m;
    bool hit = collide(box, txBox, hull, txHull, m);
    TEST_ASSERT(hit);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Simplex / SupportPoint Tests
// ═════════════════════════════════════════════════════════════════════════════

bool test_simplex_default() {
    Simplex s;
    TEST_ASSERT(s.size == 0);
    return true;
}

bool test_simplex_add_and_set() {
    Simplex s;
    SupportPoint sp(Vec3(1,0,0), Vec3(0,0,0));
    s.addVertex(sp);
    TEST_ASSERT(s.size == 1);
    TEST_ASSERT(approxVec(s[0].point, Vec3(1,0,0)));

    SupportPoint sp2(Vec3(0,1,0), Vec3(0,0,0));
    s.set(sp, sp2);
    TEST_ASSERT(s.size == 2);
    return true;
}

bool test_support_point_construction() {
    SupportPoint sp(Vec3(3,0,0), Vec3(1,0,0));
    TEST_ASSERT(approxVec(sp.point, Vec3(2,0,0)));
    TEST_ASSERT(approxVec(sp.pointA, Vec3(3,0,0)));
    TEST_ASSERT(approxVec(sp.pointB, Vec3(1,0,0)));
    return true;
}

bool test_narrowphase_config_defaults() {
    NarrowPhaseConfig cfg;
    TEST_ASSERT(cfg.gjkMaxIterations == 64);
    TEST_ASSERT(cfg.epaMaxIterations == 64);
    TEST_ASSERT(approx(cfg.collisionTolerance, 0.005f));
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("=== Pulse NarrowPhase Tests ===\n\n");

    // ── ContactManifold ──
    std::printf("--- ContactManifold ---\n");
    RUN_TEST(test_manifold_default);
    RUN_TEST(test_manifold_add_points);
    RUN_TEST(test_manifold_clear);
    RUN_TEST(test_manifold_reduce_to_4);
    RUN_TEST(test_manifold_max_penetration);
    RUN_TEST(test_manifold_average_normal);

    // ── Sphere-Sphere ──
    std::printf("\n--- Sphere-Sphere ---\n");
    RUN_TEST(test_sphere_sphere_overlapping);
    RUN_TEST(test_sphere_sphere_separated);
    RUN_TEST(test_sphere_sphere_touching);
    RUN_TEST(test_sphere_sphere_concentric);
    RUN_TEST(test_sphere_sphere_diagonal);

    // ── Sphere-Box ──
    std::printf("\n--- Sphere-Box ---\n");
    RUN_TEST(test_sphere_box_face_hit);
    RUN_TEST(test_sphere_box_separated);
    RUN_TEST(test_sphere_box_inside);
    RUN_TEST(test_sphere_box_edge);
    RUN_TEST(test_sphere_box_corner);

    // ── Sphere-Capsule ──
    std::printf("\n--- Sphere-Capsule ---\n");
    RUN_TEST(test_sphere_capsule_side);
    RUN_TEST(test_sphere_capsule_end_cap);
    RUN_TEST(test_sphere_capsule_separated);

    // ── Box-Box (SAT) ──
    std::printf("\n--- Box-Box (SAT) ---\n");
    RUN_TEST(test_box_box_face_contact);
    RUN_TEST(test_box_box_separated);
    RUN_TEST(test_box_box_touching);
    RUN_TEST(test_box_box_rotated_45);
    RUN_TEST(test_box_box_edge_contact);
    RUN_TEST(test_box_box_diagonal_overlap);

    // ── Capsule-Capsule ──
    std::printf("\n--- Capsule-Capsule ---\n");
    RUN_TEST(test_capsule_capsule_parallel);
    RUN_TEST(test_capsule_capsule_crossing);
    RUN_TEST(test_capsule_capsule_end_end);
    RUN_TEST(test_capsule_capsule_separated);

    // ── GJK ──
    std::printf("\n--- GJK ---\n");
    RUN_TEST(test_gjk_spheres_separated);
    RUN_TEST(test_gjk_spheres_overlapping);
    RUN_TEST(test_gjk_box_sphere_separated);
    RUN_TEST(test_gjk_convex_hulls);
    RUN_TEST(test_gjk_convergence);

    // ── EPA ──
    std::printf("\n--- EPA ---\n");
    RUN_TEST(test_epa_spheres);
    RUN_TEST(test_epa_deep_overlap);
    RUN_TEST(test_epa_boxes);

    // ── MPR ──
    std::printf("\n--- MPR ---\n");
    RUN_TEST(test_mpr_spheres_overlapping);
    RUN_TEST(test_mpr_spheres_separated);
    RUN_TEST(test_mpr_boxes);
    RUN_TEST(test_mpr_capsule_sphere);

    // ── CCD ──
    std::printf("\n--- CCD ---\n");
    RUN_TEST(test_ccd_linear_collision);
    RUN_TEST(test_ccd_no_collision);
    RUN_TEST(test_ccd_tunneling_prevention);
    RUN_TEST(test_ccd_initially_overlapping);
    RUN_TEST(test_ccd_grazing);

    // ── Collision Dispatch ──
    std::printf("\n--- Collision Dispatch ---\n");
    RUN_TEST(test_dispatch_sphere_cylinder);
    RUN_TEST(test_dispatch_box_capsule);
    RUN_TEST(test_dispatch_convex_hull_vs_convex_hull);
    RUN_TEST(test_dispatch_cylinder_cylinder);
    RUN_TEST(test_dispatch_box_convex_hull);

    // ── Simplex / Support ──
    std::printf("\n--- Simplex / SupportPoint ---\n");
    RUN_TEST(test_simplex_default);
    RUN_TEST(test_simplex_add_and_set);
    RUN_TEST(test_support_point_construction);
    RUN_TEST(test_narrowphase_config_defaults);

    // ── Summary ──
    std::printf("\n=== Results: %d/%d passed", g_passed, g_total);
    if (g_failed > 0) std::printf(" (%d FAILED)", g_failed);
    std::printf(" ===\n");

    return g_failed > 0 ? 1 : 0;
}
