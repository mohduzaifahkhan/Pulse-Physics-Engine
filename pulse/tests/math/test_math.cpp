/**
 * @file test_math.cpp
 * @brief Comprehensive unit tests for the Pulse math library.
 *
 * Lightweight test framework — no external dependencies. Each test is a
 * function that returns true on pass. Failed tests print the function name
 * and line number.
 */

#include <pulse/math/math_common.h>
#include <pulse/math/vec2.h>
#include <pulse/math/vec3.h>
#include <pulse/math/vec4.h>
#include <pulse/math/quat.h>
#include <pulse/math/mat3.h>
#include <pulse/math/mat4.h>
#include <pulse/math/transform.h>
#include <pulse/math/aabb.h>
#include <pulse/math/plane.h>
#include <pulse/math/ray.h>

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

// ── Vec2 Tests ────────────────────────────────────────────────────────────────

bool test_vec2_construction() {
    Vec2 a;
    TEST_ASSERT(a.x == 0.0f && a.y == 0.0f);

    Vec2 b(3.0f, 4.0f);
    TEST_ASSERT(b.x == 3.0f && b.y == 4.0f);

    Vec2 c(5.0f);
    TEST_ASSERT(c.x == 5.0f && c.y == 5.0f);
    return true;
}

bool test_vec2_arithmetic() {
    Vec2 a(1.0f, 2.0f);
    Vec2 b(3.0f, 4.0f);

    Vec2 sum = a + b;
    TEST_ASSERT(sum == Vec2(4.0f, 6.0f));

    Vec2 diff = b - a;
    TEST_ASSERT(diff == Vec2(2.0f, 2.0f));

    Vec2 scaled = a * 3.0f;
    TEST_ASSERT(scaled == Vec2(3.0f, 6.0f));

    Vec2 neg = -a;
    TEST_ASSERT(neg == Vec2(-1.0f, -2.0f));
    return true;
}

bool test_vec2_length_normalize() {
    Vec2 v(3.0f, 4.0f);
    TEST_ASSERT(math::approxEqual(v.length(), 5.0f, 0.001f));
    TEST_ASSERT(math::approxEqual(v.lengthSq(), 25.0f));

    Vec2 n = v.normalized();
    TEST_ASSERT(math::approxEqual(n.length(), 1.0f, 0.001f));
    return true;
}

bool test_vec2_dot_cross() {
    Vec2 a(1.0f, 0.0f);
    Vec2 b(0.0f, 1.0f);
    TEST_ASSERT(math::approxEqual(a.dot(b), 0.0f));
    TEST_ASSERT(math::approxEqual(a.cross(b), 1.0f));

    Vec2 c(1.0f, 1.0f);
    TEST_ASSERT(math::approxEqual(a.dot(c), 1.0f));
    return true;
}

// ── Vec3 Tests ────────────────────────────────────────────────────────────────

bool test_vec3_construction() {
    Vec3 a;
    TEST_ASSERT(math::approxEqual(a.getX(), 0.0f));
    TEST_ASSERT(math::approxEqual(a.getY(), 0.0f));
    TEST_ASSERT(math::approxEqual(a.getZ(), 0.0f));

    Vec3 b(1.0f, 2.0f, 3.0f);
    TEST_ASSERT(math::approxEqual(b.getX(), 1.0f));
    TEST_ASSERT(math::approxEqual(b.getY(), 2.0f));
    TEST_ASSERT(math::approxEqual(b.getZ(), 3.0f));

    Vec3 c(5.0f);
    TEST_ASSERT(math::approxEqual(c.getX(), 5.0f));
    return true;
}

bool test_vec3_arithmetic() {
    Vec3 a(1.0f, 2.0f, 3.0f);
    Vec3 b(4.0f, 5.0f, 6.0f);

    Vec3 sum = a + b;
    TEST_ASSERT(sum == Vec3(5.0f, 7.0f, 9.0f));

    Vec3 diff = b - a;
    TEST_ASSERT(diff == Vec3(3.0f, 3.0f, 3.0f));

    Vec3 scaled = a * 2.0f;
    TEST_ASSERT(scaled == Vec3(2.0f, 4.0f, 6.0f));

    Vec3 neg = -a;
    TEST_ASSERT(neg == Vec3(-1.0f, -2.0f, -3.0f));
    return true;
}

bool test_vec3_dot() {
    Vec3 a(1.0f, 0.0f, 0.0f);
    Vec3 b(0.0f, 1.0f, 0.0f);
    TEST_ASSERT(math::approxEqual(a.dot(b), 0.0f));

    Vec3 c(1.0f, 2.0f, 3.0f);
    Vec3 d(4.0f, 5.0f, 6.0f);
    TEST_ASSERT(math::approxEqual(c.dot(d), 32.0f, 0.001f));
    return true;
}

bool test_vec3_cross() {
    Vec3 x = Vec3::unitX();
    Vec3 y = Vec3::unitY();
    Vec3 z = x.cross(y);
    TEST_ASSERT(z == Vec3::unitZ());

    Vec3 negX = y.cross(z);
    TEST_ASSERT(negX == Vec3::unitX());
    return true;
}

bool test_vec3_normalize() {
    Vec3 v(3.0f, 4.0f, 0.0f);
    TEST_ASSERT(math::approxEqual(v.length(), 5.0f, 0.001f));

    Vec3 n = v.normalized();
    TEST_ASSERT(math::approxEqual(n.length(), 1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(n.getX(), 0.6f, 0.01f));
    TEST_ASSERT(math::approxEqual(n.getY(), 0.8f, 0.01f));

    // Zero vector normalization should return zero
    Vec3 zero = Vec3::zero().normalized();
    TEST_ASSERT(zero == Vec3::zero());
    return true;
}

bool test_vec3_lerp() {
    Vec3 a(0.0f, 0.0f, 0.0f);
    Vec3 b(10.0f, 20.0f, 30.0f);
    Vec3 mid = a.lerp(b, 0.5f);
    TEST_ASSERT(mid == Vec3(5.0f, 10.0f, 15.0f));
    return true;
}

bool test_vec3_reflect() {
    Vec3 incident(1.0f, -1.0f, 0.0f);
    Vec3 normal(0.0f, 1.0f, 0.0f);
    Vec3 reflected = incident.reflect(normal);
    TEST_ASSERT(reflected == Vec3(1.0f, 1.0f, 0.0f));
    return true;
}

// ── Vec4 Tests ────────────────────────────────────────────────────────────────

bool test_vec4_construction() {
    Vec4 a;
    TEST_ASSERT(math::approxEqual(a.getX(), 0.0f));
    TEST_ASSERT(math::approxEqual(a.getW(), 0.0f));

    Vec4 b(1.0f, 2.0f, 3.0f, 4.0f);
    TEST_ASSERT(math::approxEqual(b.getW(), 4.0f));

    Vec4 c(Vec3(1.0f, 2.0f, 3.0f), 1.0f);
    TEST_ASSERT(math::approxEqual(c.getZ(), 3.0f));
    TEST_ASSERT(math::approxEqual(c.getW(), 1.0f));
    return true;
}

bool test_vec4_dot() {
    Vec4 a(1.0f, 2.0f, 3.0f, 4.0f);
    Vec4 b(4.0f, 3.0f, 2.0f, 1.0f);
    TEST_ASSERT(math::approxEqual(a.dot(b), 20.0f, 0.001f));
    return true;
}

bool test_vec4_xyz() {
    Vec4 v(1.0f, 2.0f, 3.0f, 4.0f);
    Vec3 xyz = v.xyz();
    TEST_ASSERT(xyz == Vec3(1.0f, 2.0f, 3.0f));
    return true;
}

// ── Quaternion Tests ──────────────────────────────────────────────────────────

bool test_quat_identity() {
    Quat q;
    TEST_ASSERT(math::approxEqual(q.getX(), 0.0f));
    TEST_ASSERT(math::approxEqual(q.getY(), 0.0f));
    TEST_ASSERT(math::approxEqual(q.getZ(), 0.0f));
    TEST_ASSERT(math::approxEqual(q.getW(), 1.0f));
    return true;
}

bool test_quat_rotation() {
    // Rotate (1,0,0) by 90° around Z axis should give (0,1,0)
    Quat q = Quat::fromAxisAngle(Vec3::unitZ(), math::HalfPi);
    Vec3 rotated = q.rotate(Vec3::unitX());
    TEST_ASSERT(math::approxEqual(rotated.getX(), 0.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(rotated.getY(), 1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(rotated.getZ(), 0.0f, 0.01f));
    return true;
}

bool test_quat_multiply() {
    // Two 90° rotations around Z = one 180° rotation
    Quat q90 = Quat::fromAxisAngle(Vec3::unitZ(), math::HalfPi);
    Quat q180 = q90 * q90;
    Vec3 rotated = q180.rotate(Vec3::unitX());
    TEST_ASSERT(math::approxEqual(rotated.getX(), -1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(rotated.getY(), 0.0f, 0.01f));
    return true;
}

bool test_quat_inverse() {
    Quat q = Quat::fromAxisAngle(Vec3::unitY(), math::Pi * 0.3f);
    Quat qInv = q.inverse();
    Quat identity = q * qInv;
    TEST_ASSERT(math::approxEqual(identity.getW(), 1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(identity.getX(), 0.0f, 0.01f));
    return true;
}

bool test_quat_slerp() {
    Quat a = Quat::identity();
    Quat b = Quat::fromAxisAngle(Vec3::unitZ(), math::HalfPi);
    Quat mid = a.slerp(b, 0.5f);
    Vec3 rotated = mid.rotate(Vec3::unitX());
    // Should be roughly 45° rotation
    float angle = std::atan2(rotated.getY(), rotated.getX());
    TEST_ASSERT(math::approxEqual(angle, math::Pi / 4.0f, 0.05f));
    return true;
}

bool test_quat_euler() {
    Quat q = Quat::fromEuler(0.0f, 0.0f, math::HalfPi); // 90° roll
    float pitch, yaw, roll;
    q.toEuler(pitch, yaw, roll);
    TEST_ASSERT(math::approxEqual(roll, math::HalfPi, 0.01f));
    return true;
}

// ── Mat3 Tests ────────────────────────────────────────────────────────────────

bool test_mat3_identity() {
    Mat3 m;
    Vec3 v(1.0f, 2.0f, 3.0f);
    Vec3 result = m * v;
    TEST_ASSERT(result == v);
    return true;
}

bool test_mat3_multiply() {
    Mat3 a(
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    );
    Mat3 identity;
    Mat3 result = a * identity;
    TEST_ASSERT(result == a);
    return true;
}

bool test_mat3_transpose() {
    Mat3 m(
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    );
    Mat3 t = m.transposed();
    TEST_ASSERT(math::approxEqual(t(0, 1), 4.0f));
    TEST_ASSERT(math::approxEqual(t(1, 0), 2.0f));
    TEST_ASSERT(math::approxEqual(t(2, 0), 3.0f));
    return true;
}

bool test_mat3_determinant() {
    Mat3 m(
        1.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f,
        0.0f, 0.0f, 3.0f
    );
    TEST_ASSERT(math::approxEqual(m.determinant(), 6.0f, 0.001f));
    return true;
}

bool test_mat3_inverse() {
    Mat3 m(
        2.0f, 0.0f, 0.0f,
        0.0f, 4.0f, 0.0f,
        0.0f, 0.0f, 8.0f
    );
    Mat3 inv = m.inversed();
    Mat3 identity = m * inv;
    TEST_ASSERT(math::approxEqual(identity(0, 0), 1.0f, 0.001f));
    TEST_ASSERT(math::approxEqual(identity(1, 1), 1.0f, 0.001f));
    TEST_ASSERT(math::approxEqual(identity(2, 2), 1.0f, 0.001f));
    TEST_ASSERT(math::approxEqual(identity(0, 1), 0.0f, 0.001f));
    return true;
}

bool test_mat3_from_quat() {
    Quat q = Quat::fromAxisAngle(Vec3::unitZ(), math::HalfPi);
    Mat3 m = Mat3::fromQuat(q);
    Vec3 v = m * Vec3::unitX();
    TEST_ASSERT(math::approxEqual(v.getX(), 0.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(v.getY(), 1.0f, 0.01f));
    return true;
}

// ── Mat4 Tests ────────────────────────────────────────────────────────────────

bool test_mat4_identity() {
    Mat4 m;
    Vec4 v(1.0f, 2.0f, 3.0f, 1.0f);
    Vec4 result = m * v;
    TEST_ASSERT(result == v);
    return true;
}

bool test_mat4_translation() {
    Mat4 m = Mat4::translation(Vec3(10.0f, 20.0f, 30.0f));
    Vec3 p = m.transformPoint(Vec3::zero());
    TEST_ASSERT(p == Vec3(10.0f, 20.0f, 30.0f));
    return true;
}

bool test_mat4_multiply() {
    Mat4 t = Mat4::translation(Vec3(1.0f, 2.0f, 3.0f));
    Mat4 s = Mat4::scale(2.0f);
    Mat4 ts = t * s;
    Vec3 p = ts.transformPoint(Vec3(1.0f, 0.0f, 0.0f));
    // Scale first: (2, 0, 0), then translate: (3, 2, 3)
    TEST_ASSERT(math::approxEqual(p.getX(), 3.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(p.getY(), 2.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(p.getZ(), 3.0f, 0.01f));
    return true;
}

bool test_mat4_transpose() {
    Mat4 m(
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    );
    Mat4 t = m.transposed();
    TEST_ASSERT(math::approxEqual(t(0, 1), 5.0f));
    TEST_ASSERT(math::approxEqual(t(1, 0), 2.0f));
    return true;
}

bool test_mat4_inverse() {
    Mat4 m = Mat4::translation(Vec3(5.0f, 10.0f, 15.0f));
    Mat4 inv = m.inversed();
    Mat4 identity = m * inv;
    TEST_ASSERT(math::approxEqual(identity(0, 0), 1.0f, 0.001f));
    TEST_ASSERT(math::approxEqual(identity(3, 3), 1.0f, 0.001f));
    TEST_ASSERT(math::approxEqual(identity(0, 3), 0.0f, 0.001f));
    return true;
}

bool test_mat4_affine_inverse() {
    Mat4 m = Mat4::trs(Vec3(1.0f, 2.0f, 3.0f), Quat::fromAxisAngle(Vec3::unitY(), 0.5f), Vec3(1.0f));
    Mat4 inv = m.affineInverse();
    Vec3 p = inv.transformPoint(m.transformPoint(Vec3(7.0f, 8.0f, 9.0f)));
    TEST_ASSERT(math::approxEqual(p.getX(), 7.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(p.getY(), 8.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(p.getZ(), 9.0f, 0.01f));
    return true;
}

// ── Transform Tests ───────────────────────────────────────────────────────────

bool test_transform_point() {
    Transform t(Vec3(10.0f, 0.0f, 0.0f), Quat::identity());
    Vec3 local(1.0f, 2.0f, 3.0f);
    Vec3 world = t.transformPoint(local);
    TEST_ASSERT(world == Vec3(11.0f, 2.0f, 3.0f));
    return true;
}

bool test_transform_inverse() {
    Transform t(Vec3(5.0f, 0.0f, 0.0f), Quat::fromAxisAngle(Vec3::unitZ(), 0.5f));
    Transform inv = t.inversed();
    Vec3 p(3.0f, 4.0f, 5.0f);
    Vec3 roundTrip = inv.transformPoint(t.transformPoint(p));
    TEST_ASSERT(math::approxEqual(roundTrip.getX(), p.getX(), 0.01f));
    TEST_ASSERT(math::approxEqual(roundTrip.getY(), p.getY(), 0.01f));
    TEST_ASSERT(math::approxEqual(roundTrip.getZ(), p.getZ(), 0.01f));
    return true;
}

bool test_transform_compose() {
    Transform a(Vec3(1.0f, 0.0f, 0.0f));
    Transform b(Vec3(0.0f, 2.0f, 0.0f));
    Transform c = a * b;
    Vec3 p = c.transformPoint(Vec3::zero());
    TEST_ASSERT(math::approxEqual(p.getX(), 1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(p.getY(), 2.0f, 0.01f));
    return true;
}

// ── AABB Tests ────────────────────────────────────────────────────────────────

bool test_aabb_overlap() {
    AABB a(Vec3(0.0f, 0.0f, 0.0f), Vec3(2.0f, 2.0f, 2.0f));
    AABB b(Vec3(1.0f, 1.0f, 1.0f), Vec3(3.0f, 3.0f, 3.0f));
    TEST_ASSERT(a.overlaps(b));

    AABB c(Vec3(5.0f, 5.0f, 5.0f), Vec3(6.0f, 6.0f, 6.0f));
    TEST_ASSERT(!a.overlaps(c));
    return true;
}

bool test_aabb_contains() {
    AABB outer(Vec3(0.0f, 0.0f, 0.0f), Vec3(10.0f, 10.0f, 10.0f));
    AABB inner(Vec3(2.0f, 2.0f, 2.0f), Vec3(8.0f, 8.0f, 8.0f));
    TEST_ASSERT(outer.contains(inner));
    TEST_ASSERT(!inner.contains(outer));
    return true;
}

bool test_aabb_point_containment() {
    AABB box(Vec3(0.0f, 0.0f, 0.0f), Vec3(1.0f, 1.0f, 1.0f));
    TEST_ASSERT(box.containsPoint(Vec3(0.5f, 0.5f, 0.5f)));
    TEST_ASSERT(!box.containsPoint(Vec3(2.0f, 0.5f, 0.5f)));
    return true;
}

bool test_aabb_merge() {
    AABB a(Vec3(0.0f, 0.0f, 0.0f), Vec3(1.0f, 1.0f, 1.0f));
    AABB b(Vec3(2.0f, 2.0f, 2.0f), Vec3(3.0f, 3.0f, 3.0f));
    AABB merged = a.merged(b);
    TEST_ASSERT(merged.min == Vec3(0.0f, 0.0f, 0.0f));
    TEST_ASSERT(merged.max == Vec3(3.0f, 3.0f, 3.0f));
    return true;
}

bool test_aabb_surface_area() {
    AABB box(Vec3(0.0f, 0.0f, 0.0f), Vec3(1.0f, 2.0f, 3.0f));
    // SA = 2*(1*2 + 2*3 + 3*1) = 2*(2+6+3) = 22
    TEST_ASSERT(math::approxEqual(box.surfaceArea(), 22.0f, 0.001f));
    return true;
}

bool test_aabb_ray_intersect() {
    AABB box(Vec3(-1.0f, -1.0f, -1.0f), Vec3(1.0f, 1.0f, 1.0f));
    Ray ray(Vec3(-5.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f));
    float tMin, tMax;
    TEST_ASSERT(ray.intersectAABB(box, tMin, tMax));
    TEST_ASSERT(math::approxEqual(tMin, 4.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(tMax, 6.0f, 0.01f));

    // Ray missing the box
    Ray miss(Vec3(-5.0f, 5.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f));
    TEST_ASSERT(!miss.intersectAABB(box, tMin, tMax));
    return true;
}

// ── Plane Tests ───────────────────────────────────────────────────────────────

bool test_plane_distance() {
    Plane p(Vec3::unitY(), 0.0f); // XZ plane at y=0
    TEST_ASSERT(math::approxEqual(p.signedDistanceTo(Vec3(0.0f, 5.0f, 0.0f)), 5.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(p.signedDistanceTo(Vec3(0.0f, -3.0f, 0.0f)), -3.0f, 0.01f));
    return true;
}

bool test_plane_classify() {
    Plane p(Vec3::unitY(), 0.0f);
    TEST_ASSERT(p.classify(Vec3(0.0f, 1.0f, 0.0f)) == 1);
    TEST_ASSERT(p.classify(Vec3(0.0f, -1.0f, 0.0f)) == -1);
    TEST_ASSERT(p.classify(Vec3(0.0f, 0.0f, 0.0f)) == 0);
    return true;
}

bool test_plane_project() {
    Plane p(Vec3::unitY(), 0.0f);
    Vec3 projected = p.projectPoint(Vec3(3.0f, 5.0f, 7.0f));
    TEST_ASSERT(math::approxEqual(projected.getX(), 3.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(projected.getY(), 0.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(projected.getZ(), 7.0f, 0.01f));
    return true;
}

// ── Ray Tests ─────────────────────────────────────────────────────────────────

bool test_ray_point_at() {
    Ray r(Vec3(0.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f));
    Vec3 p = r.pointAt(5.0f);
    TEST_ASSERT(p == Vec3(5.0f, 0.0f, 0.0f));
    return true;
}

bool test_ray_sphere_intersect() {
    Ray r(Vec3(-5.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f));
    float t;
    TEST_ASSERT(r.intersectSphere(Vec3::zero(), 1.0f, t));
    TEST_ASSERT(math::approxEqual(t, 4.0f, 0.01f));
    return true;
}

bool test_ray_triangle_intersect() {
    Vec3 v0(0.0f, 0.0f, 0.0f);
    Vec3 v1(1.0f, 0.0f, 0.0f);
    Vec3 v2(0.0f, 1.0f, 0.0f);

    Ray r(Vec3(0.2f, 0.2f, -1.0f), Vec3(0.0f, 0.0f, 1.0f));
    float t, u, v;
    TEST_ASSERT(r.intersectTriangle(v0, v1, v2, t, u, v));
    TEST_ASSERT(math::approxEqual(t, 1.0f, 0.01f));

    // Miss
    Ray miss(Vec3(2.0f, 2.0f, -1.0f), Vec3(0.0f, 0.0f, 1.0f));
    TEST_ASSERT(!miss.intersectTriangle(v0, v1, v2, t, u, v));
    return true;
}

bool test_ray_closest_point() {
    Ray r(Vec3::zero(), Vec3(1.0f, 0.0f, 0.0f));
    Vec3 p(3.0f, 4.0f, 0.0f);
    Vec3 closest = r.closestPoint(p);
    TEST_ASSERT(closest == Vec3(3.0f, 0.0f, 0.0f));
    return true;
}

// ── Utility function tests ───────────────────────────────────────────────────

bool test_math_fast_sqrt() {
    TEST_ASSERT(math::approxEqual(math::fastSqrt(4.0f), 2.0f, 0.001f));
    TEST_ASSERT(math::approxEqual(math::fastSqrt(9.0f), 3.0f, 0.001f));
    TEST_ASSERT(math::approxEqual(math::fastSqrt(1.0f), 1.0f, 0.001f));
    return true;
}

bool test_math_fast_inv_sqrt() {
    TEST_ASSERT(math::approxEqual(math::fastInvSqrt(4.0f), 0.5f, 0.01f));
    TEST_ASSERT(math::approxEqual(math::fastInvSqrt(1.0f), 1.0f, 0.01f));
    return true;
}

bool test_orthonormal_basis() {
    Vec3 n = Vec3::unitY();
    Vec3 t, b;
    math::orthonormalBasis(n, t, b);
    // t and b should be perpendicular to n and each other
    TEST_ASSERT(math::approxEqual(n.dot(t), 0.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(n.dot(b), 0.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(t.dot(b), 0.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(t.length(), 1.0f, 0.01f));
    TEST_ASSERT(math::approxEqual(b.length(), 1.0f, 0.01f));
    return true;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    std::printf("╔══════════════════════════════════════════════╗\n");
    std::printf("║       PULSE Physics Engine - Math Tests      ║\n");
    std::printf("╚══════════════════════════════════════════════╝\n\n");

    // Vec2
    std::printf("── Vec2 ─────────────────────────────\n");
    RUN_TEST(test_vec2_construction);
    RUN_TEST(test_vec2_arithmetic);
    RUN_TEST(test_vec2_length_normalize);
    RUN_TEST(test_vec2_dot_cross);

    // Vec3
    std::printf("── Vec3 ─────────────────────────────\n");
    RUN_TEST(test_vec3_construction);
    RUN_TEST(test_vec3_arithmetic);
    RUN_TEST(test_vec3_dot);
    RUN_TEST(test_vec3_cross);
    RUN_TEST(test_vec3_normalize);
    RUN_TEST(test_vec3_lerp);
    RUN_TEST(test_vec3_reflect);

    // Vec4
    std::printf("── Vec4 ─────────────────────────────\n");
    RUN_TEST(test_vec4_construction);
    RUN_TEST(test_vec4_dot);
    RUN_TEST(test_vec4_xyz);

    // Quaternion
    std::printf("── Quaternion ───────────────────────\n");
    RUN_TEST(test_quat_identity);
    RUN_TEST(test_quat_rotation);
    RUN_TEST(test_quat_multiply);
    RUN_TEST(test_quat_inverse);
    RUN_TEST(test_quat_slerp);
    RUN_TEST(test_quat_euler);

    // Mat3
    std::printf("── Mat3 ─────────────────────────────\n");
    RUN_TEST(test_mat3_identity);
    RUN_TEST(test_mat3_multiply);
    RUN_TEST(test_mat3_transpose);
    RUN_TEST(test_mat3_determinant);
    RUN_TEST(test_mat3_inverse);
    RUN_TEST(test_mat3_from_quat);

    // Mat4
    std::printf("── Mat4 ─────────────────────────────\n");
    RUN_TEST(test_mat4_identity);
    RUN_TEST(test_mat4_translation);
    RUN_TEST(test_mat4_multiply);
    RUN_TEST(test_mat4_transpose);
    RUN_TEST(test_mat4_inverse);
    RUN_TEST(test_mat4_affine_inverse);

    // Transform
    std::printf("── Transform ────────────────────────\n");
    RUN_TEST(test_transform_point);
    RUN_TEST(test_transform_inverse);
    RUN_TEST(test_transform_compose);

    // AABB
    std::printf("── AABB ─────────────────────────────\n");
    RUN_TEST(test_aabb_overlap);
    RUN_TEST(test_aabb_contains);
    RUN_TEST(test_aabb_point_containment);
    RUN_TEST(test_aabb_merge);
    RUN_TEST(test_aabb_surface_area);
    RUN_TEST(test_aabb_ray_intersect);

    // Plane
    std::printf("── Plane ────────────────────────────\n");
    RUN_TEST(test_plane_distance);
    RUN_TEST(test_plane_classify);
    RUN_TEST(test_plane_project);

    // Ray
    std::printf("── Ray ──────────────────────────────\n");
    RUN_TEST(test_ray_point_at);
    RUN_TEST(test_ray_sphere_intersect);
    RUN_TEST(test_ray_triangle_intersect);
    RUN_TEST(test_ray_closest_point);

    // Utility
    std::printf("── Utility ──────────────────────────\n");
    RUN_TEST(test_math_fast_sqrt);
    RUN_TEST(test_math_fast_inv_sqrt);
    RUN_TEST(test_orthonormal_basis);

    // Summary
    std::printf("\n══════════════════════════════════════\n");
    std::printf("  Total:  %d\n", g_totalTests);
    std::printf("  Passed: %d\n", g_passedTests);
    std::printf("  Failed: %d\n", g_failedTests);
    std::printf("══════════════════════════════════════\n");

    return g_failedTests > 0 ? 1 : 0;
}
