/**
 * @file shapes_bench.cpp
 * @brief Performance benchmarks for the Pulse collision shapes module.
 *
 * Measures throughput of key shape operations: AABB computation, support function,
 * point containment, ray intersection, and mass property computation.
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
#include <chrono>

using namespace pulse;

// ── Benchmark utilities ───────────────────────────────────────────────────────

static constexpr int WARMUP_ITERS = 1000;
static constexpr int BENCH_ITERS  = 1000000;

struct BenchResult {
    double totalMs;
    double opsPerSec;
};

template <typename Func>
BenchResult benchmark(const char* name, int iters, Func&& func) {
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        func(i);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        func(i);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double opsPerSec = static_cast<double>(iters) / (ms * 0.001);

    std::printf("  %-40s %10.3f ms  %12.0f ops/s\n", name, ms, opsPerSec);
    return { ms, opsPerSec };
}

// Volatile sink to prevent optimizer from eliminating dead code
static volatile float g_sink = 0.0f;
static volatile bool  g_boolSink = false;

PULSE_FORCE_INLINE void sink(float v) { g_sink = v; }
PULSE_FORCE_INLINE void sink(Vec3 v) { g_sink = v.getX(); }
PULSE_FORCE_INLINE void sink(AABB a) { g_sink = a.min.getX(); }
PULSE_FORCE_INLINE void sink(bool b) { g_boolSink = b; }
PULSE_FORCE_INLINE void sink(MassProperties m) { g_sink = m.mass; }
PULSE_FORCE_INLINE void sink(ShapeRayResult r) { g_sink = r.t; }

// ── Benchmark groups ──────────────────────────────────────────────────────────

void bench_sphere() {
    std::printf("\n── Sphere ──\n");
    Sphere s(2.0f);
    Transform tx(Vec3(1.0f, 2.0f, 3.0f), Quat::fromAxisAngle(Vec3::unitY(), 0.5f));
    Vec3 dir(1.0f, 1.0f, 0.0f);
    Vec3 origin(0.0f, 0.0f, -5.0f);
    Vec3 rayDir(0.0f, 0.0f, 1.0f);

    benchmark("computeAABB", BENCH_ITERS, [&](int) { sink(s.computeAABB(tx)); });
    benchmark("support", BENCH_ITERS, [&](int) { sink(s.support(dir)); });
    benchmark("containsPoint", BENCH_ITERS, [&](int) { sink(s.containsPoint(dir)); });
    benchmark("closestPoint", BENCH_ITERS, [&](int) { sink(s.closestPoint(dir)); });
    benchmark("rayIntersect", BENCH_ITERS, [&](int) { sink(s.rayIntersect(origin, rayDir)); });
    benchmark("computeMass", BENCH_ITERS, [&](int) { sink(s.computeMass(1.0f)); });
}

void bench_box() {
    std::printf("\n── Box ──\n");
    Box b(1.0f, 2.0f, 3.0f);
    Transform tx(Vec3(1.0f, 2.0f, 3.0f), Quat::fromAxisAngle(Vec3(1.0f, 1.0f, 0.0f).normalized(), 0.7f));
    Vec3 dir(1.0f, -1.0f, 0.5f);
    Vec3 origin(-5.0f, 0.0f, 0.0f);
    Vec3 rayDir(1.0f, 0.0f, 0.0f);

    benchmark("computeAABB (rotated)", BENCH_ITERS, [&](int) { sink(b.computeAABB(tx)); });
    benchmark("support", BENCH_ITERS, [&](int) { sink(b.support(dir)); });
    benchmark("supportWorld", BENCH_ITERS, [&](int) { sink(b.supportWorld(dir, tx)); });
    benchmark("containsPoint", BENCH_ITERS, [&](int) { sink(b.containsPoint(dir)); });
    benchmark("rayIntersect", BENCH_ITERS, [&](int) { sink(b.rayIntersect(origin, rayDir)); });
    benchmark("computeMass", BENCH_ITERS, [&](int) { sink(b.computeMass(1.0f)); });
}

void bench_capsule() {
    std::printf("\n── Capsule ──\n");
    Capsule c(1.0f, 2.0f);
    Transform tx(Vec3(1.0f, 2.0f, 3.0f));
    Vec3 dir(0.0f, 1.0f, 0.0f);

    benchmark("computeAABB", BENCH_ITERS, [&](int) { sink(c.computeAABB(tx)); });
    benchmark("support", BENCH_ITERS, [&](int) { sink(c.support(dir)); });
    benchmark("containsPoint", BENCH_ITERS, [&](int) { sink(c.containsPoint(dir)); });
    benchmark("computeMass", BENCH_ITERS, [&](int) { sink(c.computeMass(1.0f)); });
}

void bench_cylinder() {
    std::printf("\n── Cylinder ──\n");
    Cylinder cy(1.0f, 2.0f);
    Transform tx(Vec3(1.0f, 2.0f, 3.0f));
    Vec3 dir(1.0f, 1.0f, 0.0f);
    Vec3 origin(-5.0f, 0.0f, 0.0f);
    Vec3 rayDir(1.0f, 0.0f, 0.0f);

    benchmark("computeAABB", BENCH_ITERS, [&](int) { sink(cy.computeAABB(tx)); });
    benchmark("support", BENCH_ITERS, [&](int) { sink(cy.support(dir)); });
    benchmark("containsPoint", BENCH_ITERS, [&](int) { sink(cy.containsPoint(dir)); });
    benchmark("rayIntersect", BENCH_ITERS, [&](int) { sink(cy.rayIntersect(origin, rayDir)); });
    benchmark("computeMass", BENCH_ITERS, [&](int) { sink(cy.computeMass(1.0f)); });
}

void bench_convex_hull() {
    std::printf("\n── ConvexHull ──\n");

    // 8-vertex cube
    Vec3 cubeVerts[8] = {
        Vec3(-1, -1, -1), Vec3(1, -1, -1), Vec3(-1, 1, -1), Vec3(1, 1, -1),
        Vec3(-1, -1,  1), Vec3(1, -1,  1), Vec3(-1, 1,  1), Vec3(1, 1,  1),
    };
    ConvexHull hull8(cubeVerts, 8);
    Transform tx;
    Vec3 dir(1.0f, 1.0f, 1.0f);

    benchmark("support (8 verts)", BENCH_ITERS, [&](int) { sink(hull8.support(dir)); });
    benchmark("computeAABB (8 verts)", BENCH_ITERS, [&](int) { sink(hull8.computeAABB(tx)); });
    benchmark("containsPoint (8 verts)", BENCH_ITERS / 10, [&](int) { sink(hull8.containsPoint(Vec3(0.5f, 0.5f, 0.5f))); });

    // 64-vertex sphere approximation
    Vec3 sphere64[64];
    for (int i = 0; i < 64; ++i) {
        float phi = math::Pi * static_cast<float>(i) / 63.0f;
        float theta = 2.0f * math::Pi * static_cast<float>(i * 7 % 64) / 64.0f;
        sphere64[i] = Vec3(
            std::sin(phi) * std::cos(theta),
            std::sin(phi) * std::sin(theta),
            std::cos(phi)
        );
    }
    ConvexHull hull64(sphere64, 64);

    benchmark("support (64 verts)", BENCH_ITERS, [&](int) { sink(hull64.support(dir)); });
    benchmark("computeAABB (64 verts)", BENCH_ITERS / 10, [&](int) { sink(hull64.computeAABB(tx)); });
}

void bench_tri_mesh() {
    std::printf("\n── TriMesh ──\n");

    // Create a simple grid mesh (10x10 = 200 triangles)
    constexpr int gridSize = 11;
    constexpr int numVerts = gridSize * gridSize;
    constexpr int numTris = (gridSize - 1) * (gridSize - 1) * 2;

    Vec3 meshVerts[numVerts];
    uint32_t meshIndices[numTris * 3];

    for (int z = 0; z < gridSize; ++z) {
        for (int x = 0; x < gridSize; ++x) {
            meshVerts[z * gridSize + x] = Vec3(
                static_cast<float>(x) - 5.0f, 0.0f,
                static_cast<float>(z) - 5.0f
            );
        }
    }

    int idx = 0;
    for (int z = 0; z < gridSize - 1; ++z) {
        for (int x = 0; x < gridSize - 1; ++x) {
            int base = z * gridSize + x;
            meshIndices[idx++] = static_cast<uint32_t>(base);
            meshIndices[idx++] = static_cast<uint32_t>(base + 1);
            meshIndices[idx++] = static_cast<uint32_t>(base + gridSize + 1);

            meshIndices[idx++] = static_cast<uint32_t>(base);
            meshIndices[idx++] = static_cast<uint32_t>(base + gridSize + 1);
            meshIndices[idx++] = static_cast<uint32_t>(base + gridSize);
        }
    }

    TriMesh mesh(meshVerts, numVerts, meshIndices, numTris);

    Vec3 origin(0.0f, 5.0f, 0.0f);
    Vec3 rayDir(0.0f, -1.0f, 0.0f);

    benchmark("rayIntersect (200 tris)", BENCH_ITERS / 100, [&](int) {
        sink(mesh.rayIntersect(origin, rayDir));
    });

    benchmark("closestPoint (200 tris)", BENCH_ITERS / 100, [&](int) {
        sink(mesh.closestPoint(Vec3(0.5f, 1.0f, 0.5f)));
    });

    benchmark("computeLocalAABB (121 verts)", BENCH_ITERS / 10, [&](int) {
        sink(mesh.computeLocalAABB());
    });
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    std::printf("═══════════════════════════════════════════════════════\n");
    std::printf("  Pulse Physics Engine — Shapes Module Benchmarks\n");
    std::printf("═══════════════════════════════════════════════════════\n");

    bench_sphere();
    bench_box();
    bench_capsule();
    bench_cylinder();
    bench_convex_hull();
    bench_tri_mesh();

    std::printf("\n═══════════════════════════════════════════════════════\n");
    std::printf("  Benchmarks complete.\n");
    std::printf("═══════════════════════════════════════════════════════\n");

    return 0;
}
