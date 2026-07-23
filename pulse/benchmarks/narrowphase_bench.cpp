/**
 * @file narrowphase_bench.cpp
 * @brief Performance benchmarks for the Pulse narrowphase module.
 *
 * Measures throughput of specialized and generic collision routines.
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
#include <chrono>
#include <cmath>

using namespace pulse;

// ── Benchmark timing ─────────────────────────────────────────────────────────

static double benchmarkMs(auto&& fn, uint32_t iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < iterations; ++i) {
        fn(i);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

#define BENCH(name, iterations, fn) \
    do { \
        double ms = benchmarkMs(fn, iterations); \
        double nsPerOp = (ms * 1000000.0) / static_cast<double>(iterations); \
        double opsPerSec = static_cast<double>(iterations) / (ms / 1000.0); \
        std::printf("  %-35s %8u ops  %8.2f ms  %8.1f ns/op  %10.0f ops/s\n", \
                    name, iterations, ms, nsPerOp, opsPerSec); \
    } while(0)

// ── Simple pseudo-random ─────────────────────────────────────────────────────

static float randf(uint32_t& seed) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return static_cast<float>(seed & 0xFFFF) / 65535.0f;
}

// ═════════════════════════════════════════════════════════════════════════════
// Benchmarks
// ═════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("=== Pulse NarrowPhase Benchmarks ===\n\n");

    volatile float sink = 0.0f; // Prevent optimization

    // ── 1. Sphere-Sphere batch ──
    std::printf("--- Sphere-Sphere (Analytic) ---\n");
    {
        Sphere s(1.0f);
        BENCH("sphere_sphere_overlapping", 100000, [&](uint32_t i) {
            float offset = 1.5f + 0.001f * static_cast<float>(i % 100);
            Transform txA(Vec3(0,0,0)), txB(Vec3(offset, 0, 0));
            ContactManifold m;
            bool hit = collide(s, txA, s, txB, m);
            sink += hit ? m.points[0].penetration : 0.0f;
        });

        BENCH("sphere_sphere_separated", 100000, [&](uint32_t i) {
            float offset = 3.0f + 0.001f * static_cast<float>(i % 100);
            Transform txA(Vec3(0,0,0)), txB(Vec3(offset, 0, 0));
            ContactManifold m;
            bool hit = collide(s, txA, s, txB, m);
            sink += hit ? 1.0f : 0.0f;
        });
    }

    // ── 2. Sphere-Box batch ──
    std::printf("\n--- Sphere-Box (Specialized) ---\n");
    {
        Sphere s(0.5f);
        Box b(1.0f, 1.0f, 1.0f);
        BENCH("sphere_box_overlapping", 100000, [&](uint32_t i) {
            float offset = 1.2f + 0.001f * static_cast<float>(i % 100);
            Transform txS(Vec3(offset, 0, 0)), txB;
            ContactManifold m;
            bool hit = collide(s, txS, b, txB, m);
            sink += hit ? m.points[0].penetration : 0.0f;
        });
    }

    // ── 3. Box-Box SAT batch ──
    std::printf("\n--- Box-Box (SAT) ---\n");
    {
        Box b(1.0f, 1.0f, 1.0f);
        BENCH("box_box_aligned", 50000, [&](uint32_t i) {
            float offset = 1.5f + 0.001f * static_cast<float>(i % 100);
            Transform txA, txB(Vec3(offset, 0, 0));
            ContactManifold m;
            bool hit = collide(b, txA, b, txB, m);
            sink += hit ? m.points[0].penetration : 0.0f;
        });

        BENCH("box_box_rotated", 50000, [&](uint32_t i) {
            float angle = 0.1f * static_cast<float>(i % 30);
            Quat rot = Quat::fromAxisAngle(Vec3(0,1,0), angle);
            Transform txA, txB(Vec3(1.5f, 0, 0), rot);
            ContactManifold m;
            bool hit = collide(b, txA, b, txB, m);
            sink += hit ? 1.0f : 0.0f;
        });

        BENCH("box_box_separated", 50000, [&](uint32_t i) {
            float offset = 3.0f + 0.001f * static_cast<float>(i % 100);
            Transform txA, txB(Vec3(offset, 0, 0));
            ContactManifold m;
            bool hit = collide(b, txA, b, txB, m);
            sink += hit ? 1.0f : 0.0f;
        });
    }

    // ── 4. Capsule-Capsule batch ──
    std::printf("\n--- Capsule-Capsule (Specialized) ---\n");
    {
        Capsule c(0.3f, 1.0f);
        BENCH("capsule_capsule_parallel", 100000, [&](uint32_t i) {
            float offset = 0.4f + 0.001f * static_cast<float>(i % 100);
            Transform txA, txB(Vec3(offset, 0, 0));
            ContactManifold m;
            bool hit = collide(c, txA, c, txB, m);
            sink += hit ? m.points[0].penetration : 0.0f;
        });
    }

    // ── 5. GJK ConvexHull batch ──
    std::printf("\n--- GJK (ConvexHull) ---\n");
    {
        Vec3 vertsA[8], vertsB[8];
        Box(1.0f, 1.0f, 1.0f).getVertices(vertsA);
        Box(1.0f, 1.0f, 1.0f).getVertices(vertsB);
        ConvexHull hullA(vertsA, 8);
        ConvexHull hullB(vertsB, 8);

        BENCH("gjk_hull_overlapping", 10000, [&](uint32_t i) {
            float offset = 1.0f + 0.001f * static_cast<float>(i % 100);
            Transform txA, txB(Vec3(offset, 0, 0));
            GjkResult result;
            gjkQuery(hullA, txA, hullB, txB, result);
            sink += result.distance;
        });

        BENCH("gjk_hull_separated", 10000, [&](uint32_t i) {
            float offset = 3.0f + 0.001f * static_cast<float>(i % 100);
            Transform txA, txB(Vec3(offset, 0, 0));
            GjkResult result;
            gjkQuery(hullA, txA, hullB, txB, result);
            sink += result.distance;
        });
    }

    // ── 6. MPR batch ──
    std::printf("\n--- MPR ---\n");
    {
        Sphere a(1.0f), b(1.0f);
        BENCH("mpr_sphere_overlapping", 50000, [&](uint32_t i) {
            float offset = 1.0f + 0.001f * static_cast<float>(i % 100);
            Transform txA(Vec3(0,0,0)), txB(Vec3(offset, 0, 0));
            MprResult result;
            mprQuery(a, txA, b, txB, result);
            sink += result.penetration;
        });

        Box ba(1.0f, 1.0f, 1.0f), bb(1.0f, 1.0f, 1.0f);
        BENCH("mpr_box_overlapping", 50000, [&](uint32_t i) {
            float offset = 1.5f + 0.001f * static_cast<float>(i % 100);
            Transform txA, txB(Vec3(offset, 0, 0));
            MprResult result;
            mprQuery(ba, txA, bb, txB, result);
            sink += result.penetration;
        });
    }

    // ── 7. CCD batch ──
    std::printf("\n--- CCD (Conservative Advancement) ---\n");
    {
        Sphere a(0.5f), b(0.5f);
        BENCH("ccd_sphere_linear", 5000, [&](uint32_t i) {
            float speed = 5.0f + 0.01f * static_cast<float>(i % 100);
            Transform txA_start(Vec3(-speed, 0, 0)), txA_end(Vec3(speed, 0, 0));
            Transform txB_start(Vec3(2, 0, 0)), txB_end(Vec3(2, 0, 0));
            CcdResult result;
            ccdQuery(a, txA_start, txA_end, b, txB_start, txB_end, result);
            sink += result.timeOfImpact;
        });

        BENCH("ccd_sphere_no_hit", 5000, [&](uint32_t i) {
            float offset = 5.0f + 0.01f * static_cast<float>(i % 100);
            Transform txA_start(Vec3(-5, 3, 0)), txA_end(Vec3(5, 3, 0));
            Transform txB_start(Vec3(0, 0, 0)), txB_end(Vec3(0, 0, 0));
            CcdResult result;
            ccdQuery(a, txA_start, txA_end, b, txB_start, txB_end, result);
            sink += result.timeOfImpact;
        });
    }

    std::printf("\n=== Benchmarks Complete ===\n");
    std::printf("(sink = %.2f)\n", static_cast<double>(sink));

    return 0;
}
