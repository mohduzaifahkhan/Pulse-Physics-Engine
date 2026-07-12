/**
 * @file math_bench.cpp
 * @brief Performance benchmarks for the Pulse math library.
 *
 * Measures throughput of critical math operations (dot, cross, normalize,
 * mat4 multiply, quaternion rotate) to verify SIMD codegen is working and
 * to establish baseline performance numbers.
 */

#include <pulse/math/math_common.h>
#include <pulse/math/vec3.h>
#include <pulse/math/vec4.h>
#include <pulse/math/quat.h>
#include <pulse/math/mat3.h>
#include <pulse/math/mat4.h>
#include <pulse/math/aabb.h>

#include <cstdio>
#include <chrono>

using namespace pulse;

// Prevent compiler from optimizing away the result
template <typename T>
static void doNotOptimize(T const& val) {
    // Volatile read of the value
    volatile auto sink = val;
    (void)sink;
}

struct BenchResult {
    const char* name;
    double opsPerSecond;
    double nsPerOp;
};

template <typename Func>
BenchResult benchmark(const char* name, int iterations, Func&& func) {
    // Warm up
    for (int i = 0; i < 1000; ++i) {
        func(i);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func(i);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();
    double opsPerSec = static_cast<double>(iterations) / elapsed;
    double nsPerOp = (elapsed * 1.0e9) / static_cast<double>(iterations);

    return {name, opsPerSec, nsPerOp};
}

void printResult(const BenchResult& r) {
    std::printf("  %-30s %12.0f ops/s  |  %.2f ns/op\n", r.name, r.opsPerSecond, r.nsPerOp);
}

int main() {
    constexpr int N = 10'000'000;

    std::printf("╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║          PULSE Physics Engine - Math Benchmarks             ║\n");
    std::printf("╠══════════════════════════════════════════════════════════════╣\n");
    std::printf("║  Iterations per benchmark: %d                      ║\n", N);
    std::printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    // ── Vec3 ──────────────────────────────────────────────────────────────
    std::printf("── Vec3 ─────────────────────────────────────────────────────\n");

    Vec3 va(1.0f, 2.0f, 3.0f);
    Vec3 vb(4.0f, 5.0f, 6.0f);

    printResult(benchmark("Vec3 add", N, [&](int) {
        Vec3 r = va + vb;
        doNotOptimize(r);
    }));

    printResult(benchmark("Vec3 dot", N, [&](int) {
        float r = va.dot(vb);
        doNotOptimize(r);
    }));

    printResult(benchmark("Vec3 cross", N, [&](int) {
        Vec3 r = va.cross(vb);
        doNotOptimize(r);
    }));

    printResult(benchmark("Vec3 normalize", N, [&](int) {
        Vec3 r = va.normalized();
        doNotOptimize(r);
    }));

    printResult(benchmark("Vec3 length", N, [&](int) {
        float r = va.length();
        doNotOptimize(r);
    }));

    printResult(benchmark("Vec3 lerp", N, [&](int i) {
        float t = static_cast<float>(i % 100) * 0.01f;
        Vec3 r = va.lerp(vb, t);
        doNotOptimize(r);
    }));

    // ── Vec4 ──────────────────────────────────────────────────────────────
    std::printf("\n── Vec4 ─────────────────────────────────────────────────────\n");

    Vec4 v4a(1.0f, 2.0f, 3.0f, 4.0f);
    Vec4 v4b(5.0f, 6.0f, 7.0f, 8.0f);

    printResult(benchmark("Vec4 dot", N, [&](int) {
        float r = v4a.dot(v4b);
        doNotOptimize(r);
    }));

    printResult(benchmark("Vec4 normalize", N, [&](int) {
        Vec4 r = v4a.normalized();
        doNotOptimize(r);
    }));

    // ── Quaternion ────────────────────────────────────────────────────────
    std::printf("\n── Quaternion ───────────────────────────────────────────────\n");

    Quat qa = Quat::fromAxisAngle(Vec3::unitY(), 0.5f);
    Quat qb = Quat::fromAxisAngle(Vec3::unitZ(), 0.3f);
    Vec3 rotv(1.0f, 0.0f, 0.0f);

    printResult(benchmark("Quat multiply", N, [&](int) {
        Quat r = qa * qb;
        doNotOptimize(r);
    }));

    printResult(benchmark("Quat rotate Vec3", N, [&](int) {
        Vec3 r = qa.rotate(rotv);
        doNotOptimize(r);
    }));

    printResult(benchmark("Quat slerp", N, [&](int i) {
        float t = static_cast<float>(i % 100) * 0.01f;
        Quat r = qa.slerp(qb, t);
        doNotOptimize(r);
    }));

    printResult(benchmark("Quat nlerp", N, [&](int i) {
        float t = static_cast<float>(i % 100) * 0.01f;
        Quat r = qa.nlerp(qb, t);
        doNotOptimize(r);
    }));

    // ── Mat3 ──────────────────────────────────────────────────────────────
    std::printf("\n── Mat3 ─────────────────────────────────────────────────────\n");

    Mat3 m3a = Mat3::fromQuat(qa);
    Mat3 m3b = Mat3::fromQuat(qb);

    printResult(benchmark("Mat3 multiply", N, [&](int) {
        Mat3 r = m3a * m3b;
        doNotOptimize(r);
    }));

    printResult(benchmark("Mat3 * Vec3", N, [&](int) {
        Vec3 r = m3a * va;
        doNotOptimize(r);
    }));

    printResult(benchmark("Mat3 inverse", N, [&](int) {
        Mat3 r = m3a.inversed();
        doNotOptimize(r);
    }));

    printResult(benchmark("Mat3 transpose", N, [&](int) {
        Mat3 r = m3a.transposed();
        doNotOptimize(r);
    }));

    // ── Mat4 ──────────────────────────────────────────────────────────────
    std::printf("\n── Mat4 ─────────────────────────────────────────────────────\n");

    Mat4 m4a = Mat4::trs(Vec3(1, 2, 3), qa, Vec3(1.0f));
    Mat4 m4b = Mat4::trs(Vec3(4, 5, 6), qb, Vec3(1.0f));

    printResult(benchmark("Mat4 multiply", N, [&](int) {
        Mat4 r = m4a * m4b;
        doNotOptimize(r);
    }));

    printResult(benchmark("Mat4 * Vec4", N, [&](int) {
        Vec4 r = m4a * v4a;
        doNotOptimize(r);
    }));

    printResult(benchmark("Mat4 transpose", N, [&](int) {
        Mat4 r = m4a.transposed();
        doNotOptimize(r);
    }));

    printResult(benchmark("Mat4 inverse (full)", N / 10, [&](int) {
        Mat4 r = m4a.inversed();
        doNotOptimize(r);
    }));

    printResult(benchmark("Mat4 affine inverse", N, [&](int) {
        Mat4 r = m4a.affineInverse();
        doNotOptimize(r);
    }));

    printResult(benchmark("Mat4 transformPoint", N, [&](int) {
        Vec3 r = m4a.transformPoint(va);
        doNotOptimize(r);
    }));

    // ── AABB ──────────────────────────────────────────────────────────────
    std::printf("\n── AABB ─────────────────────────────────────────────────────\n");

    AABB aabb1(Vec3(0, 0, 0), Vec3(1, 1, 1));
    AABB aabb2(Vec3(0.5f, 0.5f, 0.5f), Vec3(1.5f, 1.5f, 1.5f));

    printResult(benchmark("AABB overlap test", N, [&](int) {
        bool r = aabb1.overlaps(aabb2);
        doNotOptimize(r);
    }));

    printResult(benchmark("AABB merge", N, [&](int) {
        AABB r = aabb1.merged(aabb2);
        doNotOptimize(r);
    }));

    printResult(benchmark("AABB contains point", N, [&](int) {
        bool r = aabb1.containsPoint(Vec3(0.5f, 0.5f, 0.5f));
        doNotOptimize(r);
    }));

    printResult(benchmark("AABB surface area", N, [&](int) {
        float r = aabb1.surfaceArea();
        doNotOptimize(r);
    }));

    std::printf("\n══════════════════════════════════════════════════════════════\n");
    std::printf("  Done.\n");

    return 0;
}
