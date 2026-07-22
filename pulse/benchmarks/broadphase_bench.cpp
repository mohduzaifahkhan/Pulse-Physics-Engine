/**
 * @file broadphase_bench.cpp
 * @brief Performance benchmarks for the Pulse broadphase module.
 */

#include <pulse/broadphase/broadphase_common.h>
#include <pulse/broadphase/dynamic_aabb_tree.h>
#include <pulse/broadphase/sap.h>
#include <pulse/broadphase/uniform_grid.h>
#include <pulse/broadphase/bvh.h>

#include <pulse/math/math_common.h>

#include <cstdio>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>

using namespace pulse;

// ── Benchmark utilities ───────────────────────────────────────────────────────

static constexpr int WARMUP = 3;

template <typename Func>
void benchmark(const char* name, int iters, Func&& func) {
    for (int i = 0; i < WARMUP; ++i) func(i);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) func(i);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ops = static_cast<double>(iters) / (ms * 0.001);
    std::printf("  %-48s %10.3f ms  %12.0f ops/s\n", name, ms, ops);
}

static volatile float g_fSink = 0.0f;
static volatile uint32_t g_uSink = 0;

static AABB makeAABB(float x, float y, float z, float half) {
    return AABB(Vec3(x-half, y-half, z-half), Vec3(x+half, y+half, z+half));
}

// ── Broadphase scene setups ───────────────────────────────────────────────────

// Grid-spaced proxy positions
static void makeGridPositions(float* xs, float* ys, float* zs, uint32_t n, float spacing = 5.0f) {
    uint32_t side = static_cast<uint32_t>(std::ceil(std::cbrt(static_cast<double>(n))));
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t x = i % side;
        uint32_t y = (i / side) % side;
        uint32_t z = i / (side * side);
        xs[i] = static_cast<float>(x) * spacing;
        ys[i] = static_cast<float>(y) * spacing;
        zs[i] = static_cast<float>(z) * spacing;
    }
}

// ── DynamicAABBTree benchmarks ────────────────────────────────────────────────

void bench_dynamic_aabb_tree() {
    std::printf("\n── DynamicAABBTree ──\n");

    static constexpr uint32_t N64   = 64;
    static constexpr uint32_t N256  = 256;
    static constexpr uint32_t N1024 = 1024;

    // Shared positions
    float xs[N1024], ys[N1024], zs[N1024];
    makeGridPositions(xs, ys, zs, N1024, 5.0f);

    // --- Insert / destroy 64 proxies ---
    benchmark("Insert 64 proxies", 10000, [&](int) {
        DynamicAABBTree tree(128);
        for (uint32_t i = 0; i < N64; ++i) {
            tree.createProxy(makeAABB(xs[i], ys[i], zs[i], 1), i);
        }
        g_uSink = tree.proxyCount();
    });

    // --- Insert / destroy 256 proxies ---
    benchmark("Insert 256 proxies", 1000, [&](int) {
        DynamicAABBTree tree(512);
        for (uint32_t i = 0; i < N256; ++i) {
            tree.createProxy(makeAABB(xs[i], ys[i], zs[i], 1), i);
        }
        g_uSink = tree.proxyCount();
    });

    // --- Insert 1024 proxies ---
    benchmark("Insert 1024 proxies", 100, [&](int) {
        DynamicAABBTree tree(2048);
        for (uint32_t i = 0; i < N1024; ++i) {
            tree.createProxy(makeAABB(xs[i], ys[i], zs[i], 1), i);
        }
        g_uSink = tree.proxyCount();
    });

    // --- Persistent tree: moveProxy 256 proxies ---
    {
        DynamicAABBTree tree(512);
        ProxyHandle handles[N256];
        for (uint32_t i = 0; i < N256; ++i) {
            handles[i] = tree.createProxy(makeAABB(xs[i], ys[i], zs[i], 1), i);
        }

        benchmark("moveProxy 256 (small delta)", 10000, [&](int k) {
            float delta = (k % 2 == 0) ? 0.05f : -0.05f;
            for (uint32_t i = 0; i < N256; ++i) {
                tree.moveProxy(handles[i], makeAABB(xs[i] + delta, ys[i], zs[i], 1));
            }
        });
    }

    // --- AABB query 256 proxies ---
    {
        DynamicAABBTree tree(512);
        for (uint32_t i = 0; i < N256; ++i) {
            tree.createProxy(makeAABB(xs[i], ys[i], zs[i], 1), i);
        }
        AABB query = makeAABB(12, 12, 12, 10);

        benchmark("queryAABB (256 proxies)", 100000, [&](int) {
            ProxyHandle hits[64];
            g_uSink = tree.queryAABB(query, hits, 64);
        });
    }

    // --- computePairs 64 proxies ---
    {
        DynamicAABBTree tree(128);
        for (uint32_t i = 0; i < N64; ++i) {
            tree.createProxy(makeAABB(xs[i], ys[i], zs[i], 1), i);
        }

        benchmark("computePairs 64 proxies", 10000, [&](int) {
            OverlapPair pairs[512];
            g_uSink = tree.computePairs(pairs, 512);
        });
    }

    // --- computePairs 256 proxies ---
    {
        DynamicAABBTree tree(512);
        for (uint32_t i = 0; i < N256; ++i) {
            tree.createProxy(makeAABB(xs[i], ys[i], zs[i], 1), i);
        }

        benchmark("computePairs 256 proxies", 1000, [&](int) {
            OverlapPair pairs[4096];
            g_uSink = tree.computePairs(pairs, 4096);
        });
    }

    // --- Raycast 256 proxies ---
    {
        DynamicAABBTree tree(512);
        for (uint32_t i = 0; i < N256; ++i) {
            tree.createProxy(makeAABB(xs[i], ys[i], zs[i], 1), i);
        }

        benchmark("raycast (256 proxies)", 100000, [&](int) {
            ProxyHandle hits[32];
            g_uSink = tree.raycast(Vec3(0,0,-10), Vec3(0,0,1), 1000, hits, 32);
        });
    }
}

// ── SAP benchmarks ────────────────────────────────────────────────────────────

void bench_sap() {
    std::printf("\n── SAP ──\n");

    float xs[256], ys[256], zs[256];
    makeGridPositions(xs, ys, zs, 256, 5.0f);

    // computePairs 64
    {
        SAP sap;
        for (uint32_t i = 0; i < 64; ++i) {
            sap.addProxy(makeAABB(xs[i], ys[i], zs[i], 1), nullptr);
        }
        benchmark("computePairs 64 proxies", 10000, [&](int) {
            OverlapPair pairs[512];
            g_uSink = sap.computePairs(pairs, 512);
        });
    }

    // computePairs 256
    {
        SAP sap;
        for (uint32_t i = 0; i < 256; ++i) {
            sap.addProxy(makeAABB(xs[i], ys[i], zs[i], 1), nullptr);
        }
        benchmark("computePairs 256 proxies", 1000, [&](int) {
            OverlapPair pairs[4096];
            g_uSink = sap.computePairs(pairs, 4096);
        });
    }
}

// ── UniformGrid benchmarks ────────────────────────────────────────────────────

void bench_uniform_grid() {
    std::printf("\n── UniformGrid ──\n");

    static constexpr uint32_t N = 256;
    float xs[N], ys[N], zs[N];
    makeGridPositions(xs, ys, zs, N, 3.0f);

    AABB aabbs[N];
    for (uint32_t i = 0; i < N; ++i) {
        aabbs[i] = makeAABB(xs[i], ys[i], zs[i], 1);
    }

    benchmark("beginFrame + insert 256", 10000, [&](int) {
        UniformGrid grid(3.0f, 1024, 8192);
        grid.beginFrame();
        for (uint32_t i = 0; i < N; ++i) {
            grid.insert(ProxyHandle(i), aabbs[i]);
        }
        g_uSink = grid.poolCount();
    });

    {
        UniformGrid grid(3.0f, 1024, 8192);
        grid.beginFrame();
        for (uint32_t i = 0; i < N; ++i) {
            grid.insert(ProxyHandle(i), aabbs[i]);
        }

        benchmark("computePairs 256 proxies", 1000, [&](int) {
            OverlapPair pairs[4096];
            g_uSink = grid.computePairs(aabbs, N, pairs, 4096);
        });
    }
}

// ── BVH benchmarks ───────────────────────────────────────────────────────────

void bench_bvh() {
    std::printf("\n── BVH ──\n");

    // Build benchmarks
    static constexpr uint32_t N100  = 100;
    static constexpr uint32_t N1000 = 1000;

    AABB leaves100[N100];
    for (uint32_t i = 0; i < N100; ++i) {
        float x = static_cast<float>(i % 10) * 3;
        float y = static_cast<float>(i / 10) * 3;
        leaves100[i] = makeAABB(x, y, 0, 1);
    }

    benchmark("build 100 leaves", 10000, [&](int) {
        BVH bvh;
        bvh.build(leaves100, N100);
        g_uSink = bvh.nodeCount();
    });

    // Build 1000 leaves
    AABB* leaves1000 = static_cast<AABB*>(std::malloc(N1000 * sizeof(AABB)));
    for (uint32_t i = 0; i < N1000; ++i) {
        float x = static_cast<float>(i % 32) * 3;
        float y = static_cast<float>(i / 32) * 3;
        leaves1000[i] = makeAABB(x, y, 0, 1);
    }

    benchmark("build 1000 leaves", 1000, [&](int) {
        BVH bvh;
        bvh.build(leaves1000, N1000);
        g_uSink = bvh.nodeCount();
    });

    // Query benchmarks
    {
        BVH bvh;
        bvh.build(leaves1000, N1000);

        benchmark("queryAABB (1000 leaves)", 100000, [&](int) {
            uint32_t results[64];
            g_uSink = bvh.queryAABB(makeAABB(45, 12, 0, 5), results, 64);
        });

        Vec3 origin(45, 12, -10);
        Vec3 dir(0, 0, 1);
        Vec3 invDir(1.0f/dir.getX(), 1.0f/dir.getY(), 1.0f/dir.getZ());

        benchmark("raycast (1000 leaves)", 100000, [&](int) {
            uint32_t results[32];
            g_uSink = bvh.raycast(origin, invDir, 100.0f, results, 32);
        });
    }

    std::free(leaves1000);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    std::printf("═══════════════════════════════════════════════════════\n");
    std::printf("  Pulse Physics Engine — BroadPhase Module Benchmarks\n");
    std::printf("═══════════════════════════════════════════════════════\n");

    bench_dynamic_aabb_tree();
    bench_sap();
    bench_uniform_grid();
    bench_bvh();

    std::printf("\n═══════════════════════════════════════════════════════\n");
    std::printf("  Benchmarks complete.\n");
    std::printf("═══════════════════════════════════════════════════════\n");
    return 0;
}
