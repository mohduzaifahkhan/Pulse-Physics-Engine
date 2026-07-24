/**
 * @file contact_bench.cpp
 * @brief Performance benchmarks for the Pulse contact module.
 *
 * Measures throughput of cache lookups, insertions, contact matching,
 * frame updates, warm starting, and memory footprint.
 */

#include <pulse/contact/contact_common.h>
#include <pulse/contact/persistent_contact.h>
#include <pulse/contact/contact_manifold_persistent.h>
#include <pulse/contact/contact_cache.h>
#include <pulse/contact/warm_start.h>

#include <pulse/narrowphase/narrowphase_common.h>
#include <pulse/math/math_common.h>

#include <cstdio>
#include <chrono>
#include <cmath>

using namespace pulse;

// ── Benchmark timing ─────────────────────────────────────────────────────────

template <typename Fn>
static double benchmarkMs(Fn&& fn, uint32_t iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < iterations; ++i) {
        fn(i);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

#define BENCH(name, iterations, ...) \
    do { \
        auto fn = __VA_ARGS__; \
        double ms = benchmarkMs(fn, iterations); \
        double nsPerOp = (ms * 1000000.0) / static_cast<double>(iterations); \
        double opsPerSec = static_cast<double>(iterations) / (ms / 1000.0); \
        std::printf("  %-40s %8u ops  %8.2f ms  %8.1f ns/op  %10.0f ops/s\n", \
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
    std::printf("=== Pulse Contact Module Benchmarks ===\n\n");

    volatile float sink = 0.0f;

    // ── 1. Cache Lookup ─────────────────────────────────────────────────

    std::printf("--- 1. Cache Lookup ---\n");
    {
        ContactCache cache(32768);
        // Pre-populate with 10K pairs
        for (uint32_t i = 0; i < 10000; ++i) {
            cache.insertOrFind(BodyPairKey(i, i + 50000), i, i + 50000);
        }

        BENCH("find_existing_10K", 100000, [&](uint32_t i) {
            uint32_t idx = i % 10000;
            BodyPairKey key(idx, idx + 50000);
            PersistentManifold* pm = cache.find(key);
            sink += (pm != nullptr) ? 1.0f : 0.0f;
        });

        BENCH("find_miss_10K", 100000, [&](uint32_t i) {
            BodyPairKey key(i + 100000, i + 200000);
            PersistentManifold* pm = cache.find(key);
            sink += (pm != nullptr) ? 1.0f : 0.0f;
        });
    }

    // ── 2. Cache Insert ─────────────────────────────────────────────────

    std::printf("\n--- 2. Cache Insert ---\n");
    {
        BENCH("insert_1K_pairs", 1000, [&](uint32_t i) {
            ContactCache cache(4096);
            for (uint32_t j = 0; j < 1000; ++j) {
                cache.insertOrFind(BodyPairKey(j + i * 1000, j + i * 1000 + 50000), j, j + 50000);
            }
            sink += static_cast<float>(cache.count());
        });

        BENCH("insert_5K_pairs", 200, [&](uint32_t i) {
            ContactCache cache(16384);
            for (uint32_t j = 0; j < 5000; ++j) {
                cache.insertOrFind(BodyPairKey(j + i * 5000, j + i * 5000 + 50000), j, j + 50000);
            }
            sink += static_cast<float>(cache.count());
        });
    }

    // ── 3. Contact Matching ─────────────────────────────────────────────

    std::printf("\n--- 3. Contact Matching ---\n");
    {
        ContactConfig config;
        uint32_t seed = 12345;

        BENCH("match_4pt_manifold", 100000, [&](uint32_t) {
            PersistentManifold pm(1, 2);

            // Create initial manifold
            ContactPoint pts[4];
            for (int k = 0; k < 4; ++k) {
                float x = randf(seed) * 2.0f - 1.0f;
                float z = randf(seed) * 2.0f - 1.0f;
                pts[k] = ContactPoint(Vec3(x, 0, z), Vec3(x, -1, z), Vec3(0, 1, 0), 0.1f);
                pts[k].featureIdA = static_cast<uint32_t>(k);
                pts[k].featureIdB = static_cast<uint32_t>(k);
            }
            ContactManifold cm1;
            for (int k = 0; k < 4; ++k) cm1.addPoint(pts[k]);
            pm.mergeContacts(cm1, config.matchDistanceSq);

            // Match with slightly perturbed contacts
            ContactManifold cm2;
            for (int k = 0; k < 4; ++k) {
                ContactPoint cp = pts[k];
                cp.positionOnA = cp.positionOnA + Vec3(0.001f, 0, 0);
                cp.penetration += 0.01f;
                cm2.addPoint(cp);
            }
            pm.mergeContacts(cm2, config.matchDistanceSq);
            sink += static_cast<float>(pm.contactCount);
        });

        BENCH("match_proximity_only", 100000, [&](uint32_t) {
            PersistentManifold pm(1, 2);
            ContactConfig cfg;

            // Contacts with no feature IDs
            ContactPoint cp1(Vec3(0,0,0), Vec3(0,-1,0), Vec3(0,1,0), 0.1f);
            ContactManifold cm1;
            cm1.addPoint(cp1);
            pm.mergeContacts(cm1, cfg.matchDistanceSq);

            ContactPoint cp2(Vec3(0.01f, 0, 0), Vec3(0.01f, -1, 0), Vec3(0, 1, 0), 0.11f);
            ContactManifold cm2;
            cm2.addPoint(cp2);
            pm.mergeContacts(cm2, cfg.matchDistanceSq);
            sink += static_cast<float>(pm.contactCount);
        });
    }

    // ── 4. Frame Update ─────────────────────────────────────────────────

    std::printf("\n--- 4. Frame Update ---\n");
    {
        ContactConfig config;

        BENCH("frame_update_1K_pairs", 100, [&](uint32_t) {
            ContactCache cache(4096);
            cache.beginFrame();
            for (uint32_t j = 0; j < 1000; ++j) {
                ContactPoint cp(Vec3(static_cast<float>(j), 0, 0),
                                Vec3(static_cast<float>(j), -1, 0),
                                Vec3(0, 1, 0), 0.1f);
                ContactManifold cm;
                cm.addPoint(cp);
                cache.processManifold(j, j + 10000, cm, config);
            }
            cache.endFrame();
            sink += static_cast<float>(cache.count());
        });

        BENCH("frame_update_5K_pairs", 20, [&](uint32_t) {
            ContactCache cache(16384);
            cache.beginFrame();
            for (uint32_t j = 0; j < 5000; ++j) {
                ContactPoint cp(Vec3(static_cast<float>(j), 0, 0),
                                Vec3(static_cast<float>(j), -1, 0),
                                Vec3(0, 1, 0), 0.1f);
                ContactManifold cm;
                cm.addPoint(cp);
                cache.processManifold(j, j + 50000, cm, config);
            }
            cache.endFrame();
            sink += static_cast<float>(cache.count());
        });
    }

    // ── 5. Warm Start ───────────────────────────────────────────────────

    std::printf("\n--- 5. Warm Start ---\n");
    {
        ContactConfig config;

        // Pre-populate cache
        ContactCache cache(16384);
        cache.beginFrame();
        for (uint32_t j = 0; j < 5000; ++j) {
            ContactPoint cp(Vec3(static_cast<float>(j), 0, 0),
                            Vec3(static_cast<float>(j), -1, 0),
                            Vec3(0, 1, 0), 0.1f);
            ContactManifold cm;
            cm.addPoint(cp);
            cache.processManifold(j, j + 50000, cm, config);
        }

        BENCH("warm_start_5K_manifolds", 1000, [&](uint32_t) {
            warmStartAllManifolds(cache, config);
            sink += cache.loadFactor();
        });
    }

    // ── 6. Memory Footprint ─────────────────────────────────────────────

    std::printf("\n--- 6. Memory Footprint ---\n");
    {
        std::printf("  sizeof(BodyPairKey)         = %zu bytes\n", sizeof(BodyPairKey));
        std::printf("  sizeof(PersistentContact)   = %zu bytes\n", sizeof(PersistentContact));
        std::printf("  sizeof(PersistentManifold)  = %zu bytes\n", sizeof(PersistentManifold));
        std::printf("  sizeof(ContactConfig)       = %zu bytes\n", sizeof(ContactConfig));

        uint32_t capacity = 16384;
        size_t cacheBytes = capacity * (sizeof(BodyPairKey) + sizeof(PersistentManifold) + sizeof(bool));
        std::printf("  ContactCache (cap=%u)       = %zu bytes (%.1f KB)\n",
                    capacity, cacheBytes, static_cast<double>(cacheBytes) / 1024.0);
        std::printf("  Per-pair overhead            = %zu bytes\n",
                    sizeof(BodyPairKey) + sizeof(PersistentManifold) + sizeof(bool));
    }

    std::printf("\n=== Benchmarks complete ===\n");
    (void)sink;
    return 0;
}
