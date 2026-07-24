/**
 * @file test_contact.cpp
 * @brief Comprehensive unit tests for the Pulse contact module.
 *
 * Tests BodyPairKey, PersistentContact, PersistentManifold, ContactCache,
 * warm starting, the full frame-update pipeline, edge cases, and
 * multi-frame simulation scenarios.
 */

#include <pulse/contact/contact_common.h>
#include <pulse/contact/persistent_contact.h>
#include <pulse/contact/contact_manifold_persistent.h>
#include <pulse/contact/contact_cache.h>
#include <pulse/contact/warm_start.h>

#include <pulse/narrowphase/narrowphase_common.h>
#include <pulse/math/math_common.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <utility>

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

/// Create a simple contact point for testing.
static ContactPoint makeContact(Vec3 posA, Vec3 posB, Vec3 normal, float depth,
                                uint32_t fidA = 0xFFFFFFFFu, uint32_t fidB = 0xFFFFFFFFu) {
    ContactPoint cp(posA, posB, normal, depth);
    cp.featureIdA = fidA;
    cp.featureIdB = fidB;
    return cp;
}

/// Create a ContactManifold with the given contacts.
static ContactManifold makeManifold(const ContactPoint* points, uint32_t count) {
    ContactManifold m;
    for (uint32_t i = 0; i < count; ++i) {
        m.addPoint(points[i]);
    }
    return m;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 1: BodyPairKey
// ═════════════════════════════════════════════════════════════════════════════

static bool test_bodypairkey_canonical_order() {
    BodyPairKey k1(10, 20);
    BodyPairKey k2(20, 10);
    TEST_ASSERT(k1 == k2);
    TEST_ASSERT(k1.low == 10);
    TEST_ASSERT(k1.high == 20);
    TEST_ASSERT(k2.low == 10);
    TEST_ASSERT(k2.high == 20);
    return true;
}

static bool test_bodypairkey_same_id() {
    BodyPairKey k(5, 5);
    TEST_ASSERT(k.low == 5);
    TEST_ASSERT(k.high == 5);
    TEST_ASSERT(k.isValid());
    return true;
}

static bool test_bodypairkey_hash_consistency() {
    BodyPairKey k1(100, 200);
    BodyPairKey k2(200, 100);
    TEST_ASSERT(k1.hash() == k2.hash());
    return true;
}

static bool test_bodypairkey_hash_uniqueness() {
    BodyPairKey k1(1, 2);
    BodyPairKey k2(1, 3);
    BodyPairKey k3(2, 3);
    // Hashes should be different (not guaranteed, but likely for small inputs)
    TEST_ASSERT(k1.hash() != k2.hash() || k1.hash() != k3.hash());
    return true;
}

static bool test_bodypairkey_packed() {
    BodyPairKey k(3, 7);
    uint64_t packed = k.packed();
    TEST_ASSERT((packed & 0xFFFFFFFFu) == 3);   // low
    TEST_ASSERT((packed >> 32) == 7);            // high
    return true;
}

static bool test_bodypairkey_invalid_default() {
    BodyPairKey k;
    TEST_ASSERT(!k.isValid());
    return true;
}

static bool test_bodypairkey_valid() {
    BodyPairKey k(0, 0);
    TEST_ASSERT(k.isValid());
    return true;
}

static bool test_bodypairkey_less_than() {
    BodyPairKey k1(1, 2);
    BodyPairKey k2(1, 3);
    TEST_ASSERT(k1 < k2);
    TEST_ASSERT(!(k2 < k1));
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 2: ContactFlags
// ═════════════════════════════════════════════════════════════════════════════

static bool test_flags_none() {
    ContactFlags f = ContactFlags::None;
    TEST_ASSERT(!hasFlag(f, ContactFlags::New));
    TEST_ASSERT(!hasFlag(f, ContactFlags::Persisted));
    return true;
}

static bool test_flags_set_and_test() {
    ContactFlags f = ContactFlags::New | ContactFlags::HasImpulse;
    TEST_ASSERT(hasFlag(f, ContactFlags::New));
    TEST_ASSERT(hasFlag(f, ContactFlags::HasImpulse));
    TEST_ASSERT(!hasFlag(f, ContactFlags::Persisted));
    TEST_ASSERT(!hasFlag(f, ContactFlags::Removed));
    return true;
}

static bool test_flags_combine() {
    ContactFlags f = ContactFlags::None;
    f |= ContactFlags::Persisted;
    f |= ContactFlags::HasImpulse;
    TEST_ASSERT(hasFlag(f, ContactFlags::Persisted));
    TEST_ASSERT(hasFlag(f, ContactFlags::HasImpulse));
    return true;
}

static bool test_flags_mask() {
    ContactFlags f = ContactFlags::New | ContactFlags::Persisted | ContactFlags::HasImpulse;
    f &= ~ContactFlags::Persisted;
    TEST_ASSERT(hasFlag(f, ContactFlags::New));
    TEST_ASSERT(!hasFlag(f, ContactFlags::Persisted));
    TEST_ASSERT(hasFlag(f, ContactFlags::HasImpulse));
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 3: PersistentContact
// ═════════════════════════════════════════════════════════════════════════════

static bool test_persistent_contact_default() {
    PersistentContact pc;
    TEST_ASSERT(approx(pc.normalImpulse, 0.0f));
    TEST_ASSERT(approx(pc.tangentImpulse0, 0.0f));
    TEST_ASSERT(approx(pc.tangentImpulse1, 0.0f));
    TEST_ASSERT(pc.contactAge == 0);
    TEST_ASSERT(hasFlag(pc.flags, ContactFlags::New));
    return true;
}

static bool test_persistent_contact_from_cp() {
    ContactPoint cp(Vec3(1,0,0), Vec3(0,0,0), Vec3(1,0,0), 0.5f);
    PersistentContact pc(cp);
    TEST_ASSERT(approxVec(pc.positionOnA, Vec3(1,0,0)));
    TEST_ASSERT(approxVec(pc.normal, Vec3(1,0,0)));
    TEST_ASSERT(approx(pc.penetration, 0.5f));
    TEST_ASSERT(approx(pc.normalImpulse, 0.0f));
    return true;
}

static bool test_persistent_contact_tangent_basis_x() {
    PersistentContact pc;
    pc.normal = Vec3(1, 0, 0);
    pc.computeTangents();

    // tangent0 and tangent1 should be perpendicular to normal and each other
    TEST_ASSERT(approx(pc.tangent0.dot(pc.normal), 0.0f, 0.001f));
    TEST_ASSERT(approx(pc.tangent1.dot(pc.normal), 0.0f, 0.001f));
    TEST_ASSERT(approx(pc.tangent0.dot(pc.tangent1), 0.0f, 0.001f));
    TEST_ASSERT(approx(pc.tangent0.length(), 1.0f, 0.001f));
    TEST_ASSERT(approx(pc.tangent1.length(), 1.0f, 0.001f));
    return true;
}

static bool test_persistent_contact_tangent_basis_y() {
    PersistentContact pc;
    pc.normal = Vec3(0, 1, 0);
    pc.computeTangents();

    TEST_ASSERT(approx(pc.tangent0.dot(pc.normal), 0.0f, 0.001f));
    TEST_ASSERT(approx(pc.tangent1.dot(pc.normal), 0.0f, 0.001f));
    TEST_ASSERT(approx(pc.tangent0.dot(pc.tangent1), 0.0f, 0.001f));
    return true;
}

static bool test_persistent_contact_tangent_basis_z() {
    PersistentContact pc;
    pc.normal = Vec3(0, 0, 1);
    pc.computeTangents();

    TEST_ASSERT(approx(pc.tangent0.dot(pc.normal), 0.0f, 0.001f));
    TEST_ASSERT(approx(pc.tangent1.dot(pc.normal), 0.0f, 0.001f));
    TEST_ASSERT(approx(pc.tangent0.dot(pc.tangent1), 0.0f, 0.001f));
    return true;
}

static bool test_persistent_contact_tangent_basis_diagonal() {
    PersistentContact pc;
    Vec3 n(1, 1, 1);
    pc.normal = n * (1.0f / n.length());
    pc.computeTangents();

    TEST_ASSERT(approx(pc.tangent0.dot(pc.normal), 0.0f, 0.001f));
    TEST_ASSERT(approx(pc.tangent1.dot(pc.normal), 0.0f, 0.001f));
    TEST_ASSERT(approx(pc.tangent0.dot(pc.tangent1), 0.0f, 0.001f));
    return true;
}

static bool test_persistent_contact_scale_impulses() {
    PersistentContact pc;
    pc.normalImpulse = 10.0f;
    pc.tangentImpulse0 = 5.0f;
    pc.tangentImpulse1 = -3.0f;
    pc.scaleImpulses(0.5f);
    TEST_ASSERT(approx(pc.normalImpulse, 5.0f));
    TEST_ASSERT(approx(pc.tangentImpulse0, 2.5f));
    TEST_ASSERT(approx(pc.tangentImpulse1, -1.5f));
    return true;
}

static bool test_persistent_contact_clear_impulses() {
    PersistentContact pc;
    pc.normalImpulse = 10.0f;
    pc.tangentImpulse0 = 5.0f;
    pc.tangentImpulse1 = -3.0f;
    pc.clearImpulses();
    TEST_ASSERT(approx(pc.normalImpulse, 0.0f));
    TEST_ASSERT(approx(pc.tangentImpulse0, 0.0f));
    TEST_ASSERT(approx(pc.tangentImpulse1, 0.0f));
    return true;
}

static bool test_persistent_contact_has_impulse() {
    PersistentContact pc;
    pc.flags = ContactFlags::HasImpulse;
    TEST_ASSERT(!pc.hasImpulseData()); // impulses are zero
    pc.normalImpulse = 1.0f;
    TEST_ASSERT(pc.hasImpulseData());
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 4: PersistentManifold
// ═════════════════════════════════════════════════════════════════════════════

static bool test_manifold_default() {
    PersistentManifold pm;
    TEST_ASSERT(pm.isEmpty());
    TEST_ASSERT(pm.contactCount == 0);
    TEST_ASSERT(pm.framesActive == 0);
    return true;
}

static bool test_manifold_construct_with_ids() {
    PersistentManifold pm(10, 20);
    TEST_ASSERT(pm.bodyIdA == 10);
    TEST_ASSERT(pm.bodyIdB == 20);
    TEST_ASSERT(pm.isEmpty());
    return true;
}

static bool test_manifold_merge_new_contacts() {
    PersistentManifold pm(1, 2);
    ContactConfig config;

    ContactPoint pts[2] = {
        makeContact(Vec3(0,0,0), Vec3(0,-1,0), Vec3(0,1,0), 0.1f),
        makeContact(Vec3(1,0,0), Vec3(1,-1,0), Vec3(0,1,0), 0.2f)
    };
    ContactManifold cm = makeManifold(pts, 2);

    pm.mergeContacts(cm, config.matchDistanceSq);
    TEST_ASSERT(pm.contactCount == 2);
    TEST_ASSERT(pm.framesActive == 1);
    TEST_ASSERT(pm.refreshedThisFrame);
    return true;
}

static bool test_manifold_match_by_feature_id() {
    PersistentManifold pm(1, 2);
    ContactConfig config;

    // Frame 1: add contact with feature IDs
    ContactPoint cp1 = makeContact(Vec3(0,0,0), Vec3(0,-1,0), Vec3(0,1,0), 0.1f, 42, 99);
    ContactManifold cm1 = makeManifold(&cp1, 1);
    pm.mergeContacts(cm1, config.matchDistanceSq);

    // Set some impulse
    pm.contacts[0].normalImpulse = 5.0f;
    pm.contacts[0].flags |= ContactFlags::HasImpulse;

    // Frame 2: same feature IDs, slightly different position
    ContactPoint cp2 = makeContact(Vec3(0.01f,0,0), Vec3(0.01f,-0.99f,0), Vec3(0,1,0), 0.11f, 42, 99);
    ContactManifold cm2 = makeManifold(&cp2, 1);
    pm.mergeContacts(cm2, config.matchDistanceSq);

    TEST_ASSERT(pm.contactCount == 1);
    // Impulse should be preserved
    TEST_ASSERT(approx(pm.contacts[0].normalImpulse, 5.0f));
    // Geometry should be updated
    TEST_ASSERT(approx(pm.contacts[0].penetration, 0.11f));
    // Flag should be Persisted
    TEST_ASSERT(hasFlag(pm.contacts[0].flags, ContactFlags::Persisted));
    return true;
}

static bool test_manifold_match_by_proximity() {
    PersistentManifold pm(1, 2);
    ContactConfig config;

    // Frame 1: contact with no feature IDs (0xFFFFFFFF)
    ContactPoint cp1 = makeContact(Vec3(0,0,0), Vec3(0,-1,0), Vec3(0,1,0), 0.1f);
    ContactManifold cm1 = makeManifold(&cp1, 1);
    pm.mergeContacts(cm1, config.matchDistanceSq);
    pm.contacts[0].normalImpulse = 3.0f;
    pm.contacts[0].flags |= ContactFlags::HasImpulse;

    // Frame 2: nearby position (within match distance of 4cm)
    ContactPoint cp2 = makeContact(Vec3(0.02f,0,0), Vec3(0.02f,-1,0), Vec3(0,1,0), 0.12f);
    ContactManifold cm2 = makeManifold(&cp2, 1);
    pm.mergeContacts(cm2, config.matchDistanceSq);

    TEST_ASSERT(pm.contactCount == 1);
    TEST_ASSERT(approx(pm.contacts[0].normalImpulse, 3.0f));
    TEST_ASSERT(approx(pm.contacts[0].penetration, 0.12f));
    return true;
}

static bool test_manifold_no_match_creates_new() {
    PersistentManifold pm(1, 2);
    ContactConfig config;

    // Frame 1
    ContactPoint cp1 = makeContact(Vec3(0,0,0), Vec3(0,-1,0), Vec3(0,1,0), 0.1f);
    ContactManifold cm1 = makeManifold(&cp1, 1);
    pm.mergeContacts(cm1, config.matchDistanceSq);
    pm.contacts[0].normalImpulse = 5.0f;

    // Frame 2: contact far away — no match
    ContactPoint cp2 = makeContact(Vec3(10,0,0), Vec3(10,-1,0), Vec3(0,1,0), 0.2f);
    ContactManifold cm2 = makeManifold(&cp2, 1);
    pm.mergeContacts(cm2, config.matchDistanceSq);

    TEST_ASSERT(pm.contactCount == 1);
    // Impulse should be zero (new contact)
    TEST_ASSERT(approx(pm.contacts[0].normalImpulse, 0.0f));
    TEST_ASSERT(hasFlag(pm.contacts[0].flags, ContactFlags::New));
    return true;
}

static bool test_manifold_merge_4_contacts() {
    PersistentManifold pm(1, 2);
    ContactConfig config;

    ContactPoint pts[4] = {
        makeContact(Vec3( 1, 0,  1), Vec3( 1,-1, 1), Vec3(0,1,0), 0.1f),
        makeContact(Vec3(-1, 0,  1), Vec3(-1,-1, 1), Vec3(0,1,0), 0.2f),
        makeContact(Vec3(-1, 0, -1), Vec3(-1,-1,-1), Vec3(0,1,0), 0.3f),
        makeContact(Vec3( 1, 0, -1), Vec3( 1,-1,-1), Vec3(0,1,0), 0.4f)
    };
    ContactManifold cm = makeManifold(pts, 4);
    pm.mergeContacts(cm, config.matchDistanceSq);

    TEST_ASSERT(pm.contactCount == 4);
    return true;
}

static bool test_manifold_prune_separated() {
    PersistentManifold pm(1, 2);
    ContactConfig config;

    // Add contacts
    ContactPoint pts[2] = {
        makeContact(Vec3(0,0,0), Vec3(0,-1,0), Vec3(0,1,0), 0.1f),
        makeContact(Vec3(1,0,0), Vec3(1,-1,0), Vec3(0,1,0), -0.05f) // slightly separated
    };
    ContactManifold cm = makeManifold(pts, 2);
    pm.mergeContacts(cm, config.matchDistanceSq);
    TEST_ASSERT(pm.contactCount == 2);

    // Prune with default break distance (0.02m)
    // The second contact has pen=-0.05 which is < -0.02, so it should be pruned
    pm.pruneStale(config.breakDistance);
    TEST_ASSERT(pm.contactCount == 1);
    TEST_ASSERT(approx(pm.contacts[0].penetration, 0.1f));
    return true;
}

static bool test_manifold_prune_keeps_valid() {
    PersistentManifold pm(1, 2);
    ContactConfig config;

    // All contacts within break distance
    ContactPoint pts[3] = {
        makeContact(Vec3(0,0,0), Vec3(0,-1,0), Vec3(0,1,0), 0.1f),
        makeContact(Vec3(1,0,0), Vec3(1,-1,0), Vec3(0,1,0), 0.05f),
        makeContact(Vec3(2,0,0), Vec3(2,-1,0), Vec3(0,1,0), -0.01f) // barely separated
    };
    ContactManifold cm = makeManifold(pts, 3);
    pm.mergeContacts(cm, config.matchDistanceSq);

    pm.pruneStale(config.breakDistance);
    TEST_ASSERT(pm.contactCount == 3); // all valid
    return true;
}

static bool test_manifold_get_key() {
    PersistentManifold pm(10, 20);
    BodyPairKey key = pm.getKey();
    TEST_ASSERT(key.low == 10);
    TEST_ASSERT(key.high == 20);
    return true;
}

static bool test_manifold_max_penetration() {
    PersistentManifold pm(1, 2);
    ContactConfig config;

    ContactPoint pts[3] = {
        makeContact(Vec3(0,0,0), Vec3(0,-1,0), Vec3(0,1,0), 0.1f),
        makeContact(Vec3(1,0,0), Vec3(1,-1,0), Vec3(0,1,0), 0.5f),
        makeContact(Vec3(2,0,0), Vec3(2,-1,0), Vec3(0,1,0), 0.3f)
    };
    ContactManifold cm = makeManifold(pts, 3);
    pm.mergeContacts(cm, config.matchDistanceSq);

    TEST_ASSERT(approx(pm.getMaxPenetration(), 0.5f));
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 5: ContactCache
// ═════════════════════════════════════════════════════════════════════════════

static bool test_cache_empty() {
    ContactCache cache(64);
    TEST_ASSERT(cache.count() == 0);
    TEST_ASSERT(cache.capacity() >= 64);
    return true;
}

static bool test_cache_insert_and_find() {
    ContactCache cache(64);
    BodyPairKey key(1, 2);
    PersistentManifold* pm = cache.insertOrFind(key, 1, 2);
    TEST_ASSERT(pm != nullptr);
    TEST_ASSERT(cache.count() == 1);

    PersistentManifold* found = cache.find(key);
    TEST_ASSERT(found != nullptr);
    TEST_ASSERT(found->bodyIdA == 1);
    TEST_ASSERT(found->bodyIdB == 2);
    return true;
}

static bool test_cache_insert_duplicate() {
    ContactCache cache(64);
    BodyPairKey key(3, 7);
    PersistentManifold* pm1 = cache.insertOrFind(key, 3, 7);
    PersistentManifold* pm2 = cache.insertOrFind(key, 3, 7);
    TEST_ASSERT(pm1 != nullptr);
    TEST_ASSERT(pm2 != nullptr);
    TEST_ASSERT(cache.count() == 1); // Should not create a second entry
    return true;
}

static bool test_cache_find_not_present() {
    ContactCache cache(64);
    BodyPairKey key(1, 2);
    PersistentManifold* found = cache.find(key);
    TEST_ASSERT(found == nullptr);
    return true;
}

static bool test_cache_remove() {
    ContactCache cache(64);
    BodyPairKey key(5, 10);
    cache.insertOrFind(key, 5, 10);
    TEST_ASSERT(cache.count() == 1);

    bool removed = cache.remove(key);
    TEST_ASSERT(removed);
    TEST_ASSERT(cache.count() == 0);
    TEST_ASSERT(cache.find(key) == nullptr);
    return true;
}

static bool test_cache_remove_not_present() {
    ContactCache cache(64);
    BodyPairKey key(1, 2);
    bool removed = cache.remove(key);
    TEST_ASSERT(!removed);
    return true;
}

static bool test_cache_multiple_pairs() {
    ContactCache cache(256);

    for (uint32_t i = 0; i < 50; ++i) {
        BodyPairKey key(i, i + 100);
        cache.insertOrFind(key, i, i + 100);
    }
    TEST_ASSERT(cache.count() == 50);

    // Verify all are findable
    for (uint32_t i = 0; i < 50; ++i) {
        BodyPairKey key(i, i + 100);
        PersistentManifold* pm = cache.find(key);
        TEST_ASSERT(pm != nullptr);
        TEST_ASSERT(pm->bodyIdA == i);
    }
    return true;
}

static bool test_cache_clear_all() {
    ContactCache cache(64);
    for (uint32_t i = 0; i < 20; ++i) {
        cache.insertOrFind(BodyPairKey(i, i + 50), i, i + 50);
    }
    TEST_ASSERT(cache.count() == 20);
    cache.clearAll();
    TEST_ASSERT(cache.count() == 0);
    return true;
}

static bool test_cache_load_factor() {
    ContactCache cache(64);
    TEST_ASSERT(approx(cache.loadFactor(), 0.0f));
    cache.insertOrFind(BodyPairKey(1, 2), 1, 2);
    TEST_ASSERT(cache.loadFactor() > 0.0f);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 6: Frame Update Pipeline
// ═════════════════════════════════════════════════════════════════════════════

static bool test_pipeline_process_manifold() {
    ContactCache cache(256);
    ContactConfig config;

    ContactPoint cp = makeContact(Vec3(0,0,0), Vec3(0,-1,0), Vec3(0,1,0), 0.1f);
    ContactManifold cm = makeManifold(&cp, 1);

    PersistentManifold* pm = cache.processManifold(1, 2, cm, config);
    TEST_ASSERT(pm != nullptr);
    TEST_ASSERT(pm->contactCount == 1);
    TEST_ASSERT(cache.count() == 1);
    return true;
}

static bool test_pipeline_begin_end_frame() {
    ContactCache cache(256);
    ContactConfig config;

    // Frame 1: add a pair
    cache.beginFrame();
    ContactPoint cp = makeContact(Vec3(0,0,0), Vec3(0,-1,0), Vec3(0,1,0), 0.1f);
    ContactManifold cm = makeManifold(&cp, 1);
    cache.processManifold(1, 2, cm, config);
    TEST_ASSERT(cache.count() == 1);

    // Frame 2: don't refresh the pair → endFrame should remove it
    cache.beginFrame();
    cache.endFrame();
    TEST_ASSERT(cache.count() == 0);
    return true;
}

static bool test_pipeline_persistent_pair() {
    ContactCache cache(256);
    ContactConfig config;

    // Frame 1
    cache.beginFrame();
    ContactPoint cp1 = makeContact(Vec3(0,0,0), Vec3(0,-1,0), Vec3(0,1,0), 0.1f, 1, 1);
    ContactManifold cm1 = makeManifold(&cp1, 1);
    cache.processManifold(1, 2, cm1, config);

    // Simulate solver writing impulse
    PersistentManifold* pm = cache.find(BodyPairKey(1, 2));
    TEST_ASSERT(pm != nullptr);
    pm->contacts[0].normalImpulse = 10.0f;
    pm->contacts[0].flags |= ContactFlags::HasImpulse;

    // Frame 2: same pair, refreshed
    cache.beginFrame();
    ContactPoint cp2 = makeContact(Vec3(0.001f,0,0), Vec3(0.001f,-1,0), Vec3(0,1,0), 0.11f, 1, 1);
    ContactManifold cm2 = makeManifold(&cp2, 1);
    cache.processManifold(1, 2, cm2, config);
    cache.endFrame();

    TEST_ASSERT(cache.count() == 1);
    pm = cache.find(BodyPairKey(1, 2));
    TEST_ASSERT(pm != nullptr);
    TEST_ASSERT(pm->contactCount == 1);
    // Impulse should be preserved from matching
    TEST_ASSERT(approx(pm->contacts[0].normalImpulse, 10.0f));
    return true;
}

static bool test_pipeline_remove_old_keep_new() {
    ContactCache cache(256);
    ContactConfig config;

    // Frame 1: two pairs
    cache.beginFrame();
    ContactPoint cp = makeContact(Vec3(0,0,0), Vec3(0,-1,0), Vec3(0,1,0), 0.1f);
    ContactManifold cm = makeManifold(&cp, 1);
    cache.processManifold(1, 2, cm, config);
    cache.processManifold(3, 4, cm, config);
    TEST_ASSERT(cache.count() == 2);

    // Frame 2: only refresh pair (1,2)
    cache.beginFrame();
    cache.processManifold(1, 2, cm, config);
    cache.endFrame();

    TEST_ASSERT(cache.count() == 1);
    TEST_ASSERT(cache.find(BodyPairKey(1, 2)) != nullptr);
    TEST_ASSERT(cache.find(BodyPairKey(3, 4)) == nullptr);
    return true;
}

static bool test_pipeline_total_contact_count() {
    ContactCache cache(256);
    ContactConfig config;

    cache.beginFrame();

    ContactPoint pts2[2] = {
        makeContact(Vec3(0,0,0), Vec3(0,-1,0), Vec3(0,1,0), 0.1f),
        makeContact(Vec3(1,0,0), Vec3(1,-1,0), Vec3(0,1,0), 0.2f)
    };
    ContactManifold cm2 = makeManifold(pts2, 2);
    cache.processManifold(1, 2, cm2, config);

    ContactPoint pts3[3] = {
        makeContact(Vec3(0,0,0), Vec3(0,-1,0), Vec3(0,1,0), 0.1f),
        makeContact(Vec3(1,0,0), Vec3(1,-1,0), Vec3(0,1,0), 0.2f),
        makeContact(Vec3(2,0,0), Vec3(2,-1,0), Vec3(0,1,0), 0.3f)
    };
    ContactManifold cm3 = makeManifold(pts3, 3);
    cache.processManifold(3, 4, cm3, config);

    TEST_ASSERT(cache.totalContactCount() == 5);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 7: Warm Starting
// ═════════════════════════════════════════════════════════════════════════════

static bool test_warm_start_manifold() {
    PersistentManifold pm(1, 2);
    ContactConfig config;

    ContactPoint cp = makeContact(Vec3(0,0,0), Vec3(0,-1,0), Vec3(0,1,0), 0.1f);
    ContactManifold cm = makeManifold(&cp, 1);
    pm.mergeContacts(cm, config.matchDistanceSq);

    // Simulate solver writing impulse
    pm.contacts[0].normalImpulse = 10.0f;
    pm.contacts[0].tangentImpulse0 = 2.0f;
    pm.contacts[0].flags = ContactFlags::Persisted | ContactFlags::HasImpulse;

    warmStartManifold(pm, 0.85f, 0.3f);
    // Persisted contact should use factor 0.85
    TEST_ASSERT(approx(pm.contacts[0].normalImpulse, 8.5f));
    TEST_ASSERT(approx(pm.contacts[0].tangentImpulse0, 1.7f));
    return true;
}

static bool test_warm_start_new_contact_damping() {
    PersistentManifold pm(1, 2);
    ContactConfig config;

    ContactPoint cp = makeContact(Vec3(0,0,0), Vec3(0,-1,0), Vec3(0,1,0), 0.1f);
    ContactManifold cm = makeManifold(&cp, 1);
    pm.mergeContacts(cm, config.matchDistanceSq);

    pm.contacts[0].normalImpulse = 10.0f;
    // Keep as New flag
    TEST_ASSERT(hasFlag(pm.contacts[0].flags, ContactFlags::New));

    warmStartManifold(pm, 0.85f, 0.3f);
    // New contact should use damping 0.3
    TEST_ASSERT(approx(pm.contacts[0].normalImpulse, 3.0f));
    return true;
}

static bool test_warm_start_all() {
    ContactCache cache(256);
    ContactConfig config;

    cache.beginFrame();

    ContactPoint cp = makeContact(Vec3(0,0,0), Vec3(0,-1,0), Vec3(0,1,0), 0.1f, 1, 1);
    ContactManifold cm = makeManifold(&cp, 1);
    cache.processManifold(1, 2, cm, config);

    // Set impulse
    PersistentManifold* pm = cache.find(BodyPairKey(1, 2));
    pm->contacts[0].normalImpulse = 10.0f;
    pm->contacts[0].flags = ContactFlags::Persisted | ContactFlags::HasImpulse;

    warmStartAllManifolds(cache, config);

    pm = cache.find(BodyPairKey(1, 2));
    TEST_ASSERT(approx(pm->contacts[0].normalImpulse, 10.0f * config.warmStartFactor));
    return true;
}

static bool test_warm_start_stats() {
    ContactCache cache(256);
    ContactConfig config;

    cache.beginFrame();

    ContactPoint pts[2] = {
        makeContact(Vec3(0,0,0), Vec3(0,-1,0), Vec3(0,1,0), 0.1f),
        makeContact(Vec3(1,0,0), Vec3(1,-1,0), Vec3(0,1,0), 0.2f)
    };
    ContactManifold cm = makeManifold(pts, 2);
    cache.processManifold(1, 2, cm, config);

    WarmStartStats stats = getWarmStartStats(cache);
    TEST_ASSERT(stats.totalManifolds == 1);
    TEST_ASSERT(stats.totalContacts == 2);
    TEST_ASSERT(stats.newContacts == 2); // All new this frame
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 8: Edge Cases and Multi-Frame
// ═════════════════════════════════════════════════════════════════════════════

static bool test_edge_empty_manifold_merge() {
    PersistentManifold pm(1, 2);
    ContactConfig config;

    ContactManifold empty;
    pm.mergeContacts(empty, config.matchDistanceSq);

    TEST_ASSERT(pm.contactCount == 0);
    TEST_ASSERT(pm.refreshedThisFrame);
    return true;
}

static bool test_edge_self_pair_key() {
    BodyPairKey key(42, 42);
    TEST_ASSERT(key.low == 42);
    TEST_ASSERT(key.high == 42);
    TEST_ASSERT(key.isValid());
    return true;
}

static bool test_edge_cache_stress() {
    ContactCache cache(1024);
    ContactConfig config;

    // Insert many pairs
    uint32_t inserted = 0;
    for (uint32_t i = 0; i < 500; ++i) {
        BodyPairKey key(i, i + 1000);
        PersistentManifold* pm = cache.insertOrFind(key, i, i + 1000);
        if (pm) inserted++;
    }
    TEST_ASSERT(inserted == 500);
    TEST_ASSERT(cache.count() == 500);

    // Remove half
    for (uint32_t i = 0; i < 250; ++i) {
        cache.remove(BodyPairKey(i, i + 1000));
    }
    TEST_ASSERT(cache.count() == 250);

    // Remaining should still be findable
    for (uint32_t i = 250; i < 500; ++i) {
        TEST_ASSERT(cache.find(BodyPairKey(i, i + 1000)) != nullptr);
    }
    return true;
}

static bool test_multiframe_contact_aging() {
    ContactCache cache(256);
    ContactConfig config;

    ContactPoint cp = makeContact(Vec3(0,0,0), Vec3(0,-1,0), Vec3(0,1,0), 0.1f, 1, 1);
    ContactManifold cm = makeManifold(&cp, 1);

    // Simulate 10 frames with the same contact
    for (int frame = 0; frame < 10; ++frame) {
        cache.beginFrame();
        cache.processManifold(1, 2, cm, config);
        cache.endFrame();
    }

    PersistentManifold* pm = cache.find(BodyPairKey(1, 2));
    TEST_ASSERT(pm != nullptr);
    TEST_ASSERT(pm->contactCount == 1);
    TEST_ASSERT(pm->contacts[0].contactAge >= 9); // aged through merging
    TEST_ASSERT(pm->framesActive == 10);
    return true;
}

static bool test_multiframe_impulse_continuity() {
    ContactCache cache(256);
    ContactConfig config;
    config.warmStartFactor = 1.0f;  // No scaling for this test
    config.newContactDamping = 0.0f;

    ContactPoint cp = makeContact(Vec3(0,0,0), Vec3(0,-1,0), Vec3(0,1,0), 0.1f, 1, 1);
    ContactManifold cm = makeManifold(&cp, 1);

    // Frame 1: establish contact
    cache.beginFrame();
    cache.processManifold(1, 2, cm, config);
    PersistentManifold* pm = cache.find(BodyPairKey(1, 2));
    pm->contacts[0].normalImpulse = 20.0f;
    pm->contacts[0].flags = ContactFlags::Persisted | ContactFlags::HasImpulse;
    cache.endFrame();

    // Frame 2: contact persists → impulse should survive
    cache.beginFrame();
    cache.processManifold(1, 2, cm, config);
    warmStartAllManifolds(cache, config);
    cache.endFrame();

    pm = cache.find(BodyPairKey(1, 2));
    TEST_ASSERT(pm != nullptr);
    // With factor 1.0 and Persisted flag, impulse should be 20.0
    TEST_ASSERT(approx(pm->contacts[0].normalImpulse, 20.0f));
    return true;
}

static bool test_multiframe_contacts_appear_disappear() {
    ContactCache cache(256);
    ContactConfig config;

    // Frame 1-5: pair (1,2) exists
    for (int f = 0; f < 5; ++f) {
        cache.beginFrame();
        ContactPoint cp = makeContact(Vec3(0,0,0), Vec3(0,-1,0), Vec3(0,1,0), 0.1f);
        ContactManifold cm = makeManifold(&cp, 1);
        cache.processManifold(1, 2, cm, config);
        cache.endFrame();
    }
    TEST_ASSERT(cache.count() == 1);

    // Frame 6: pair (1,2) disappears, pair (3,4) appears
    cache.beginFrame();
    ContactPoint cp2 = makeContact(Vec3(5,0,0), Vec3(5,-1,0), Vec3(0,1,0), 0.2f);
    ContactManifold cm2 = makeManifold(&cp2, 1);
    cache.processManifold(3, 4, cm2, config);
    cache.endFrame();

    TEST_ASSERT(cache.count() == 1);
    TEST_ASSERT(cache.find(BodyPairKey(1, 2)) == nullptr);
    TEST_ASSERT(cache.find(BodyPairKey(3, 4)) != nullptr);
    return true;
}

static bool test_cache_move_constructor() {
    ContactCache cache1(128);
    cache1.insertOrFind(BodyPairKey(1, 2), 1, 2);
    cache1.insertOrFind(BodyPairKey(3, 4), 3, 4);
    TEST_ASSERT(cache1.count() == 2);

    ContactCache cache2(std::move(cache1));
    TEST_ASSERT(cache2.count() == 2);
    TEST_ASSERT(cache2.find(BodyPairKey(1, 2)) != nullptr);
    TEST_ASSERT(cache2.find(BodyPairKey(3, 4)) != nullptr);
    return true;
}

static bool test_config_custom() {
    ContactConfig config(0.05f, 0.1f, 0.9f, 0.5f, 8192);
    TEST_ASSERT(approx(config.breakDistance, 0.05f));
    TEST_ASSERT(approx(config.matchDistanceSq, 0.01f)); // 0.1^2
    TEST_ASSERT(approx(config.warmStartFactor, 0.9f));
    TEST_ASSERT(approx(config.newContactDamping, 0.5f));
    TEST_ASSERT(config.maxCachedPairs == 8192);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("=== Pulse Contact Module Tests ===\n\n");

    // Group 1: BodyPairKey
    std::printf("--- BodyPairKey ---\n");
    RUN_TEST(test_bodypairkey_canonical_order);
    RUN_TEST(test_bodypairkey_same_id);
    RUN_TEST(test_bodypairkey_hash_consistency);
    RUN_TEST(test_bodypairkey_hash_uniqueness);
    RUN_TEST(test_bodypairkey_packed);
    RUN_TEST(test_bodypairkey_invalid_default);
    RUN_TEST(test_bodypairkey_valid);
    RUN_TEST(test_bodypairkey_less_than);

    // Group 2: ContactFlags
    std::printf("--- ContactFlags ---\n");
    RUN_TEST(test_flags_none);
    RUN_TEST(test_flags_set_and_test);
    RUN_TEST(test_flags_combine);
    RUN_TEST(test_flags_mask);

    // Group 3: PersistentContact
    std::printf("--- PersistentContact ---\n");
    RUN_TEST(test_persistent_contact_default);
    RUN_TEST(test_persistent_contact_from_cp);
    RUN_TEST(test_persistent_contact_tangent_basis_x);
    RUN_TEST(test_persistent_contact_tangent_basis_y);
    RUN_TEST(test_persistent_contact_tangent_basis_z);
    RUN_TEST(test_persistent_contact_tangent_basis_diagonal);
    RUN_TEST(test_persistent_contact_scale_impulses);
    RUN_TEST(test_persistent_contact_clear_impulses);
    RUN_TEST(test_persistent_contact_has_impulse);

    // Group 4: PersistentManifold
    std::printf("--- PersistentManifold ---\n");
    RUN_TEST(test_manifold_default);
    RUN_TEST(test_manifold_construct_with_ids);
    RUN_TEST(test_manifold_merge_new_contacts);
    RUN_TEST(test_manifold_match_by_feature_id);
    RUN_TEST(test_manifold_match_by_proximity);
    RUN_TEST(test_manifold_no_match_creates_new);
    RUN_TEST(test_manifold_merge_4_contacts);
    RUN_TEST(test_manifold_prune_separated);
    RUN_TEST(test_manifold_prune_keeps_valid);
    RUN_TEST(test_manifold_get_key);
    RUN_TEST(test_manifold_max_penetration);

    // Group 5: ContactCache
    std::printf("--- ContactCache ---\n");
    RUN_TEST(test_cache_empty);
    RUN_TEST(test_cache_insert_and_find);
    RUN_TEST(test_cache_insert_duplicate);
    RUN_TEST(test_cache_find_not_present);
    RUN_TEST(test_cache_remove);
    RUN_TEST(test_cache_remove_not_present);
    RUN_TEST(test_cache_multiple_pairs);
    RUN_TEST(test_cache_clear_all);
    RUN_TEST(test_cache_load_factor);

    // Group 6: Frame Update Pipeline
    std::printf("--- Frame Update Pipeline ---\n");
    RUN_TEST(test_pipeline_process_manifold);
    RUN_TEST(test_pipeline_begin_end_frame);
    RUN_TEST(test_pipeline_persistent_pair);
    RUN_TEST(test_pipeline_remove_old_keep_new);
    RUN_TEST(test_pipeline_total_contact_count);

    // Group 7: Warm Starting
    std::printf("--- Warm Starting ---\n");
    RUN_TEST(test_warm_start_manifold);
    RUN_TEST(test_warm_start_new_contact_damping);
    RUN_TEST(test_warm_start_all);
    RUN_TEST(test_warm_start_stats);

    // Group 8: Edge Cases & Multi-Frame
    std::printf("--- Edge Cases & Multi-Frame ---\n");
    RUN_TEST(test_edge_empty_manifold_merge);
    RUN_TEST(test_edge_self_pair_key);
    RUN_TEST(test_edge_cache_stress);
    RUN_TEST(test_multiframe_contact_aging);
    RUN_TEST(test_multiframe_impulse_continuity);
    RUN_TEST(test_multiframe_contacts_appear_disappear);
    RUN_TEST(test_cache_move_constructor);
    RUN_TEST(test_config_custom);

    // Summary
    std::printf("\n=== Results: %d/%d passed", g_passed, g_total);
    if (g_failed > 0) std::printf(", %d FAILED", g_failed);
    std::printf(" ===\n");

    return (g_failed > 0) ? 1 : 0;
}
