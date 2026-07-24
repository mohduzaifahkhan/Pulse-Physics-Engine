/**
 * @file contact_stub.cpp
 * @brief Compilation stub for the Pulse contact module.
 *
 * Ensures all contact headers compile cleanly as part of the static library.
 */

#include <pulse/contact/contact_common.h>
#include <pulse/contact/persistent_contact.h>
#include <pulse/contact/contact_manifold_persistent.h>
#include <pulse/contact/contact_cache.h>
#include <pulse/contact/warm_start.h>

namespace pulse {
namespace {
    [[maybe_unused]] void contact_compile_check() {
        // contact_common.h
        BodyPairKey key(1, 2);
        (void)key.hash();
        (void)key.packed();
        ContactFlags flags = ContactFlags::New | ContactFlags::Persisted;
        (void)hasFlag(flags, ContactFlags::New);
        ContactConfig config;
        (void)config;

        // persistent_contact.h
        PersistentContact pc;
        pc.computeTangents();
        pc.scaleImpulses(0.5f);
        pc.clearImpulses();
        (void)pc.hasImpulseData();

        // contact_manifold_persistent.h
        PersistentManifold pm(1, 2);
        (void)pm.getKey();
        (void)pm.isEmpty();
        (void)pm.getMaxPenetration();
        pm.clear();

        // contact_cache.h
        ContactCache cache(64);
        (void)cache.find(key);
        (void)cache.count();
        (void)cache.capacity();
        cache.beginFrame();
        cache.endFrame();

        // warm_start.h
        WarmStartStats stats = getWarmStartStats(cache);
        (void)stats;
    }
}
}
