/**
 * @file contact_cache.h
 * @brief Hash-map cache for persistent contact manifolds.
 *
 * Open-addressing hash table (Robin Hood hashing) indexed by BodyPairKey.
 * This is the central data structure that persists contact manifolds
 * across physics frames, enabling warm starting of the constraint solver.
 *
 * Design: Power-of-2 capacity, linear probing with Robin Hood displacement.
 * Load factor is kept below 0.75. Tombstones are avoided by using
 * backward-shift deletion, keeping the probe sequences contiguous.
 *
 * Thread safety: NOT thread-safe. All access should be from the main
 * physics thread or protected externally.
 */

#pragma once

#include "contact_manifold_persistent.h"
#include "contact_common.h"
#include <pulse/math/math_common.h>

#include <cstdint>
#include <cstring>

namespace pulse {

/**
 * @class ContactCache
 * @brief Open-addressing hash map of BodyPairKey → PersistentManifold.
 *
 * Primary per-frame workflow:
 * 1. `beginFrame()` — clear dirty bits.
 * 2. `updateFrame()` — merge new narrow-phase results into cache.
 * 3. `warmStartAll()` — prepare impulses for the solver.
 * 4. After solving, impulses are written back into the manifolds.
 */
class ContactCache {
public:
    // ── Construction / destruction ────────────────────────────────────────

    /// Construct with a maximum capacity (rounded up to power of 2).
    explicit ContactCache(uint32_t maxPairs = 16384u) noexcept {
        // Round up to next power of 2
        capacity_ = nextPow2(maxPairs < 64u ? 64u : maxPairs);
        mask_ = capacity_ - 1;
        count_ = 0;
        frameNumber_ = 0;

        // Allocate storage
        keys_      = new BodyPairKey[capacity_];
        manifolds_ = new PersistentManifold[capacity_];
        occupied_  = new bool[capacity_];

        // Initialize all slots as empty
        for (uint32_t i = 0; i < capacity_; ++i) {
            occupied_[i] = false;
        }
    }

    ~ContactCache() noexcept {
        delete[] keys_;
        delete[] manifolds_;
        delete[] occupied_;
    }

    // Non-copyable, movable
    ContactCache(const ContactCache&) = delete;
    ContactCache& operator=(const ContactCache&) = delete;

    ContactCache(ContactCache&& other) noexcept
        : keys_(other.keys_), manifolds_(other.manifolds_),
          occupied_(other.occupied_), capacity_(other.capacity_),
          mask_(other.mask_), count_(other.count_),
          frameNumber_(other.frameNumber_)
    {
        other.keys_ = nullptr;
        other.manifolds_ = nullptr;
        other.occupied_ = nullptr;
        other.capacity_ = 0;
        other.mask_ = 0;
        other.count_ = 0;
    }

    ContactCache& operator=(ContactCache&& other) noexcept {
        if (this != &other) {
            delete[] keys_;
            delete[] manifolds_;
            delete[] occupied_;
            keys_      = other.keys_;
            manifolds_ = other.manifolds_;
            occupied_  = other.occupied_;
            capacity_  = other.capacity_;
            mask_      = other.mask_;
            count_     = other.count_;
            frameNumber_ = other.frameNumber_;
            other.keys_ = nullptr;
            other.manifolds_ = nullptr;
            other.occupied_ = nullptr;
            other.capacity_ = 0;
            other.mask_ = 0;
            other.count_ = 0;
        }
        return *this;
    }

    // ── Lookup ───────────────────────────────────────────────────────────

    /**
     * @brief Find a manifold by body pair key.
     * @return Pointer to the manifold, or nullptr if not found.
     */
    [[nodiscard]] PersistentManifold* find(const BodyPairKey& key) noexcept {
        uint32_t idx = key.hash() & mask_;
        for (uint32_t probe = 0; probe < capacity_; ++probe) {
            uint32_t slot = (idx + probe) & mask_;
            if (!occupied_[slot]) return nullptr;
            if (keys_[slot] == key) return &manifolds_[slot];
        }
        return nullptr;
    }

    [[nodiscard]] const PersistentManifold* find(const BodyPairKey& key) const noexcept {
        uint32_t idx = key.hash() & mask_;
        for (uint32_t probe = 0; probe < capacity_; ++probe) {
            uint32_t slot = (idx + probe) & mask_;
            if (!occupied_[slot]) return nullptr;
            if (keys_[slot] == key) return &manifolds_[slot];
        }
        return nullptr;
    }

    // ── Insert ───────────────────────────────────────────────────────────

    /**
     * @brief Insert or find a manifold for the given body pair.
     *
     * If the pair already exists, returns the existing manifold.
     * Otherwise, inserts a new empty manifold and returns it.
     *
     * @param key     Body pair key.
     * @param bodyIdA Body A identifier (used for new manifold construction).
     * @param bodyIdB Body B identifier (used for new manifold construction).
     * @return Pointer to the manifold (new or existing), or nullptr if full.
     */
    PersistentManifold* insertOrFind(const BodyPairKey& key,
                                      uint32_t bodyIdA, uint32_t bodyIdB) noexcept
    {
        // Check if already present
        PersistentManifold* existing = find(key);
        if (existing) return existing;

        // Check load factor (< 75%)
        if (count_ * 4 >= capacity_ * 3) return nullptr;

        // Robin Hood insertion
        uint32_t idx = key.hash() & mask_;
        BodyPairKey insertKey = key;
        PersistentManifold insertManifold(bodyIdA, bodyIdB);
        uint32_t insertDist = 0;

        for (uint32_t probe = 0; probe < capacity_; ++probe) {
            uint32_t slot = (idx + probe) & mask_;

            if (!occupied_[slot]) {
                // Empty slot — place here
                keys_[slot] = insertKey;
                manifolds_[slot] = insertManifold;
                occupied_[slot] = true;
                count_++;
                // Return the newly inserted entry (find it by key to handle swaps)
                return find(key);
            }

            // Compute probe distance of existing element
            uint32_t existingDist = probeDistance(slot);

            if (insertDist > existingDist) {
                // Robin Hood: swap with richer element
                BodyPairKey tmpKey = keys_[slot];
                PersistentManifold tmpManifold = manifolds_[slot];

                keys_[slot] = insertKey;
                manifolds_[slot] = insertManifold;

                insertKey = tmpKey;
                insertManifold = tmpManifold;
                insertDist = existingDist;
            }

            insertDist++;
        }

        return nullptr; // Should never reach here if load < 75%
    }

    // ── Remove ───────────────────────────────────────────────────────────

    /**
     * @brief Remove a manifold by body pair key.
     *
     * Uses backward-shift deletion to maintain probe sequence integrity
     * (no tombstones needed).
     *
     * @return true if the pair was found and removed.
     */
    bool remove(const BodyPairKey& key) noexcept {
        uint32_t idx = key.hash() & mask_;

        // Find the slot
        for (uint32_t probe = 0; probe < capacity_; ++probe) {
            uint32_t slot = (idx + probe) & mask_;
            if (!occupied_[slot]) return false;
            if (keys_[slot] == key) {
                // Found — backward-shift delete
                backwardShiftDelete(slot);
                count_--;
                return true;
            }
        }
        return false;
    }

    // ── Per-frame pipeline ───────────────────────────────────────────────

    /**
     * @brief Begin a new physics frame.
     *
     * Clears the refreshed flag on all manifolds so we can detect
     * which pairs were not produced by narrow-phase this frame.
     */
    void beginFrame() noexcept {
        frameNumber_++;
        for (uint32_t i = 0; i < capacity_; ++i) {
            if (occupied_[i]) {
                manifolds_[i].refreshedThisFrame = false;
            }
        }
    }

    /**
     * @brief Process a new narrow-phase manifold for a body pair.
     *
     * Finds or creates the persistent manifold and merges the new
     * contacts into it, preserving matched impulses.
     *
     * @param bodyIdA   Body A identifier.
     * @param bodyIdB   Body B identifier.
     * @param newManifold  Fresh narrow-phase contacts.
     * @param config    Contact configuration parameters.
     * @return Pointer to the updated persistent manifold, or nullptr if cache is full.
     */
    PersistentManifold* processManifold(uint32_t bodyIdA, uint32_t bodyIdB,
                                         const ContactManifold& newManifold,
                                         const ContactConfig& config) noexcept
    {
        BodyPairKey key(bodyIdA, bodyIdB);
        PersistentManifold* pm = insertOrFind(key, bodyIdA, bodyIdB);
        if (!pm) return nullptr;

        pm->mergeContacts(newManifold, config.matchDistanceSq);
        pm->pruneStale(config.breakDistance);
        return pm;
    }

    /**
     * @brief End-of-frame cleanup: remove manifolds not refreshed this frame.
     *
     * If a pair was not produced by narrow-phase this frame, the bodies
     * are no longer overlapping and the manifold should be removed.
     */
    void endFrame() noexcept {
        for (uint32_t i = 0; i < capacity_;) {
            if (occupied_[i] && !manifolds_[i].refreshedThisFrame) {
                backwardShiftDelete(i);
                count_--;
                // Don't increment i — the shifted element needs checking
            } else {
                i++;
            }
        }
    }

    // ── Warm starting ────────────────────────────────────────────────────

    /**
     * @brief Prepare all manifolds for warm starting.
     */
    void warmStartAll(float factor, float newContactDamping) noexcept {
        for (uint32_t i = 0; i < capacity_; ++i) {
            if (occupied_[i] && manifolds_[i].contactCount > 0) {
                manifolds_[i].prepareWarmStart(factor, newContactDamping);
            }
        }
    }

    // ── Iteration ────────────────────────────────────────────────────────

    /// Get the internal capacity.
    [[nodiscard]] uint32_t capacity() const noexcept { return capacity_; }

    /// Get the number of active pairs.
    [[nodiscard]] uint32_t count() const noexcept { return count_; }

    /// Get the current frame number.
    [[nodiscard]] uint64_t frameNumber() const noexcept { return frameNumber_; }

    /// Check if a slot is occupied (for iteration).
    [[nodiscard]] bool isOccupied(uint32_t slot) const noexcept {
        return slot < capacity_ && occupied_[slot];
    }

    /// Get the manifold at a slot (for iteration). Only valid if isOccupied(slot).
    [[nodiscard]] PersistentManifold& manifoldAt(uint32_t slot) noexcept {
        return manifolds_[slot];
    }
    [[nodiscard]] const PersistentManifold& manifoldAt(uint32_t slot) const noexcept {
        return manifolds_[slot];
    }

    /// Get the key at a slot (for iteration). Only valid if isOccupied(slot).
    [[nodiscard]] const BodyPairKey& keyAt(uint32_t slot) const noexcept {
        return keys_[slot];
    }

    /// Get total contact count across all manifolds.
    [[nodiscard]] uint32_t totalContactCount() const noexcept {
        uint32_t total = 0;
        for (uint32_t i = 0; i < capacity_; ++i) {
            if (occupied_[i]) total += manifolds_[i].contactCount;
        }
        return total;
    }

    /// Load factor (0.0 – 1.0).
    [[nodiscard]] float loadFactor() const noexcept {
        return static_cast<float>(count_) / static_cast<float>(capacity_);
    }

    /// Clear all cached manifolds.
    void clearAll() noexcept {
        for (uint32_t i = 0; i < capacity_; ++i) {
            occupied_[i] = false;
        }
        count_ = 0;
    }

private:
    BodyPairKey*         keys_;       ///< Key array.
    PersistentManifold*  manifolds_;  ///< Manifold array.
    bool*                occupied_;   ///< Occupancy flags.
    uint32_t             capacity_;   ///< Power-of-2 capacity.
    uint32_t             mask_;       ///< capacity_ - 1 (for fast modulo).
    uint32_t             count_;      ///< Number of occupied slots.
    uint64_t             frameNumber_; ///< Current frame counter.

    // ── Helpers ──────────────────────────────────────────────────────────

    /// Compute probe distance for an element at `slot`.
    [[nodiscard]] PULSE_FORCE_INLINE uint32_t probeDistance(uint32_t slot) const noexcept {
        uint32_t ideal = keys_[slot].hash() & mask_;
        return (slot - ideal + capacity_) & mask_;
    }

    /// Backward-shift deletion: remove element at `slot` and shift
    /// subsequent elements backward to fill the gap.
    void backwardShiftDelete(uint32_t slot) noexcept {
        occupied_[slot] = false;
        uint32_t next = (slot + 1) & mask_;

        while (occupied_[next]) {
            uint32_t dist = probeDistance(next);
            if (dist == 0) break; // This element is at its ideal position

            // Shift this element backward
            keys_[slot] = keys_[next];
            manifolds_[slot] = manifolds_[next];
            occupied_[slot] = true;
            occupied_[next] = false;

            slot = next;
            next = (next + 1) & mask_;
        }
    }

    /// Round up to next power of 2.
    static PULSE_FORCE_INLINE uint32_t nextPow2(uint32_t v) noexcept {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    }
};

} // namespace pulse
