/**
 * @file island_manager.h
 * @brief Island detection via union-find — partitions bodies into independent groups.
 *
 * Bodies connected by contacts or constraints are placed in the same island.
 * Each island can be solved independently (enabling parallel dispatch via JobSystem).
 *
 * Algorithm: Disjoint-set forest (union-find) with:
 *  - Path compression in find() — amortised O(α(n)) ≈ O(1).
 *  - Union by rank in unite()   — keeps tree balanced.
 *
 * Usage:
 *  1. Call reset(bodyCount) at the start of each frame.
 *  2. Call unite(bodyA, bodyB) for each contact pair / constraint pair.
 *  3. Call buildIslands() to flatten the union-find into island lists.
 *  4. Query getIslandCount() and getIsland(i) for the solver.
 *
 * Static bodies are excluded from islands (infinite mass, never solved).
 *
 * Thread safety: NOT thread-safe. External synchronization required.
 */

#pragma once

#include <pulse/math/math_common.h>
#include <pulse/utilities/assert.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace pulse {

// ── Island info ──────────────────────────────────────────────────────────────

/**
 * @struct IslandInfo
 * @brief Describes a single simulation island — a group of connected bodies.
 *
 * Bodies in the same island are connected transitively through contacts
 * or constraints and must be solved together.
 */
struct IslandInfo {
    uint32_t* bodyIndices;      ///< Dense body indices in this island.
    uint32_t  bodyCount;        ///< Number of bodies in this island.
    uint32_t  capacity;         ///< Allocated capacity of bodyIndices.
    bool      allSleeping;      ///< True if all bodies in this island are sleeping.

    IslandInfo() noexcept
        : bodyIndices(nullptr), bodyCount(0), capacity(0), allSleeping(true) {}
};

// ── IslandManager ────────────────────────────────────────────────────────────

/**
 * @class IslandManager
 * @brief Union-find based island detection for rigid body simulation.
 */
class IslandManager {
public:
    // ── Construction / Destruction ───────────────────────────────────────

    explicit IslandManager(std::size_t maxBodies = 256) noexcept
        : maxBodies_(maxBodies),
          parent_(nullptr),
          rank_(nullptr),
          islandCount_(0),
          islands_(nullptr),
          islandsCapacity_(0)
    {
        parent_ = static_cast<uint32_t*>(std::malloc(maxBodies_ * sizeof(uint32_t)));
        rank_   = static_cast<uint32_t*>(std::malloc(maxBodies_ * sizeof(uint32_t)));
        PULSE_ASSERT_MSG(parent_ != nullptr, "IslandManager: parent alloc failed");
        PULSE_ASSERT_MSG(rank_ != nullptr, "IslandManager: rank alloc failed");

        // Pre-allocate island array.
        islandsCapacity_ = 64;
        islands_ = static_cast<IslandInfo*>(std::malloc(islandsCapacity_ * sizeof(IslandInfo)));
        PULSE_ASSERT_MSG(islands_ != nullptr, "IslandManager: islands alloc failed");
    }

    ~IslandManager() noexcept {
        freeIslands();
        std::free(parent_);
        std::free(rank_);
        std::free(islands_);
    }

    // Non-copyable.
    IslandManager(const IslandManager&) = delete;
    IslandManager& operator=(const IslandManager&) = delete;

    // ── Reset ────────────────────────────────────────────────────────────

    /// Reset the union-find for a new frame with the given body count.
    /// Each body starts as its own set.
    void reset(std::size_t bodyCount) noexcept {
        if (bodyCount > maxBodies_) {
            growUF(bodyCount);
        }
        bodyCount_ = bodyCount;

        // Make-set: each element is its own parent with rank 0.
        for (std::size_t i = 0; i < bodyCount_; ++i) {
            parent_[i] = static_cast<uint32_t>(i);
            rank_[i] = 0;
        }

        // Free old island body index arrays.
        freeIslands();
        islandCount_ = 0;
    }

    // ── Union-Find operations ────────────────────────────────────────────

    /// Find the root (representative) of the set containing x.
    /// Uses path compression for amortised O(α(n)).
    [[nodiscard]] PULSE_FORCE_INLINE uint32_t find(uint32_t x) noexcept {
        PULSE_ASSERT(x < bodyCount_);
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]]; // Path halving.
            x = parent_[x];
        }
        return x;
    }

    /// Unite the sets containing a and b.
    /// Uses union-by-rank to keep the tree balanced.
    PULSE_FORCE_INLINE void unite(uint32_t a, uint32_t b) noexcept {
        uint32_t rootA = find(a);
        uint32_t rootB = find(b);
        if (rootA == rootB) return;

        if (rank_[rootA] < rank_[rootB]) {
            parent_[rootA] = rootB;
        } else if (rank_[rootA] > rank_[rootB]) {
            parent_[rootB] = rootA;
        } else {
            parent_[rootB] = rootA;
            rank_[rootA]++;
        }
    }

    /// Check if a and b are in the same set.
    [[nodiscard]] PULSE_FORCE_INLINE bool connected(uint32_t a, uint32_t b) noexcept {
        return find(a) == find(b);
    }

    // ── Island building ──────────────────────────────────────────────────

    /**
     * @brief Build island lists from the current union-find state.
     *
     * @param isStaticOrSleeping  Array of booleans, one per body. If true,
     *        the body is static/kinematic and should be excluded from islands
     *        (but can still connect other bodies through contacts).
     *        Pass nullptr to include all bodies.
     *
     * After calling, query getIslandCount() and getIsland(i).
     */
    void buildIslands(const bool* isStaticOrSleeping = nullptr) noexcept {
        // Free previous island data.
        freeIslands();
        islandCount_ = 0;

        if (bodyCount_ == 0) return;

        // Temporary: map root → island index.
        // Using a simple array since roots are in [0, bodyCount_).
        auto* rootToIsland = static_cast<uint32_t*>(std::malloc(bodyCount_ * sizeof(uint32_t)));
        PULSE_ASSERT_MSG(rootToIsland != nullptr, "IslandManager: temp alloc failed");
        std::memset(rootToIsland, 0xFF, bodyCount_ * sizeof(uint32_t));

        // First pass: count bodies per island and assign island IDs.
        for (std::size_t i = 0; i < bodyCount_; ++i) {
            // Skip static/kinematic bodies for island membership.
            if (isStaticOrSleeping && isStaticOrSleeping[i]) continue;

            uint32_t root = find(static_cast<uint32_t>(i));

            if (rootToIsland[root] == 0xFFFFFFFFu) {
                // New island.
                uint32_t islandIdx = islandCount_++;
                rootToIsland[root] = islandIdx;
            }
        }

        // Grow island array if needed.
        if (islandCount_ > islandsCapacity_) {
            std::size_t newCap = islandsCapacity_;
            while (newCap < islandCount_) newCap *= 2;
            islands_ = static_cast<IslandInfo*>(std::realloc(islands_, newCap * sizeof(IslandInfo)));
            PULSE_ASSERT_MSG(islands_ != nullptr, "IslandManager: islands realloc failed");
            islandsCapacity_ = newCap;
        }

        // Initialize island infos with body count = 0.
        for (uint32_t i = 0; i < islandCount_; ++i) {
            islands_[i] = IslandInfo();
        }

        // Count bodies per island.
        for (std::size_t i = 0; i < bodyCount_; ++i) {
            if (isStaticOrSleeping && isStaticOrSleeping[i]) continue;
            uint32_t root = find(static_cast<uint32_t>(i));
            uint32_t islandIdx = rootToIsland[root];
            if (islandIdx != 0xFFFFFFFFu) {
                islands_[islandIdx].bodyCount++;
            }
        }

        // Allocate body index arrays for each island.
        for (uint32_t i = 0; i < islandCount_; ++i) {
            uint32_t count = islands_[i].bodyCount;
            islands_[i].capacity = count;
            if (count > 0) {
                islands_[i].bodyIndices = static_cast<uint32_t*>(
                    std::malloc(count * sizeof(uint32_t)));
                PULSE_ASSERT_MSG(islands_[i].bodyIndices != nullptr,
                    "IslandManager: island body alloc failed");
            }
            islands_[i].bodyCount = 0; // Reset for filling.
        }

        // Second pass: fill body indices into islands.
        for (std::size_t i = 0; i < bodyCount_; ++i) {
            if (isStaticOrSleeping && isStaticOrSleeping[i]) continue;
            uint32_t root = find(static_cast<uint32_t>(i));
            uint32_t islandIdx = rootToIsland[root];
            if (islandIdx != 0xFFFFFFFFu) {
                auto& island = islands_[islandIdx];
                island.bodyIndices[island.bodyCount++] = static_cast<uint32_t>(i);
            }
        }

        std::free(rootToIsland);
    }

    // ── Queries ──────────────────────────────────────────────────────────

    /// Number of islands detected.
    [[nodiscard]] uint32_t getIslandCount() const noexcept { return islandCount_; }

    /// Get island info by index.
    [[nodiscard]] const IslandInfo& getIsland(uint32_t index) const noexcept {
        PULSE_ASSERT(index < islandCount_);
        return islands_[index];
    }

    /// Get mutable island info (for setting allSleeping flag).
    [[nodiscard]] IslandInfo& getIsland(uint32_t index) noexcept {
        PULSE_ASSERT(index < islandCount_);
        return islands_[index];
    }

    /// Number of bodies in the union-find.
    [[nodiscard]] std::size_t bodyCount() const noexcept { return bodyCount_; }

private:
    std::size_t maxBodies_;
    std::size_t bodyCount_ = 0;

    uint32_t* parent_;   ///< Parent array for union-find.
    uint32_t* rank_;     ///< Rank array for union-by-rank.

    uint32_t    islandCount_;
    IslandInfo* islands_;
    std::size_t islandsCapacity_;

    /// Grow the union-find arrays.
    void growUF(std::size_t newSize) noexcept {
        std::size_t newCap = maxBodies_;
        while (newCap < newSize) newCap *= 2;

        parent_ = static_cast<uint32_t*>(std::realloc(parent_, newCap * sizeof(uint32_t)));
        rank_   = static_cast<uint32_t*>(std::realloc(rank_, newCap * sizeof(uint32_t)));
        PULSE_ASSERT_MSG(parent_ != nullptr, "IslandManager: parent realloc failed");
        PULSE_ASSERT_MSG(rank_ != nullptr, "IslandManager: rank realloc failed");

        maxBodies_ = newCap;
    }

    /// Free all island body index arrays.
    void freeIslands() noexcept {
        for (uint32_t i = 0; i < islandCount_; ++i) {
            std::free(islands_[i].bodyIndices);
            islands_[i].bodyIndices = nullptr;
        }
    }
};

} // namespace pulse
