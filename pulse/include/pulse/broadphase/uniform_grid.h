/**
 * @file uniform_grid.h
 * @brief Spatial hash grid broad-phase.
 *
 * Divides space into a 3D grid of uniform cells. Each proxy is inserted into
 * every cell its AABB touches. Overlap pairs are found by checking all pairs
 * of proxies sharing at least one cell.
 *
 * Properties:
 * - O(1) cell lookup via spatial hash
 * - O(k) pair generation (k = proxies sharing a cell)
 * - Excellent for uniform-density scenes (particles, cloth, fluids)
 * - Poor for scenes with large size variation (large objects touch many cells)
 *
 * Hash function: Murmur-inspired spatial hash
 *   h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)
 *
 * Collision resolution: open chaining via linked list nodes from a pool.
 * Pair deduplication: open-addressing hash table with linear probing.
 *
 * The grid is rebuilt each frame — no incremental updates. This is fine for
 * particle systems where every body moves every frame anyway.
 */

#pragma once

#include "broadphase_common.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <utility> // std::exchange

namespace pulse {

/**
 * @class UniformGrid
 * @brief Spatial hash broad-phase for uniform-density scenes.
 *
 * Usage:
 *   1. Call `beginFrame()` to clear the grid.
 *   2. Call `insert(handle, aabb)` for each proxy.
 *   3. Call `computePairs(out, max)` to get all overlapping pairs.
 */
class UniformGrid {
public:

    static constexpr uint32_t DefaultTableSize  = 4096;  ///< Hash table buckets (power of 2).
    static constexpr uint32_t DefaultPoolSize   = 16384; ///< Max cell entries per frame.
    static constexpr float    DefaultCellSize   = 2.0f;  ///< Cell size in world units.

    // ── Cell entry (intrusive linked list) ────────────────────────────────

    struct CellEntry {
        uint32_t proxyIndex; ///< ProxyHandle::raw
        uint32_t next;       ///< Next entry index in bucket, 0xFFFFFFFF = end.
    };

    // ── Construction / Destruction ────────────────────────────────────────

    explicit UniformGrid(float cellSize   = DefaultCellSize,
                         uint32_t tableSize = DefaultTableSize,
                         uint32_t poolSize  = DefaultPoolSize) noexcept
        : cellSize_(cellSize),
          invCellSize_(1.0f / cellSize),
          tableSize_(tableSize),
          tableMask_(tableSize - 1),
          buckets_(nullptr),
          pool_(nullptr),
          poolCount_(0),
          poolCapacity_(poolSize)
    {
        assert((tableSize & (tableSize - 1)) == 0 && "tableSize must be power of 2");
        buckets_ = static_cast<uint32_t*>(std::malloc(tableSize * sizeof(uint32_t)));
        pool_    = static_cast<CellEntry*>(std::malloc(poolSize * sizeof(CellEntry)));
        assert(buckets_ && pool_);
        clearBuckets();
    }

    ~UniformGrid() noexcept {
        std::free(buckets_);
        std::free(pool_);
    }

    // Non-copyable
    UniformGrid(const UniformGrid&) = delete;
    UniformGrid& operator=(const UniformGrid&) = delete;

    // Movable
    UniformGrid(UniformGrid&& other) noexcept
        : cellSize_(other.cellSize_),
          invCellSize_(other.invCellSize_),
          tableSize_(other.tableSize_),
          tableMask_(other.tableMask_),
          buckets_(std::exchange(other.buckets_, nullptr)),
          pool_(std::exchange(other.pool_, nullptr)),
          poolCount_(std::exchange(other.poolCount_, 0u)),
          poolCapacity_(std::exchange(other.poolCapacity_, 0u))
    {}

    UniformGrid& operator=(UniformGrid&& other) noexcept {
        if (this != &other) {
            std::free(buckets_);
            std::free(pool_);
            cellSize_     = other.cellSize_;
            invCellSize_  = other.invCellSize_;
            tableSize_    = other.tableSize_;
            tableMask_    = other.tableMask_;
            buckets_      = std::exchange(other.buckets_, nullptr);
            pool_         = std::exchange(other.pool_, nullptr);
            poolCount_    = std::exchange(other.poolCount_, 0u);
            poolCapacity_ = std::exchange(other.poolCapacity_, 0u);
        }
        return *this;
    }

    // ── Frame lifecycle ───────────────────────────────────────────────────

    /// Reset the grid for a new frame. O(tableSize).
    void beginFrame() noexcept {
        clearBuckets();
        poolCount_ = 0;
    }

    /// Insert a proxy into all cells touched by its AABB.
    void insert(ProxyHandle handle, const AABB& aabb) noexcept {
        int32_t x0, y0, z0, x1, y1, z1;
        worldToCell(aabb.min, x0, y0, z0);
        worldToCell(aabb.max, x1, y1, z1);

        for (int32_t x = x0; x <= x1; ++x) {
            for (int32_t y = y0; y <= y1; ++y) {
                for (int32_t z = z0; z <= z1; ++z) {
                    insertIntoCell(x, y, z, handle.raw);
                }
            }
        }
    }

    /**
     * @brief Query all proxies overlapping a given AABB.
     * @param query      The test AABB.
     * @param proxyAABBs Array of proxy AABBs indexed by proxy slot.
     * @param proxyCount Number of entries in proxyAABBs.
     * @param out        Output array for ProxyHandle results.
     * @param maxOut     Capacity of output array.
     * @return Number of unique overlapping proxies.
     */
    uint32_t queryAABB(const AABB& query,
                       const AABB* proxyAABBs, uint32_t proxyCount,
                       ProxyHandle* out, uint32_t maxOut) const noexcept
    {
        if (maxOut == 0) return 0;

        // Find which cells the query AABB touches
        int32_t x0, y0, z0, x1, y1, z1;
        worldToCell(query.min, x0, y0, z0);
        worldToCell(query.max, x1, y1, z1);

        // Simple seen-bitset for dedup (bounded by proxyCount)
        uint32_t resultCount = 0;

        // Use a small linear dedup for modest proxy counts
        static constexpr uint32_t MaxSeen = 256;
        uint32_t seenBuf[MaxSeen];
        uint32_t seenCount = 0;

        for (int32_t x = x0; x <= x1; ++x) {
            for (int32_t y = y0; y <= y1; ++y) {
                for (int32_t z = z0; z <= z1; ++z) {
                    uint32_t bucket = cellHash(x, y, z);
                    uint32_t idx = buckets_[bucket];
                    while (idx != 0xFFFFFFFFu && resultCount < maxOut) {
                        uint32_t pi = pool_[idx].proxyIndex;
                        // Dedup check
                        bool alreadySeen = false;
                        for (uint32_t s = 0; s < seenCount; ++s) {
                            if (seenBuf[s] == pi) { alreadySeen = true; break; }
                        }
                        if (!alreadySeen && pi < proxyCount &&
                            proxyAABBs[pi].overlaps(query))
                        {
                            out[resultCount++] = ProxyHandle(pi);
                            if (seenCount < MaxSeen) seenBuf[seenCount++] = pi;
                        }
                        idx = pool_[idx].next;
                    }
                }
            }
        }
        return resultCount;
    }

    /**
     * @brief Find all overlapping proxy pairs.
     *
     * For each non-empty bucket, check all pairs of proxies in that bucket
     * for full 3D AABB overlap. Uses an open-addressing hash set with linear
     * probing for robust deduplication (no silent collision overwrites).
     *
     * @param proxyAABBs Array of proxy AABBs indexed by proxy slot.
     * @param proxyCount Number of entries in proxyAABBs.
     * @param out        Output pair array.
     * @param maxOut     Capacity of output array.
     * @return Number of pairs found.
     */
    uint32_t computePairs(const AABB* proxyAABBs, uint32_t proxyCount,
                          OverlapPair* out, uint32_t maxOut) const noexcept {
        if (maxOut == 0) return 0;

        uint32_t pairCount = 0;

        // Open-addressing hash set with linear probing for dedup
        static constexpr uint32_t DedupeCapacity = 8192;
        static constexpr uint64_t EmptySlot = 0xFFFFFFFFFFFFFFFFull;
        uint64_t* dedup = static_cast<uint64_t*>(std::malloc(DedupeCapacity * sizeof(uint64_t)));
        if (!dedup) return 0;
        std::memset(dedup, 0xFF, DedupeCapacity * sizeof(uint64_t));

        for (uint32_t bucket = 0; bucket < tableSize_; ++bucket) {
            uint32_t idx = buckets_[bucket];
            if (idx == 0xFFFFFFFFu) continue;

            // Collect proxies in this bucket (bounded)
            uint32_t inBucket[64];
            uint32_t bucketCount = 0;
            while (idx != 0xFFFFFFFFu && bucketCount < 64) {
                inBucket[bucketCount++] = pool_[idx].proxyIndex;
                idx = pool_[idx].next;
            }

            // Check all pairs in bucket
            for (uint32_t i = 0; i < bucketCount && pairCount < maxOut; ++i) {
                for (uint32_t j = i + 1; j < bucketCount && pairCount < maxOut; ++j) {
                    uint32_t a = inBucket[i];
                    uint32_t b = inBucket[j];
                    if (a == b) continue;

                    // Canonical pair (lower index first)
                    if (a > b) { uint32_t t = a; a = b; b = t; }

                    // Deduplicate with linear probing
                    uint64_t key = (static_cast<uint64_t>(a) << 32) | b;
                    uint32_t slot = hashPair(a, b) & (DedupeCapacity - 1);
                    bool isDuplicate = false;

                    for (uint32_t probe = 0; probe < 32; ++probe) {
                        uint32_t s = (slot + probe) & (DedupeCapacity - 1);
                        if (dedup[s] == key) {
                            isDuplicate = true;
                            break;
                        }
                        if (dedup[s] == EmptySlot) {
                            dedup[s] = key; // Insert
                            break;
                        }
                        // Collision — continue probing
                    }

                    if (isDuplicate) continue;

                    // Full AABB overlap test
                    if (a < proxyCount && b < proxyCount &&
                        proxyAABBs[a].overlaps(proxyAABBs[b]))
                    {
                        out[pairCount++] = OverlapPair(ProxyHandle(a), ProxyHandle(b));
                    }
                }
            }
        }

        std::free(dedup);
        return pairCount;
    }

    // ── Configuration ─────────────────────────────────────────────────────

    [[nodiscard]] float    cellSize()   const noexcept { return cellSize_; }
    [[nodiscard]] uint32_t tableSize()  const noexcept { return tableSize_; }
    [[nodiscard]] uint32_t poolCount()  const noexcept { return poolCount_; }

    /// Change cell size (takes effect next beginFrame / insert cycle).
    void setCellSize(float s) noexcept {
        assert(s > 0.0f);
        cellSize_    = s;
        invCellSize_ = 1.0f / s;
    }

private:
    float    cellSize_;
    float    invCellSize_;
    uint32_t tableSize_;
    uint32_t tableMask_;

    uint32_t*  buckets_;
    CellEntry* pool_;
    uint32_t   poolCount_;
    uint32_t   poolCapacity_;

    void clearBuckets() noexcept {
        std::memset(buckets_, 0xFF, tableSize_ * sizeof(uint32_t));
    }

    void worldToCell(Vec3 p, int32_t& cx, int32_t& cy, int32_t& cz) const noexcept {
        cx = static_cast<int32_t>(std::floor(p.getX() * invCellSize_));
        cy = static_cast<int32_t>(std::floor(p.getY() * invCellSize_));
        cz = static_cast<int32_t>(std::floor(p.getZ() * invCellSize_));
    }

    [[nodiscard]] uint32_t cellHash(int32_t x, int32_t y, int32_t z) const noexcept {
        uint32_t ux = static_cast<uint32_t>(x);
        uint32_t uy = static_cast<uint32_t>(y);
        uint32_t uz = static_cast<uint32_t>(z);
        return ((ux * 73856093u) ^ (uy * 19349663u) ^ (uz * 83492791u)) & tableMask_;
    }

    [[nodiscard]] static uint32_t hashPair(uint32_t a, uint32_t b) noexcept {
        // Szudzik pairing for dedup hash
        return (a >= b) ? (a * a + a + b) : (a + b * b);
    }

    void insertIntoCell(int32_t x, int32_t y, int32_t z, uint32_t proxyRaw) noexcept {
        if (poolCount_ >= poolCapacity_) return;

        uint32_t bucket = cellHash(x, y, z);
        uint32_t entryIdx = poolCount_++;

        pool_[entryIdx].proxyIndex = proxyRaw;
        pool_[entryIdx].next       = buckets_[bucket];
        buckets_[bucket]           = entryIdx;
    }
};

} // namespace pulse
