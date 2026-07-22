/**
 * @file sap.h
 * @brief Sweep-and-Prune (SAP) broad-phase — single-axis sort.
 *
 * Sorts proxy endpoints along a chosen axis and finds overlapping pairs by
 * scanning for interval overlaps. Excellent for mostly-static scenes where
 * insertion sort is O(n + k) due to nearly-sorted data.
 *
 * This is a 1D SAP (single axis). A 3-axis SAP variant can be added later.
 * The sort axis is chosen by maximum variance of AABB centers.
 *
 * Properties:
 * - O(n + k) incremental update via insertion sort (k = swap inversions)
 * - O(n + p) pair query (p = overlapping pairs)
 * - Worst case with many overlaps = O(n²)
 */

#pragma once

#include "broadphase_common.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <utility> // std::exchange

namespace pulse {

/**
 * @class SAP
 * @brief Single-axis Sweep-and-Prune broad-phase.
 *
 * Stores sorted endpoint list. Each proxy contributes a min and max endpoint
 * on the sort axis. Overlapping pairs are found by scanning left-to-right
 * and tracking active intervals.
 */
class SAP {
public:

    static constexpr uint32_t DefaultCapacity = 256;

    // ── Endpoint entry ─────────────────────────────────────────────────────

    struct Endpoint {
        float       value;       ///< Min or max value on sort axis.
        uint32_t    proxyIndex;  ///< Which proxy this endpoint belongs to.
        bool        isMax;       ///< True = max endpoint, false = min endpoint.
        uint8_t     _pad[3];

        [[nodiscard]] PULSE_FORCE_INLINE bool operator<(const Endpoint& rhs) const noexcept {
            return value < rhs.value;
        }
    };

    // ── Per-proxy data ─────────────────────────────────────────────────────

    struct ProxyEntry {
        AABB     aabb;
        void*    userData;
        bool     active;
        uint8_t  _pad[7];
    };

    // ── Construction ───────────────────────────────────────────────────────

    explicit SAP(uint32_t maxProxies = DefaultCapacity) noexcept
        : proxies_(nullptr), endpoints_(nullptr),
          proxyCount_(0), proxyCapacity_(0),
          endpointCount_(0), sortAxis_(0), dirty_(false)
    {
        proxyCapacity_ = maxProxies;
        proxies_   = static_cast<ProxyEntry*>(std::malloc(maxProxies * sizeof(ProxyEntry)));
        endpoints_ = static_cast<Endpoint*>(std::malloc(maxProxies * 2 * sizeof(Endpoint)));
        assert(proxies_ && endpoints_);
        std::memset(proxies_, 0, maxProxies * sizeof(ProxyEntry));
    }

    ~SAP() noexcept {
        std::free(proxies_);
        std::free(endpoints_);
    }

    // Non-copyable
    SAP(const SAP&) = delete;
    SAP& operator=(const SAP&) = delete;

    // Movable
    SAP(SAP&& other) noexcept
        : proxies_(std::exchange(other.proxies_, nullptr)),
          endpoints_(std::exchange(other.endpoints_, nullptr)),
          proxyCount_(std::exchange(other.proxyCount_, 0u)),
          proxyCapacity_(std::exchange(other.proxyCapacity_, 0u)),
          endpointCount_(std::exchange(other.endpointCount_, 0u)),
          sortAxis_(std::exchange(other.sortAxis_, 0)),
          dirty_(std::exchange(other.dirty_, false))
    {}

    SAP& operator=(SAP&& other) noexcept {
        if (this != &other) {
            std::free(proxies_);
            std::free(endpoints_);
            proxies_       = std::exchange(other.proxies_, nullptr);
            endpoints_     = std::exchange(other.endpoints_, nullptr);
            proxyCount_    = std::exchange(other.proxyCount_, 0u);
            proxyCapacity_ = std::exchange(other.proxyCapacity_, 0u);
            endpointCount_ = std::exchange(other.endpointCount_, 0u);
            sortAxis_      = std::exchange(other.sortAxis_, 0);
            dirty_         = std::exchange(other.dirty_, false);
        }
        return *this;
    }

    // ── Proxy management ───────────────────────────────────────────────────

    /**
     * @brief Add a proxy. Returns a handle (encoded proxy slot index).
     */
    [[nodiscard]] ProxyHandle addProxy(const AABB& fatAABB, void* userData) noexcept {
        uint32_t slot = findFreeSlot();
        ProxyEntry& entry = proxies_[slot];
        entry.aabb     = fatAABB;
        entry.userData = userData;
        entry.active   = true;

        // Use current sort axis (not hardcoded X)
        float mn = getAxisValue(fatAABB.min);
        float mx = getAxisValue(fatAABB.max);

        uint32_t minIdx = endpointCount_++;
        uint32_t maxIdx = endpointCount_++;
        endpoints_[minIdx] = { mn, slot, false, {} };
        endpoints_[maxIdx] = { mx, slot, true,  {} };

        ++proxyCount_;
        dirty_ = true;

        return ProxyHandle(slot);
    }

    /**
     * @brief Remove a proxy.
     */
    void removeProxy(ProxyHandle handle) noexcept {
        uint32_t slot = handle.index();
        assert(slot < proxyCapacity_ && proxies_[slot].active);
        proxies_[slot].active = false;
        --proxyCount_;

        // Remove endpoints belonging to this proxy (compact)
        uint32_t write = 0;
        for (uint32_t i = 0; i < endpointCount_; ++i) {
            if (endpoints_[i].proxyIndex != slot) {
                endpoints_[write++] = endpoints_[i];
            }
        }
        endpointCount_ = write;
        dirty_ = true;
    }

    /**
     * @brief Move a proxy to a new fat AABB.
     */
    void moveProxy(ProxyHandle handle, const AABB& newAABB) noexcept {
        uint32_t slot = handle.index();
        proxies_[slot].aabb = newAABB;

        float newMin = getAxisValue(newAABB.min);
        float newMax = getAxisValue(newAABB.max);

        for (uint32_t i = 0; i < endpointCount_; ++i) {
            if (endpoints_[i].proxyIndex == slot) {
                endpoints_[i].value = endpoints_[i].isMax ? newMax : newMin;
            }
        }
        dirty_ = true;
    }

    /**
     * @brief Recompute sort axis by maximum variance of AABB centers.
     * Should be called periodically (not every frame).
     */
    void updateSortAxis() noexcept {
        if (proxyCount_ < 2) return;

        float mean[3] = {0, 0, 0};
        float var[3]  = {0, 0, 0};
        float n = static_cast<float>(proxyCount_);

        for (uint32_t i = 0; i < proxyCapacity_; ++i) {
            if (!proxies_[i].active) continue;
            Vec3 c = proxies_[i].aabb.center();
            mean[0] += c.getX();
            mean[1] += c.getY();
            mean[2] += c.getZ();
        }
        mean[0] /= n; mean[1] /= n; mean[2] /= n;

        for (uint32_t i = 0; i < proxyCapacity_; ++i) {
            if (!proxies_[i].active) continue;
            Vec3 c = proxies_[i].aabb.center();
            float dx = c.getX() - mean[0];
            float dy = c.getY() - mean[1];
            float dz = c.getZ() - mean[2];
            var[0] += dx * dx;
            var[1] += dy * dy;
            var[2] += dz * dz;
        }

        sortAxis_ = 0;
        if (var[1] > var[0]) sortAxis_ = 1;
        if (var[2] > var[sortAxis_]) sortAxis_ = 2;

        rebuildEndpoints();
        dirty_ = true;
    }

    /**
     * @brief Sort endpoints (using insertion sort for nearly-sorted data) and
     *        compute overlapping pairs.
     * @param out     Output array for overlap pairs.
     * @param maxOut  Capacity of the output array.
     * @return Number of pairs found.
     */
    uint32_t computePairs(OverlapPair* out, uint32_t maxOut) noexcept {
        if (proxyCount_ < 2 || maxOut == 0) return 0;

        // Insertion sort — O(n + k) for nearly-sorted data
        insertionSort(endpoints_, endpointCount_);
        dirty_ = false;

        uint32_t pairCount = 0;

        // Active set: heap-allocated buffer sized to actual proxyCount
        // to avoid stack overflow with large proxy counts
        uint32_t* active = static_cast<uint32_t*>(
            std::malloc(proxyCount_ * sizeof(uint32_t)));
        if (!active) return 0;
        uint32_t activeCount = 0;

        for (uint32_t i = 0; i < endpointCount_ && pairCount < maxOut; ++i) {
            const Endpoint& ep = endpoints_[i];
            if (!proxies_[ep.proxyIndex].active) continue;

            if (!ep.isMax) {
                // Min endpoint — test against all active, add to active set
                for (uint32_t j = 0; j < activeCount && pairCount < maxOut; ++j) {
                    uint32_t other = active[j];
                    if (!proxies_[other].active) continue;

                    // Full 3D AABB overlap test
                    if (proxies_[ep.proxyIndex].aabb.overlaps(proxies_[other].aabb)) {
                        out[pairCount++] = OverlapPair(
                            ProxyHandle(ep.proxyIndex),
                            ProxyHandle(other)
                        );
                    }
                }
                if (activeCount < proxyCount_) {
                    active[activeCount++] = ep.proxyIndex;
                }
            } else {
                // Max endpoint — remove from active set
                for (uint32_t j = 0; j < activeCount; ++j) {
                    if (active[j] == ep.proxyIndex) {
                        active[j] = active[--activeCount];
                        break;
                    }
                }
            }
        }

        std::free(active);
        return pairCount;
    }

    // ── Accessors ──────────────────────────────────────────────────────────

    [[nodiscard]] uint32_t proxyCount()    const noexcept { return proxyCount_; }
    [[nodiscard]] int32_t  sortAxis()      const noexcept { return sortAxis_; }
    [[nodiscard]] const AABB& getAABB(ProxyHandle h) const noexcept {
        return proxies_[h.index()].aabb;
    }

private:
    ProxyEntry* proxies_;
    Endpoint*   endpoints_;
    uint32_t    proxyCount_;
    uint32_t    proxyCapacity_;
    uint32_t    endpointCount_;
    int32_t     sortAxis_;
    bool        dirty_;

    float getAxisValue(Vec3 v) const noexcept {
        if (sortAxis_ == 0) return v.getX();
        if (sortAxis_ == 1) return v.getY();
        return v.getZ();
    }

    uint32_t findFreeSlot() noexcept {
        for (uint32_t i = 0; i < proxyCapacity_; ++i) {
            if (!proxies_[i].active) return i;
        }
        assert(false && "SAP: no free proxy slots");
        return 0;
    }

    void rebuildEndpoints() noexcept {
        endpointCount_ = 0;
        for (uint32_t i = 0; i < proxyCapacity_; ++i) {
            if (!proxies_[i].active) continue;
            float mn = getAxisValue(proxies_[i].aabb.min);
            float mx = getAxisValue(proxies_[i].aabb.max);
            endpoints_[endpointCount_++] = { mn, i, false, {} };
            endpoints_[endpointCount_++] = { mx, i, true,  {} };
        }
    }

    /// Insertion sort — O(n + k) for nearly-sorted data, which is the common
    /// case in SAP since objects move small distances between frames.
    static void insertionSort(Endpoint* arr, uint32_t n) noexcept {
        for (uint32_t i = 1; i < n; ++i) {
            Endpoint key = arr[i];
            int32_t j = static_cast<int32_t>(i) - 1;
            while (j >= 0 && arr[j].value > key.value) {
                arr[j + 1] = arr[j];
                --j;
            }
            arr[j + 1] = key;
        }
    }
};

} // namespace pulse
