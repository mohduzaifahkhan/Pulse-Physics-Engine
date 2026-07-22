/**
 * @file bvh.h
 * @brief Static Bounding Volume Hierarchy — top-down SAH build.
 *
 * Builds a static BVH from a set of leaf AABBs using the Surface Area
 * Heuristic (SAH). Intended for immovable geometry (environment, TriMesh
 * acceleration structure). Cannot be updated incrementally — rebuild from
 * scratch when geometry changes.
 *
 * Build algorithm:
 *   1. Compute bounding box of all current leaf centroids.
 *   2. Choose split axis = longest axis of centroid bounds.
 *   3. Evaluate SAH cost for 8 evenly spaced split planes on that axis.
 *   4. Pick the minimum-cost split and partition leaves.
 *   5. Recurse on each half until leaf threshold is reached.
 *   6. Fallback: median split when SAH fails (all centroids on one side).
 *
 * Properties:
 * - O(n log²n) build (sorting per level)
 * - O(log n) best-case query, O(n) worst
 * - Read-only after build — no add/remove
 * - Ordered ray traversal (near child first) for fast early termination
 *
 * SAH cost formula:
 *   C(split) = C_traversal + SA(left)/SA(root) * N_left
 *                          + SA(right)/SA(root) * N_right
 */

#pragma once

#include "broadphase_common.h"
#include <pulse/math/ray.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <utility> // std::exchange

namespace pulse {

// ── BVH Node ──────────────────────────────────────────────────────────────────

/**
 * @struct BVHNode
 * @brief A node in the static BVH.
 *
 * Leaf: leftChild = -1, firstLeaf / leafCount give the range in the leaf array.
 * Internal: leftChild >= 0, rightChild = leftChild + 1 (always paired).
 */
struct BVHNode {
    AABB    aabb;        ///< Tight AABB covering this subtree.
    int32_t leftChild;  ///< Left child index. -1 if leaf.
    union {
        int32_t rightChild;  ///< Right child (= leftChild + 1 by construction).
        int32_t firstLeaf;   ///< First leaf index in leaves_ array (leaf only).
    };
    uint32_t leafCount; ///< Number of leaves in this node (0 for internal).

    [[nodiscard]] PULSE_FORCE_INLINE bool isLeaf() const noexcept { return leftChild == -1; }
};

// ── BVH ───────────────────────────────────────────────────────────────────────

/**
 * @class BVH
 * @brief Static BVH for immovable geometry queries.
 *
 * After `build()`, supports:
 * - `queryAABB(aabb, out, max)` — find all leaves overlapping the query
 * - `raycast(origin, invDir, tMax, out, max)` — ordered ray traversal
 */
class BVH {
public:

    static constexpr uint32_t LeafThreshold    = 4;  ///< Max leaves per leaf node.
    static constexpr uint32_t SahBinCount      = 8;  ///< SAH split candidates per axis.
    static constexpr float    SahTraversalCost = 1.0f;
    static constexpr float    SahLeafCost      = 1.0f;

    // ── Construction ──────────────────────────────────────────────────────

    BVH() noexcept
        : nodes_(nullptr), nodeCount_(0), nodeCapacity_(0),
          leaves_(nullptr), leafCount_(0)
    {}

    ~BVH() noexcept {
        std::free(nodes_);
        std::free(leaves_);
    }

    // Non-copyable
    BVH(const BVH&) = delete;
    BVH& operator=(const BVH&) = delete;

    // Movable
    BVH(BVH&& other) noexcept
        : nodes_(std::exchange(other.nodes_, nullptr)),
          nodeCount_(std::exchange(other.nodeCount_, 0u)),
          nodeCapacity_(std::exchange(other.nodeCapacity_, 0u)),
          leaves_(std::exchange(other.leaves_, nullptr)),
          leafCount_(std::exchange(other.leafCount_, 0u))
    {}

    BVH& operator=(BVH&& other) noexcept {
        if (this != &other) {
            std::free(nodes_);
            std::free(leaves_);
            nodes_        = std::exchange(other.nodes_, nullptr);
            nodeCount_    = std::exchange(other.nodeCount_, 0u);
            nodeCapacity_ = std::exchange(other.nodeCapacity_, 0u);
            leaves_       = std::exchange(other.leaves_, nullptr);
            leafCount_    = std::exchange(other.leafCount_, 0u);
        }
        return *this;
    }

    // ── Build ─────────────────────────────────────────────────────────────

    /**
     * @brief Build the BVH from an array of leaf AABBs.
     *
     * @param leafAABBs  Per-leaf bounding boxes.
     * @param count      Number of leaves.
     *
     * After build, the i-th entry in the sorted leaves_ array maps to the
     * original index via `leaves_[i]`.
     */
    void build(const AABB* leafAABBs, uint32_t count) noexcept {
        if (count == 0) return;

        leafCount_ = count;
        leaves_ = static_cast<uint32_t*>(std::realloc(leaves_, count * sizeof(uint32_t)));
        for (uint32_t i = 0; i < count; ++i) leaves_[i] = i;

        uint32_t maxNodes = 2 * count;
        nodeCapacity_ = maxNodes;
        nodes_ = static_cast<BVHNode*>(std::realloc(nodes_, maxNodes * sizeof(BVHNode)));
        nodeCount_ = 0;

        buildRecursive(leafAABBs, leaves_, count, 0, count);
    }

    // ── Queries ───────────────────────────────────────────────────────────

    /**
     * @brief Find all leaves whose AABB overlaps the query AABB.
     * @param query   Test AABB.
     * @param out     Output array of original leaf indices.
     * @param maxOut  Capacity of output array.
     * @return Number of results.
     */
    uint32_t queryAABB(const AABB& query, uint32_t* out, uint32_t maxOut) const noexcept {
        if (nodeCount_ == 0 || maxOut == 0) return 0;

        uint32_t count = 0;
        int32_t stack[256];
        int32_t top = 0;
        stack[top++] = 0;

        while (top > 0 && count < maxOut) {
            int32_t idx = stack[--top];
            const BVHNode& node = nodes_[idx];

            if (!node.aabb.overlaps(query)) continue;

            if (node.isLeaf()) {
                for (uint32_t i = 0; i < node.leafCount && count < maxOut; ++i) {
                    out[count++] = leaves_[node.firstLeaf + i];
                }
            } else {
                assert(top < 254 && "BVH: queryAABB stack overflow");
                stack[top++] = node.leftChild;
                stack[top++] = node.rightChild;
            }
        }
        return count;
    }

    /**
     * @brief Ray cast against the BVH with ordered traversal.
     *
     * Tests the near child (lower tMin) first for early termination.
     *
     * @param origin   Ray origin.
     * @param invDir   Pre-computed reciprocal of ray direction.
     * @param tMax     Maximum ray distance.
     * @param out      Output array of original leaf indices.
     * @param maxOut   Capacity of output array.
     * @return Number of hit leaves.
     */
    uint32_t raycast(Vec3 origin, Vec3 invDir, float tMax,
                     uint32_t* out, uint32_t maxOut) const noexcept {
        if (nodeCount_ == 0 || maxOut == 0) return 0;

        uint32_t count = 0;
        int32_t stack[256];
        int32_t top = 0;
        stack[top++] = 0;

        while (top > 0 && count < maxOut) {
            int32_t idx = stack[--top];
            const BVHNode& node = nodes_[idx];

            float tMin, t;
            if (!node.aabb.rayIntersect(origin, invDir, tMin, t)) continue;
            if (tMin > tMax) continue;

            if (node.isLeaf()) {
                for (uint32_t i = 0; i < node.leafCount && count < maxOut; ++i) {
                    out[count++] = leaves_[node.firstLeaf + i];
                }
            } else {
                assert(top < 254 && "BVH: raycast stack overflow");

                // Ordered traversal: visit near child first (push far child first so
                // near child pops first from the stack)
                const BVHNode& left  = nodes_[node.leftChild];
                const BVHNode& right = nodes_[node.rightChild];

                float tMinL, tL, tMinR, tR;
                bool hitL = left.aabb.rayIntersect(origin, invDir, tMinL, tL);
                bool hitR = right.aabb.rayIntersect(origin, invDir, tMinR, tR);

                if (hitL && hitR) {
                    // Push far child first (so near child pops first)
                    if (tMinL <= tMinR) {
                        stack[top++] = node.rightChild; // far
                        stack[top++] = node.leftChild;  // near — pops first
                    } else {
                        stack[top++] = node.leftChild;  // far
                        stack[top++] = node.rightChild; // near — pops first
                    }
                } else if (hitL) {
                    stack[top++] = node.leftChild;
                } else if (hitR) {
                    stack[top++] = node.rightChild;
                }
            }
        }
        return count;
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    [[nodiscard]] uint32_t nodeCount()   const noexcept { return nodeCount_; }
    [[nodiscard]] uint32_t leafCount()   const noexcept { return leafCount_; }
    [[nodiscard]] bool     isEmpty()     const noexcept { return nodeCount_ == 0; }
    [[nodiscard]] const AABB& rootAABB() const noexcept { return nodes_[0].aabb; }

    /// Height of the tree (for quality metrics).
    [[nodiscard]] int32_t height() const noexcept {
        if (nodeCount_ == 0) return 0;
        return computeHeight(0);
    }

private:
    BVHNode*  nodes_;
    uint32_t  nodeCount_;
    uint32_t  nodeCapacity_;
    uint32_t* leaves_;
    uint32_t  leafCount_;

    // ── Recursive build ───────────────────────────────────────────────────

    int32_t buildRecursive(const AABB* leafAABBs,
                            uint32_t* indices,
                            uint32_t  totalLeaves,
                            uint32_t  start,
                            uint32_t  end) noexcept
    {
        assert(nodeCount_ < nodeCapacity_);
        int32_t nodeIdx = static_cast<int32_t>(nodeCount_++);
        BVHNode& node = nodes_[nodeIdx];
        uint32_t count = end - start;

        // Compute tight AABB for this range
        node.aabb = leafAABBs[indices[start]];
        for (uint32_t i = start + 1; i < end; ++i) {
            node.aabb = node.aabb.merged(leafAABBs[indices[i]]);
        }

        // Leaf condition
        if (count <= LeafThreshold) {
            node.leftChild = -1;
            node.firstLeaf = static_cast<int32_t>(start);
            node.leafCount = count;
            return nodeIdx;
        }

        // Internal node — find split
        uint32_t splitIdx = findSplit(leafAABBs, indices, start, end, node.aabb);

        // Partition around splitIdx
        node.leftChild  = buildRecursive(leafAABBs, indices, totalLeaves, start, splitIdx);
        node.rightChild = buildRecursive(leafAABBs, indices, totalLeaves, splitIdx, end);
        node.leafCount  = 0;

        return nodeIdx;
    }

    /**
     * @brief Find SAH-optimal split position. Returns the split index in [start+1, end-1].
     *
     * Key correctness fix: evaluates all bins on all axes WITHOUT modifying
     * the indices array. Only performs the final partition once the best
     * axis/bin is known. This prevents index corruption across axis iterations.
     */
    uint32_t findSplit(const AABB* leafAABBs, uint32_t* indices,
                        uint32_t start, uint32_t end, const AABB& nodeBounds) noexcept
    {
        uint32_t count = end - start;
        float rootSA = nodeBounds.surfaceArea();
        if (rootSA <= 0.0f) {
            return start + count / 2;
        }

        float bestCost = static_cast<float>(count); // Cost of no-split (leaf node)
        int32_t bestAxis   = -1;
        float   bestPlane  = 0.0f;

        // Evaluate SAH on all 3 axes, all bins — read-only on indices
        for (int32_t axis = 0; axis < 3; ++axis) {
            float cMin =  1e30f, cMax = -1e30f;
            for (uint32_t i = start; i < end; ++i) {
                float v = getAxisCenter(leafAABBs[indices[i]], axis);
                if (v < cMin) cMin = v;
                if (v > cMax) cMax = v;
            }
            if (cMax - cMin < 1e-6f) continue;

            float binSize = (cMax - cMin) / static_cast<float>(SahBinCount);
            for (uint32_t b = 1; b < SahBinCount; ++b) {
                float splitPlane = cMin + static_cast<float>(b) * binSize;

                AABB leftBox, rightBox;
                uint32_t leftCount = 0, rightCount = 0;
                bool leftInit = false, rightInit = false;

                for (uint32_t i = start; i < end; ++i) {
                    float v = getAxisCenter(leafAABBs[indices[i]], axis);
                    if (v <= splitPlane) {
                        leftBox  = leftInit  ? leftBox.merged(leafAABBs[indices[i]])  : leafAABBs[indices[i]];
                        leftInit = true;
                        ++leftCount;
                    } else {
                        rightBox = rightInit ? rightBox.merged(leafAABBs[indices[i]]) : leafAABBs[indices[i]];
                        rightInit = true;
                        ++rightCount;
                    }
                }

                if (leftCount == 0 || rightCount == 0) continue;

                float cost = SahTraversalCost
                           + (leftBox.surfaceArea()  / rootSA) * static_cast<float>(leftCount)  * SahLeafCost
                           + (rightBox.surfaceArea() / rootSA) * static_cast<float>(rightCount) * SahLeafCost;

                if (cost < bestCost) {
                    bestCost  = cost;
                    bestAxis  = axis;
                    bestPlane = splitPlane;
                }
            }
        }

        if (bestAxis == -1) {
            return start + count / 2; // All axes degenerate — median
        }

        // Single final partition on the chosen axis/plane
        uint32_t write = start;
        for (uint32_t i = start; i < end; ++i) {
            float v = getAxisCenter(leafAABBs[indices[i]], bestAxis);
            if (v <= bestPlane) {
                std::swap(indices[write++], indices[i]);
            }
        }

        // Fallback if partition is degenerate
        if (write == start || write == end) {
            write = start + count / 2;
        }

        return write;
    }

    /// Get the centroid value on a given axis (0=X, 1=Y, 2=Z).
    static float getAxisCenter(const AABB& box, int32_t axis) noexcept {
        Vec3 c = box.center();
        if (axis == 0) return c.getX();
        if (axis == 1) return c.getY();
        return c.getZ();
    }

    int32_t computeHeight(int32_t idx) const noexcept {
        if (idx < 0 || static_cast<uint32_t>(idx) >= nodeCount_) return 0;
        const BVHNode& node = nodes_[idx];
        if (node.isLeaf()) return 1;
        int32_t l = computeHeight(node.leftChild);
        int32_t r = computeHeight(node.rightChild);
        return 1 + (l > r ? l : r);
    }
};

} // namespace pulse
