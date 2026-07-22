/**
 * @file dynamic_aabb_tree.h
 * @brief Dynamic AABB Tree — self-balancing BVH for broad-phase collision.
 *
 * Implements a dynamic BVH (Bounding Volume Hierarchy) using the Surface Area
 * Heuristic (SAH) for insertion and AVL-style rotations for balance. This is
 * the primary broad-phase structure — best general-purpose choice for scenes
 * with mixed static/dynamic objects.
 *
 * Algorithm origin: Catto (Box2D b2DynamicTree), Bullet btDbvt.
 *
 * Properties:
 * - O(log n) insert, remove, move
 * - O(log n) AABB and ray queries
 * - O(n log n) full pair enumeration via recursive self-query
 * - Fat AABBs: proxies only move if body exits the fat AABB
 *
 * Memory layout: Pool of fixed-size Node slots, indexed by int32_t.
 * Freed nodes form an intrusive free list via child[0].
 *
 * Node count: up to maxProxies * 2 - 1 internal nodes + maxProxies leaves.
 */

#pragma once

#include "broadphase_common.h"
#include <pulse/math/ray.h>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cstdlib>   // malloc/free/realloc
#include <algorithm> // std::swap
#include <utility>   // std::exchange

namespace pulse {

// ── Tree node ─────────────────────────────────────────────────────────────────

/**
 * @struct DAABBTNode
 * @brief Internal node of the Dynamic AABB Tree.
 *
 * Leaf nodes store a ProxyHandle in proxyIndex and caller data in userData.
 * Internal nodes have proxyIndex = 0xFFFFFFFF.
 */
struct PULSE_SIMD_ALIGN DAABBTNode {
    AABB    aabb;          ///< Fat AABB covering this subtree.
    int32_t parent;        ///< Parent node index. -1 = root.
    int32_t child[2];      ///< Children. -1 = none (leaf if both are -1).
    int32_t height;        ///< Subtree height. 0 = leaf, -1 = free slot.
    uint32_t proxyIndex;   ///< Leaf node index (= ProxyHandle::raw) for leaves; 0xFFFFFFFF for internal.
    uint32_t userData;     ///< Caller-supplied data (body index etc.) for leaves.
    uint32_t _pad;         ///< Alignment padding.

    [[nodiscard]] PULSE_FORCE_INLINE bool isLeaf()    const noexcept { return child[0] == -1; }
    [[nodiscard]] PULSE_FORCE_INLINE bool isFree()    const noexcept { return height == -1; }
    [[nodiscard]] PULSE_FORCE_INLINE ProxyHandle proxy() const noexcept { return ProxyHandle(proxyIndex); }
};

// ── Dynamic AABB Tree ─────────────────────────────────────────────────────────

/**
 * @class DynamicAABBTree
 * @brief General-purpose dynamic BVH for broad-phase collision detection.
 *
 * Owns all node memory. Proxies are returned as ProxyHandle values (which
 * encode the leaf node index directly for O(1) leaf access).
 */
class DynamicAABBTree {
public:

    static constexpr int32_t  NullNode = -1;
    static constexpr uint32_t DefaultCapacity = 256;

    // ── Construction / Destruction ────────────────────────────────────────

    explicit DynamicAABBTree(uint32_t initialCapacity = DefaultCapacity) noexcept
        : nodes_(nullptr), nodeCount_(0), nodeCapacity_(0),
          freeList_(NullNode), root_(NullNode), proxyCount_(0)
    {
        grow(initialCapacity);
    }

    ~DynamicAABBTree() noexcept {
        std::free(nodes_);
    }

    // Non-copyable
    DynamicAABBTree(const DynamicAABBTree&) = delete;
    DynamicAABBTree& operator=(const DynamicAABBTree&) = delete;

    // Movable
    DynamicAABBTree(DynamicAABBTree&& other) noexcept
        : nodes_(std::exchange(other.nodes_, nullptr)),
          nodeCount_(std::exchange(other.nodeCount_, 0)),
          nodeCapacity_(std::exchange(other.nodeCapacity_, 0)),
          freeList_(std::exchange(other.freeList_, -1)),
          root_(std::exchange(other.root_, -1)),
          proxyCount_(std::exchange(other.proxyCount_, 0u))
    {}

    DynamicAABBTree& operator=(DynamicAABBTree&& other) noexcept {
        if (this != &other) {
            std::free(nodes_);
            nodes_       = std::exchange(other.nodes_, nullptr);
            nodeCount_   = std::exchange(other.nodeCount_, 0);
            nodeCapacity_= std::exchange(other.nodeCapacity_, 0);
            freeList_    = std::exchange(other.freeList_, -1);
            root_        = std::exchange(other.root_, -1);
            proxyCount_  = std::exchange(other.proxyCount_, 0u);
        }
        return *this;
    }

    // ── Proxy management ──────────────────────────────────────────────────

    /**
     * @brief Create a new proxy for the given fat AABB.
     * @param fatAABB Pre-inflated AABB (caller does the inflation).
     * @param userData Opaque caller data (e.g., body index).
     * @return Handle to the new proxy (equals the leaf node index).
     */
    [[nodiscard]] ProxyHandle createProxy(const AABB& fatAABB, uint32_t userData) noexcept {
        int32_t leafIdx = allocNode();
        DAABBTNode& leaf = nodes_[leafIdx];
        leaf.aabb        = fatAABB;
        leaf.proxyIndex  = static_cast<uint32_t>(leafIdx);
        leaf.userData    = userData;
        leaf.height      = 0;
        leaf.child[0]    = NullNode;
        leaf.child[1]    = NullNode;
        leaf.parent      = NullNode;

        insertLeaf(leafIdx);
        ++proxyCount_;

        return ProxyHandle(static_cast<uint32_t>(leafIdx));
    }

    /**
     * @brief Destroy a proxy (removes from tree).
     */
    void destroyProxy(ProxyHandle handle) noexcept {
        int32_t leafIdx = static_cast<int32_t>(handle.index());
        assert(leafIdx >= 0 && leafIdx < nodeCapacity_);
        assert(nodes_[leafIdx].isLeaf());

        removeLeaf(leafIdx);
        freeNode(leafIdx);
        --proxyCount_;
    }

    /**
     * @brief Move a proxy to a new fat AABB.
     * @return True if the proxy was reinserted (AABB changed), false if the
     *         body was still inside the old fat AABB (no work done).
     */
    bool moveProxy(ProxyHandle handle, const AABB& newAABB) noexcept {
        int32_t leafIdx = static_cast<int32_t>(handle.index());
        assert(leafIdx >= 0 && leafIdx < nodeCapacity_);
        assert(nodes_[leafIdx].isLeaf());

        // Only reinsert if the new AABB has left the stored fat AABB
        if (nodes_[leafIdx].aabb.contains(newAABB)) {
            return false;
        }

        removeLeaf(leafIdx);
        nodes_[leafIdx].aabb = newAABB;
        insertLeaf(leafIdx);
        return true;
    }

    /**
     * @brief Move a proxy with displacement prediction.
     *
     * Computes a new fat AABB from the tight AABB, inflated by fatMargin and
     * extended in the displacement direction (via AABB::swept). Only reinserts
     * if the tight AABB has escaped the old fat AABB.
     *
     * @param handle       Proxy to move.
     * @param tightAABB    The body's current tight (un-inflated) AABB.
     * @param displacement Movement vector since last frame.
     * @param fatMargin    Inflation margin (e.g., 0.1f).
     * @return True if reinserted.
     */
    bool moveProxy(ProxyHandle handle, const AABB& tightAABB,
                   Vec3 displacement, float fatMargin) noexcept
    {
        int32_t leafIdx = static_cast<int32_t>(handle.index());
        assert(leafIdx >= 0 && leafIdx < nodeCapacity_);
        assert(nodes_[leafIdx].isLeaf());

        // If tight AABB is still inside the old fat AABB, skip
        if (nodes_[leafIdx].aabb.contains(tightAABB)) {
            return false;
        }

        // Build new fat AABB: inflate + sweep in displacement direction
        AABB newFat = tightAABB.expanded(fatMargin).swept(displacement);

        removeLeaf(leafIdx);
        nodes_[leafIdx].aabb = newFat;
        insertLeaf(leafIdx);
        return true;
    }

    /**
     * @brief Update the fat AABB of a proxy unconditionally.
     */
    void setAABB(ProxyHandle handle, const AABB& fatAABB) noexcept {
        int32_t leafIdx = static_cast<int32_t>(handle.index());
        assert(leafIdx >= 0 && leafIdx < nodeCapacity_);
        assert(nodes_[leafIdx].isLeaf());

        removeLeaf(leafIdx);
        nodes_[leafIdx].aabb = fatAABB;
        insertLeaf(leafIdx);
    }

    /**
     * @brief Get the stored fat AABB for a proxy.
     */
    [[nodiscard]] const AABB& getAABB(ProxyHandle handle) const noexcept {
        return nodes_[handle.index()].aabb;
    }

    /**
     * @brief Get the caller-supplied userData for a proxy.
     */
    [[nodiscard]] uint32_t getUserData(ProxyHandle handle) const noexcept {
        return nodes_[handle.index()].userData;
    }

    // ── Queries ───────────────────────────────────────────────────────────

    /**
     * @brief Enumerate all proxies whose fat AABB overlaps the query AABB.
     * @param query     The test AABB.
     * @param out       Output array for overlapping ProxyHandles.
     * @param maxOut    Capacity of the output array.
     * @return Number of results written.
     */
    uint32_t queryAABB(const AABB& query, ProxyHandle* out, uint32_t maxOut) const noexcept {
        if (root_ == NullNode || maxOut == 0) return 0;

        uint32_t count = 0;
        int32_t stack[256];
        int32_t stackTop = 0;
        stack[stackTop++] = root_;

        while (stackTop > 0 && count < maxOut) {
            int32_t nodeIdx = stack[--stackTop];
            if (nodeIdx == NullNode) continue;

            const DAABBTNode& node = nodes_[nodeIdx];
            if (!node.aabb.overlaps(query)) continue;

            if (node.isLeaf()) {
                out[count++] = node.proxy();
            } else {
                assert(stackTop < 254 && "DynamicAABBTree: queryAABB stack overflow");
                stack[stackTop++] = node.child[0];
                stack[stackTop++] = node.child[1];
            }
        }
        return count;
    }

    /**
     * @brief Enumerate all proxies within a sphere.
     * @param center    Sphere center.
     * @param radius    Sphere radius.
     * @param out       Output array for overlapping ProxyHandles.
     * @param maxOut    Capacity of the output array.
     * @return Number of results written.
     */
    uint32_t querySphere(Vec3 center, float radius, ProxyHandle* out, uint32_t maxOut) const noexcept {
        if (root_ == NullNode || maxOut == 0) return 0;

        float radiusSq = radius * radius;
        uint32_t count = 0;
        int32_t stack[256];
        int32_t stackTop = 0;
        stack[stackTop++] = root_;

        while (stackTop > 0 && count < maxOut) {
            int32_t nodeIdx = stack[--stackTop];
            if (nodeIdx == NullNode) continue;

            const DAABBTNode& node = nodes_[nodeIdx];

            // Check if the sphere overlaps the node AABB
            if (node.aabb.distanceSqTo(center) > radiusSq) continue;

            if (node.isLeaf()) {
                out[count++] = node.proxy();
            } else {
                assert(stackTop < 254 && "DynamicAABBTree: querySphere stack overflow");
                stack[stackTop++] = node.child[0];
                stack[stackTop++] = node.child[1];
            }
        }
        return count;
    }

    /**
     * @brief Enumerate all overlapping proxy pairs using recursive tree self-query.
     *
     * Descends both subtrees simultaneously, pruning when two internal AABBs
     * don't overlap. O(n log n) for well-separated distributions.
     *
     * @param out     Output array for overlap pairs.
     * @param maxOut  Capacity of the output array.
     * @return Number of pairs written.
     */
    uint32_t computePairs(OverlapPair* out, uint32_t maxOut) const noexcept {
        if (root_ == NullNode || proxyCount_ < 2 || maxOut == 0) return 0;

        uint32_t pairCount = 0;
        selfQuery(root_, root_, out, maxOut, pairCount);
        return pairCount;
    }

    /**
     * @brief Ray cast against all proxies.
     * @param origin    Ray origin.
     * @param direction Ray direction (need not be normalized).
     * @param tMax      Maximum ray distance.
     * @param out       Output array for hit proxy handles.
     * @param maxOut    Capacity of the output array.
     * @return Number of hit proxies.
     */
    uint32_t raycast(Vec3 origin, Vec3 direction, float tMax,
                     ProxyHandle* out, uint32_t maxOut) const noexcept {
        if (root_ == NullNode || maxOut == 0) return 0;

        Vec3 invDir(
            1.0f / direction.getX(),
            1.0f / direction.getY(),
            1.0f / direction.getZ()
        );

        uint32_t count = 0;
        int32_t stack[256];
        int32_t stackTop = 0;
        stack[stackTop++] = root_;

        while (stackTop > 0 && count < maxOut) {
            int32_t nodeIdx = stack[--stackTop];
            if (nodeIdx == NullNode) continue;

            const DAABBTNode& node = nodes_[nodeIdx];
            float tMin, t;
            if (!node.aabb.rayIntersect(origin, invDir, tMin, t)) continue;
            if (tMin > tMax) continue;

            if (node.isLeaf()) {
                out[count++] = node.proxy();
            } else {
                assert(stackTop < 254 && "DynamicAABBTree: raycast stack overflow");
                stack[stackTop++] = node.child[0];
                stack[stackTop++] = node.child[1];
            }
        }
        return count;
    }

    // ── Validation (debug) ────────────────────────────────────────────────

    /**
     * @brief Validate all tree invariants. Returns true if the tree is consistent.
     *
     * Checks:
     * - Parent-child bidirectional linkage
     * - Height consistency
     * - Internal node AABBs contain both children
     * - Leaves have no children
     * - No cycles (bounded by nodeCount)
     */
    [[nodiscard]] bool validate() const noexcept {
        if (root_ == NullNode) return proxyCount_ == 0;
        return validateNode(root_, NullNode);
    }

    // ── Statistics ────────────────────────────────────────────────────────

    [[nodiscard]] uint32_t proxyCount()    const noexcept { return proxyCount_; }
    [[nodiscard]] int32_t  nodeCount()     const noexcept { return nodeCount_; }
    [[nodiscard]] int32_t  nodeCapacity()  const noexcept { return nodeCapacity_; }
    [[nodiscard]] int32_t  root()          const noexcept { return root_; }
    [[nodiscard]] int32_t  treeHeight()    const noexcept {
        return root_ == NullNode ? 0 : nodes_[root_].height;
    }

    /// Compute the total surface area of all internal nodes (quality metric).
    [[nodiscard]] float computeTotalSurfaceArea() const noexcept {
        float total = 0.0f;
        for (int32_t i = 0; i < nodeCapacity_; ++i) {
            const DAABBTNode& n = nodes_[i];
            if (!n.isFree()) total += n.aabb.surfaceArea();
        }
        return total;
    }

private:
    DAABBTNode* nodes_;
    int32_t     nodeCount_;
    int32_t     nodeCapacity_;
    int32_t     freeList_;
    int32_t     root_;
    uint32_t    proxyCount_;

    // ── Node pool management ──────────────────────────────────────────────

    void grow(int32_t newCapacity) noexcept {
        if (newCapacity <= nodeCapacity_) return;

        DAABBTNode* newNodes = static_cast<DAABBTNode*>(
            std::realloc(nodes_, static_cast<std::size_t>(newCapacity) * sizeof(DAABBTNode))
        );
        assert(newNodes != nullptr);
        nodes_ = newNodes;

        // Initialize new nodes as free list
        for (int32_t i = nodeCapacity_; i < newCapacity - 1; ++i) {
            nodes_[i].child[0] = i + 1;
            nodes_[i].height   = -1;
        }
        nodes_[newCapacity - 1].child[0] = freeList_;
        nodes_[newCapacity - 1].height   = -1;
        freeList_ = nodeCapacity_;

        nodeCapacity_ = newCapacity;
    }

    int32_t allocNode() noexcept {
        if (freeList_ == NullNode) {
            grow(nodeCapacity_ * 2);
        }
        int32_t idx = freeList_;
        freeList_ = nodes_[idx].child[0];
        nodes_[idx].parent   = NullNode;
        nodes_[idx].child[0] = NullNode;
        nodes_[idx].child[1] = NullNode;
        nodes_[idx].height   = 0;
        nodes_[idx].userData = 0;
        ++nodeCount_;
        return idx;
    }

    void freeNode(int32_t idx) noexcept {
        assert(idx >= 0 && idx < nodeCapacity_);
        nodes_[idx].child[0] = freeList_;
        nodes_[idx].height   = -1;
        freeList_ = idx;
        --nodeCount_;
    }

    // ── SAH insertion ─────────────────────────────────────────────────────

    void insertLeaf(int32_t leafIdx) noexcept {
        if (root_ == NullNode) {
            root_ = leafIdx;
            nodes_[leafIdx].parent = NullNode;
            return;
        }

        AABB leafAABB = nodes_[leafIdx].aabb;

        // Walk down the tree choosing the best sibling via SAH
        int32_t bestSibling = root_;
        float bestCost = nodes_[root_].aabb.merged(leafAABB).surfaceArea();

        struct Candidate {
            int32_t nodeIdx;
            float   inheritedCost;
        };

        Candidate stack[64];
        int32_t top = 0;
        stack[top++] = { root_, 0.0f };

        float leafSA = leafAABB.surfaceArea();

        while (top > 0) {
            Candidate c = stack[--top];
            if (c.nodeIdx == NullNode) continue;

            const DAABBTNode& nd = nodes_[c.nodeIdx];
            AABB combined = nd.aabb.merged(leafAABB);
            float directCost = combined.surfaceArea();
            float cost = directCost + c.inheritedCost;

            if (cost < bestCost) {
                bestCost    = cost;
                bestSibling = c.nodeIdx;
            }

            float inheritedCostChild = c.inheritedCost + (directCost - nd.aabb.surfaceArea());
            float lowerBound         = leafSA + inheritedCostChild;

            if (lowerBound < bestCost && !nd.isLeaf() && top < 62) {
                stack[top++] = { nd.child[0], inheritedCostChild };
                stack[top++] = { nd.child[1], inheritedCostChild };
            }
        }

        // Create a new internal node linking leaf and bestSibling
        int32_t oldParent = nodes_[bestSibling].parent;
        int32_t newParent = allocNode();

        nodes_[newParent].parent     = oldParent;
        nodes_[newParent].aabb       = leafAABB.merged(nodes_[bestSibling].aabb);
        nodes_[newParent].height     = nodes_[bestSibling].height + 1;
        nodes_[newParent].proxyIndex = ProxyHandle::InvalidRaw;
        nodes_[newParent].userData   = 0;

        if (oldParent == NullNode) {
            root_ = newParent;
        } else {
            if (nodes_[oldParent].child[0] == bestSibling)
                nodes_[oldParent].child[0] = newParent;
            else
                nodes_[oldParent].child[1] = newParent;
        }

        nodes_[newParent].child[0] = bestSibling;
        nodes_[newParent].child[1] = leafIdx;
        nodes_[bestSibling].parent = newParent;
        nodes_[leafIdx].parent     = newParent;

        refitAncestors(newParent);
    }

    void removeLeaf(int32_t leafIdx) noexcept {
        if (leafIdx == root_) {
            root_ = NullNode;
            return;
        }

        int32_t parent  = nodes_[leafIdx].parent;
        int32_t grandpa = nodes_[parent].parent;
        int32_t sibling = (nodes_[parent].child[0] == leafIdx)
                           ? nodes_[parent].child[1]
                           : nodes_[parent].child[0];

        if (grandpa != NullNode) {
            if (nodes_[grandpa].child[0] == parent)
                nodes_[grandpa].child[0] = sibling;
            else
                nodes_[grandpa].child[1] = sibling;
            nodes_[sibling].parent = grandpa;
            freeNode(parent);
            refitAncestors(grandpa);
        } else {
            root_ = sibling;
            nodes_[sibling].parent = NullNode;
            freeNode(parent);
        }
    }

    // ── Refit & balance ───────────────────────────────────────────────────

    void refitAncestors(int32_t idx) noexcept {
        while (idx != NullNode) {
            idx = balance(idx);

            DAABBTNode& node = nodes_[idx];
            assert(!node.isLeaf());

            int32_t c0 = node.child[0];
            int32_t c1 = node.child[1];
            node.height = 1 + (nodes_[c0].height > nodes_[c1].height
                                ? nodes_[c0].height : nodes_[c1].height);
            node.aabb   = nodes_[c0].aabb.merged(nodes_[c1].aabb);

            idx = node.parent;
        }
    }

    int32_t balance(int32_t idx) noexcept {
        DAABBTNode& A = nodes_[idx];
        if (A.isLeaf() || A.height < 2) return idx;

        int32_t iB = A.child[0];
        int32_t iC = A.child[1];
        DAABBTNode& B = nodes_[iB];
        DAABBTNode& C = nodes_[iC];

        int32_t bal = C.height - B.height;

        // Rotate C up
        if (bal > 1) {
            int32_t iF = C.child[0];
            int32_t iG = C.child[1];
            DAABBTNode& F = nodes_[iF];
            DAABBTNode& G = nodes_[iG];

            C.child[0] = idx;
            C.parent   = A.parent;
            A.parent   = iC;

            if (C.parent != NullNode) {
                if (nodes_[C.parent].child[0] == idx)
                    nodes_[C.parent].child[0] = iC;
                else
                    nodes_[C.parent].child[1] = iC;
            } else {
                root_ = iC;
            }

            if (F.height > G.height) {
                C.child[1]  = iF;
                A.child[1]  = iG;
                G.parent    = idx;
                A.aabb      = B.aabb.merged(G.aabb);
                C.aabb      = A.aabb.merged(F.aabb);
                A.height    = 1 + (B.height > G.height ? B.height : G.height);
                C.height    = 1 + (A.height > F.height ? A.height : F.height);
            } else {
                C.child[1]  = iG;
                A.child[1]  = iF;
                F.parent    = idx;
                A.aabb      = B.aabb.merged(F.aabb);
                C.aabb      = A.aabb.merged(G.aabb);
                A.height    = 1 + (B.height > F.height ? B.height : F.height);
                C.height    = 1 + (A.height > G.height ? A.height : G.height);
            }
            return iC;
        }

        // Rotate B up
        if (bal < -1) {
            int32_t iD = B.child[0];
            int32_t iE = B.child[1];
            DAABBTNode& D = nodes_[iD];
            DAABBTNode& E = nodes_[iE];

            B.child[0] = idx;
            B.parent   = A.parent;
            A.parent   = iB;

            if (B.parent != NullNode) {
                if (nodes_[B.parent].child[0] == idx)
                    nodes_[B.parent].child[0] = iB;
                else
                    nodes_[B.parent].child[1] = iB;
            } else {
                root_ = iB;
            }

            if (D.height > E.height) {
                B.child[1]  = iD;
                A.child[0]  = iE;
                E.parent    = idx;
                A.aabb      = C.aabb.merged(E.aabb);
                B.aabb      = A.aabb.merged(D.aabb);
                A.height    = 1 + (C.height > E.height ? C.height : E.height);
                B.height    = 1 + (A.height > D.height ? A.height : D.height);
            } else {
                B.child[1]  = iE;
                A.child[0]  = iD;
                D.parent    = idx;
                A.aabb      = C.aabb.merged(D.aabb);
                B.aabb      = A.aabb.merged(E.aabb);
                A.height    = 1 + (C.height > D.height ? C.height : D.height);
                B.height    = 1 + (A.height > E.height ? A.height : E.height);
            }
            return iB;
        }

        return idx;
    }

    // ── Recursive self-query for O(n log n) pair enumeration ──────────────

    /**
     * @brief Recursive tree self-query. When nodeA == nodeB, tests all pairs
     * within that subtree. When nodeA != nodeB, tests all pairs across the
     * two subtrees. Prunes when subtree AABBs don't overlap.
     */
    void selfQuery(int32_t nodeA, int32_t nodeB,
                   OverlapPair* out, uint32_t maxOut, uint32_t& pairCount) const noexcept
    {
        if (pairCount >= maxOut) return;
        if (nodeA == NullNode || nodeB == NullNode) return;

        const DAABBTNode& A = nodes_[nodeA];
        const DAABBTNode& B = nodes_[nodeB];

        // Same-node self-query: expand into children
        if (nodeA == nodeB) {
            if (A.isLeaf()) return; // Single leaf — no pairs

            int32_t c0 = A.child[0];
            int32_t c1 = A.child[1];

            // Pairs within left subtree
            selfQuery(c0, c0, out, maxOut, pairCount);
            // Pairs within right subtree
            selfQuery(c1, c1, out, maxOut, pairCount);
            // Pairs across left × right
            selfQuery(c0, c1, out, maxOut, pairCount);
            return;
        }

        // Cross-query: prune if AABBs don't overlap
        if (!A.aabb.overlaps(B.aabb)) return;

        if (A.isLeaf() && B.isLeaf()) {
            // Both leaves — emit pair (canonical order: lower raw first)
            if (pairCount < maxOut && A.proxyIndex != B.proxyIndex) {
                out[pairCount++] = OverlapPair(A.proxy(), B.proxy());
            }
            return;
        }

        // Descend into the larger node to keep queries balanced
        if (A.isLeaf() || (!B.isLeaf() && A.height < B.height)) {
            selfQuery(nodeA, B.child[0], out, maxOut, pairCount);
            selfQuery(nodeA, B.child[1], out, maxOut, pairCount);
        } else {
            selfQuery(A.child[0], nodeB, out, maxOut, pairCount);
            selfQuery(A.child[1], nodeB, out, maxOut, pairCount);
        }
    }

    // ── Validation helpers ────────────────────────────────────────────────

    bool validateNode(int32_t idx, int32_t expectedParent) const noexcept {
        if (idx < 0 || idx >= nodeCapacity_) return false;
        const DAABBTNode& node = nodes_[idx];

        if (node.isFree()) return false;
        if (node.parent != expectedParent) return false;

        if (node.isLeaf()) {
            if (node.height != 0) return false;
            return true;
        }

        // Internal node
        int32_t c0 = node.child[0];
        int32_t c1 = node.child[1];

        if (c0 == NullNode || c1 == NullNode) return false;

        // Children's AABBs should be contained in parent
        if (!node.aabb.contains(nodes_[c0].aabb)) return false;
        if (!node.aabb.contains(nodes_[c1].aabb)) return false;

        // Height check
        int32_t expectedHeight = 1 + (nodes_[c0].height > nodes_[c1].height
                                       ? nodes_[c0].height : nodes_[c1].height);
        if (node.height != expectedHeight) return false;

        // Recurse
        if (!validateNode(c0, idx)) return false;
        if (!validateNode(c1, idx)) return false;

        return true;
    }
};

} // namespace pulse
