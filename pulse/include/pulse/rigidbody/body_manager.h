/**
 * @file body_manager.h
 * @brief Central body registry — create, destroy, and lookup rigid bodies via generational handles.
 *
 * BodyManager wraps RigidBodyStore with a HandlePool<BodyTag> to provide:
 *  - O(1) create / destroy (swap-and-pop with handle remapping)
 *  - O(1) handle validation (generation check)
 *  - O(1) handle → dense index resolution
 *  - Convenience accessors by BodyHandle
 *  - SolverBody conversion (to/from)
 *
 * Internally maintains a sparse→dense and dense→sparse mapping so that
 * handles remain stable across swap-and-pop removals.
 *
 * Thread safety: NOT thread-safe. External synchronization required.
 */

#pragma once

#include <pulse/rigidbody/rigid_body_common.h>
#include <pulse/rigidbody/rigid_body.h>
#include <pulse/utilities/handle.h>
#include <pulse/utilities/handle_pool.h>
#include <pulse/utilities/assert.h>
#include <pulse/math/math_common.h>
#include <pulse/solver/solver_common.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace pulse {

/**
 * @class BodyManager
 * @brief Manages rigid body lifecycle with generational handle safety.
 *
 * Bodies are stored in dense SoA arrays (RigidBodyStore). External references
 * use BodyHandle (generational). Two mapping arrays maintain the bidirectional
 * relationship:
 *  - sparse_[handle.index()] → dense index in RigidBodyStore
 *  - dense_[denseIndex]      → handle index (for reverse lookup on swap)
 */
class BodyManager {
public:
    // ── Construction / Destruction ───────────────────────────────────────

    explicit BodyManager(std::size_t initialCapacity = 256) noexcept
        : handlePool_(initialCapacity),
          store_(initialCapacity),
          sparseCapacity_(initialCapacity)
    {
        sparse_ = static_cast<uint32_t*>(std::malloc(sparseCapacity_ * sizeof(uint32_t)));
        dense_  = static_cast<uint32_t*>(std::malloc(sparseCapacity_ * sizeof(uint32_t)));
        PULSE_ASSERT_MSG(sparse_ != nullptr, "BodyManager: sparse alloc failed");
        PULSE_ASSERT_MSG(dense_ != nullptr, "BodyManager: dense alloc failed");

        // Initialize sparse to invalid.
        std::memset(sparse_, 0xFF, sparseCapacity_ * sizeof(uint32_t));
        std::memset(dense_, 0xFF, sparseCapacity_ * sizeof(uint32_t));
    }

    ~BodyManager() noexcept {
        std::free(sparse_);
        std::free(dense_);
    }

    // Non-copyable, movable.
    BodyManager(const BodyManager&) = delete;
    BodyManager& operator=(const BodyManager&) = delete;

    BodyManager(BodyManager&& other) noexcept
        : handlePool_(std::move(other.handlePool_)),
          store_(std::move(other.store_)),
          sparse_(other.sparse_),
          dense_(other.dense_),
          sparseCapacity_(other.sparseCapacity_)
    {
        other.sparse_ = nullptr;
        other.dense_ = nullptr;
        other.sparseCapacity_ = 0;
    }

    BodyManager& operator=(BodyManager&& other) noexcept {
        if (this != &other) {
            std::free(sparse_);
            std::free(dense_);
            handlePool_ = std::move(other.handlePool_);
            store_ = std::move(other.store_);
            sparse_ = other.sparse_;
            dense_ = other.dense_;
            sparseCapacity_ = other.sparseCapacity_;
            other.sparse_ = nullptr;
            other.dense_ = nullptr;
            other.sparseCapacity_ = 0;
        }
        return *this;
    }

    // ── Body creation / destruction ─────────────────────────────────────

    /// Create a new body from a definition. Returns a stable BodyHandle.
    [[nodiscard]] BodyHandle createBody(const BodyDef& def) noexcept {
        BodyHandle handle = handlePool_.allocate();
        uint32_t handleIdx = handle.index();

        // Grow sparse array if needed.
        if (handleIdx >= sparseCapacity_) {
            growSparse(handleIdx + 1);
        }

        // Add to dense store.
        std::size_t denseIdx = store_.add(def);

        // Map handle ↔ dense.
        sparse_[handleIdx] = static_cast<uint32_t>(denseIdx);
        dense_[denseIdx] = handleIdx;

        return handle;
    }

    /// Destroy a body by handle. O(1) via swap-and-pop.
    void destroyBody(BodyHandle handle) noexcept {
        PULSE_ASSERT(isValid(handle));

        uint32_t handleIdx = handle.index();
        uint32_t denseIdx = sparse_[handleIdx];
        std::size_t lastDense = store_.size() - 1;

        // If not the last element, swap-and-pop will move the last element
        // to denseIdx. We need to update the mapping for the moved element.
        if (denseIdx != static_cast<uint32_t>(lastDense)) {
            uint32_t movedHandleIdx = dense_[lastDense];

            // Store removes element at denseIdx and swaps in lastDense.
            store_.remove(denseIdx);

            // Update mappings for the moved element.
            sparse_[movedHandleIdx] = denseIdx;
            dense_[denseIdx] = movedHandleIdx;
        } else {
            // It was the last element — just remove it.
            store_.remove(denseIdx);
        }

        // Invalidate the destroyed handle's mappings.
        sparse_[handleIdx] = 0xFFFFFFFFu;
        dense_[lastDense] = 0xFFFFFFFFu;

        // Free the handle (increments generation, invalidating all copies).
        handlePool_.free(handle);
    }

    // ── Handle validation and resolution ─────────────────────────────────

    /// Check if a handle is currently valid.
    [[nodiscard]] PULSE_FORCE_INLINE bool isValid(BodyHandle handle) const noexcept {
        return handlePool_.isValid(handle);
    }

    /// Resolve a handle to its dense SoA index.
    /// Precondition: handle must be valid (checked by assert).
    [[nodiscard]] PULSE_FORCE_INLINE uint32_t getIndex(BodyHandle handle) const noexcept {
        PULSE_ASSERT(isValid(handle));
        return sparse_[handle.index()];
    }

    // ── Convenience accessors (by handle) ────────────────────────────────

    /// Get position.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 getPosition(BodyHandle h) const noexcept {
        return store_.position(getIndex(h));
    }

    /// Set position.
    PULSE_FORCE_INLINE void setPosition(BodyHandle h, Vec3 pos) noexcept {
        store_.setPosition(getIndex(h), pos);
    }

    /// Get rotation.
    [[nodiscard]] PULSE_FORCE_INLINE Quat getRotation(BodyHandle h) const noexcept {
        return store_.rotation(getIndex(h));
    }

    /// Set rotation.
    PULSE_FORCE_INLINE void setRotation(BodyHandle h, Quat rot) noexcept {
        store_.setRotation(getIndex(h), rot);
    }

    /// Get linear velocity.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 getLinearVelocity(BodyHandle h) const noexcept {
        return store_.linearVelocity(getIndex(h));
    }

    /// Set linear velocity.
    PULSE_FORCE_INLINE void setLinearVelocity(BodyHandle h, Vec3 vel) noexcept {
        store_.linearVelocity(getIndex(h)) = vel;
    }

    /// Get angular velocity.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 getAngularVelocity(BodyHandle h) const noexcept {
        return store_.angularVelocity(getIndex(h));
    }

    /// Set angular velocity.
    PULSE_FORCE_INLINE void setAngularVelocity(BodyHandle h, Vec3 vel) noexcept {
        store_.angularVelocity(getIndex(h)) = vel;
    }

    /// Apply a force at the center of mass (accumulated until clearForces).
    PULSE_FORCE_INLINE void applyForce(BodyHandle h, Vec3 f) noexcept {
        store_.force(getIndex(h)) += f;
    }

    /// Apply a torque (accumulated until clearForces).
    PULSE_FORCE_INLINE void applyTorque(BodyHandle h, Vec3 t) noexcept {
        store_.torque(getIndex(h)) += t;
    }

    /// Apply a linear impulse (immediate velocity change).
    PULSE_FORCE_INLINE void applyLinearImpulse(BodyHandle h, Vec3 impulse) noexcept {
        uint32_t idx = getIndex(h);
        store_.linearVelocity(idx) += impulse * store_.invMass(idx);
    }

    /// Apply an angular impulse (immediate angular velocity change).
    PULSE_FORCE_INLINE void applyAngularImpulse(BodyHandle h, Vec3 impulse) noexcept {
        uint32_t idx = getIndex(h);
        store_.angularVelocity(idx) += store_.worldInvInertia(idx) * impulse;
    }

    /// Apply a force at a world-space point (generates both force and torque).
    PULSE_FORCE_INLINE void applyForceAtPoint(BodyHandle h, Vec3 f, Vec3 worldPoint) noexcept {
        uint32_t idx = getIndex(h);
        store_.force(idx) += f;
        Vec3 r = worldPoint - store_.position(idx);
        store_.torque(idx) += r.cross(f);
    }

    /// Get body type.
    [[nodiscard]] PULSE_FORCE_INLINE BodyType getBodyType(BodyHandle h) const noexcept {
        return store_.bodyType(getIndex(h));
    }

    /// Check if body is sleeping.
    [[nodiscard]] PULSE_FORCE_INLINE bool isSleeping(BodyHandle h) const noexcept {
        return store_.isSleeping(getIndex(h));
    }

    /// Get inverse mass.
    [[nodiscard]] PULSE_FORCE_INLINE float getInvMass(BodyHandle h) const noexcept {
        return store_.invMass(getIndex(h));
    }

    // ── SolverBody conversion ────────────────────────────────────────────

    /// Build a SolverBody view for the solver from a handle.
    [[nodiscard]] PULSE_FORCE_INLINE SolverBody toSolverBody(BodyHandle h) const noexcept {
        return store_.toSolverBody(getIndex(h));
    }

    /// Build a SolverBody view from a dense index.
    [[nodiscard]] PULSE_FORCE_INLINE SolverBody toSolverBodyByIndex(uint32_t denseIdx) const noexcept {
        return store_.toSolverBody(denseIdx);
    }

    /// Write solver results back from a SolverBody.
    PULSE_FORCE_INLINE void fromSolverBody(BodyHandle h, const SolverBody& sb) noexcept {
        store_.fromSolverBody(getIndex(h), sb);
    }

    /// Write solver results back by dense index.
    PULSE_FORCE_INLINE void fromSolverBodyByIndex(uint32_t denseIdx, const SolverBody& sb) noexcept {
        store_.fromSolverBody(denseIdx, sb);
    }

    // ── Batch operations ─────────────────────────────────────────────────

    /// Clear all forces and torques.
    void clearForces() noexcept { store_.clearForces(); }

    /// Recompute world-space inertias for all dynamic bodies.
    void updateAllWorldInertias() noexcept { store_.updateAllWorldInertias(); }

    /// Clear all island assignments.
    void clearIslandIds() noexcept { store_.clearIslandIds(); }

    // ── Queries ──────────────────────────────────────────────────────────

    /// Number of bodies currently alive.
    [[nodiscard]] std::size_t bodyCount() const noexcept { return store_.size(); }

    /// Total capacity before growth.
    [[nodiscard]] std::size_t capacity() const noexcept { return store_.capacity(); }

    /// Get diagnostic statistics.
    [[nodiscard]] BodyStats getStats() const noexcept {
        BodyStats stats;
        stats.totalBodies = static_cast<uint32_t>(store_.size());
        for (std::size_t i = 0; i < store_.size(); ++i) {
            if (store_.isStatic(i)) {
                stats.staticBodies++;
            } else if (store_.isKinematic(i)) {
                stats.kinematicBodies++;
            } else if (store_.isSleeping(i)) {
                stats.sleepingBodies++;
            } else {
                stats.activeBodies++;
            }
        }
        return stats;
    }

    // ── Direct store access (for modules that need bulk operations) ──────

    /// Get the underlying SoA store (read-only).
    [[nodiscard]] const RigidBodyStore& store() const noexcept { return store_; }

    /// Get the underlying SoA store (mutable).
    [[nodiscard]] RigidBodyStore& store() noexcept { return store_; }

    /// Get the handle corresponding to a dense index.
    [[nodiscard]] PULSE_FORCE_INLINE BodyHandle getHandleByIndex(uint32_t denseIdx) const noexcept {
        PULSE_ASSERT(denseIdx < store_.size());
        uint32_t handleIdx = dense_[denseIdx];
        // Reconstruct the handle from the handle index and current generation.
        return BodyHandle(handleIdx, handlePool_.isValid(BodyHandle(handleIdx, 0))
            ? BodyHandle(handleIdx, 0).generation() : 0);
    }

    // ── Iteration ────────────────────────────────────────────────────────

    /// Iterate over all active bodies. Calls func(uint32_t denseIndex) for each.
    template <typename Func>
    void forEach(Func&& func) const {
        for (std::size_t i = 0; i < store_.size(); ++i) {
            func(static_cast<uint32_t>(i));
        }
    }

    /// Iterate over all awake dynamic bodies. Calls func(uint32_t denseIndex).
    template <typename Func>
    void forEachActive(Func&& func) const {
        for (std::size_t i = 0; i < store_.size(); ++i) {
            if (store_.isActive(i)) {
                func(static_cast<uint32_t>(i));
            }
        }
    }

private:
    util::HandlePool<BodyTag> handlePool_;
    RigidBodyStore store_;

    uint32_t* sparse_          = nullptr;  ///< handleIndex → denseIndex.
    uint32_t* dense_           = nullptr;  ///< denseIndex → handleIndex.
    std::size_t sparseCapacity_ = 0;

    /// Grow the sparse/dense mapping arrays.
    void growSparse(std::size_t minCapacity) noexcept {
        std::size_t newCap = sparseCapacity_;
        while (newCap < minCapacity) {
            newCap *= 2;
        }

        auto* newSparse = static_cast<uint32_t*>(std::realloc(sparse_, newCap * sizeof(uint32_t)));
        auto* newDense  = static_cast<uint32_t*>(std::realloc(dense_, newCap * sizeof(uint32_t)));
        PULSE_ASSERT_MSG(newSparse != nullptr, "BodyManager: sparse realloc failed");
        PULSE_ASSERT_MSG(newDense != nullptr, "BodyManager: dense realloc failed");

        // Initialize new entries.
        std::memset(newSparse + sparseCapacity_, 0xFF, (newCap - sparseCapacity_) * sizeof(uint32_t));
        std::memset(newDense + sparseCapacity_, 0xFF, (newCap - sparseCapacity_) * sizeof(uint32_t));

        sparse_ = newSparse;
        dense_ = newDense;
        sparseCapacity_ = newCap;
    }
};

} // namespace pulse
