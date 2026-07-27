/**
 * @file rigid_body.h
 * @brief SoA rigid body data store — cache-friendly parallel arrays for all per-body state.
 *
 * RigidBodyStore holds the authoritative state for all rigid bodies as separate
 * flat arrays (Structure-of-Arrays layout). Each component array is independently
 * cache-line aligned for maximum SIMD throughput during bulk operations.
 *
 * This is the low-level data container. BodyManager wraps it with handle-based
 * access, creation/destruction, and handle→dense-index mapping.
 *
 * Memory layout (conceptual):
 *   [Pos0 Pos1 ... PosN] [Rot0 Rot1 ... RotN] [LinVel0 ... LinVelN] [...]
 *   ^--- aligned           ^--- aligned           ^--- aligned
 *
 * Thread safety: NOT thread-safe. External synchronization required.
 */

#pragma once

#include <pulse/rigidbody/rigid_body_common.h>
#include <pulse/math/math_common.h>
#include <pulse/math/vec3.h>
#include <pulse/math/quat.h>
#include <pulse/math/mat3.h>
#include <pulse/math/transform.h>
#include <pulse/math/aabb.h>
#include <pulse/solver/solver_common.h>
#include <pulse/utilities/assert.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

namespace pulse {

// ── Aligned allocation helpers ───────────────────────────────────────────────

namespace detail {

inline void* bodyAlignedAlloc(std::size_t size, std::size_t alignment) noexcept {
    if (size == 0) return nullptr;
#if defined(PULSE_COMPILER_MSVC)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) return nullptr;
    return ptr;
#endif
}

inline void bodyAlignedFree(void* ptr) noexcept {
#if defined(PULSE_COMPILER_MSVC)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

} // namespace detail

// ── RigidBodyStore ───────────────────────────────────────────────────────────

/**
 * @class RigidBodyStore
 * @brief SoA storage for all rigid body state.
 *
 * Manages cache-line-aligned parallel arrays for each body component.
 * Supports swap-and-pop removal for O(1) destruction.
 */
class RigidBodyStore {
public:
    // ── Construction / Destruction ───────────────────────────────────────

    explicit RigidBodyStore(std::size_t initialCapacity = 256) noexcept
        : size_(0), capacity_(0) {
        grow(initialCapacity);
    }

    ~RigidBodyStore() noexcept {
        freeArrays();
    }

    // Non-copyable, movable.
    RigidBodyStore(const RigidBodyStore&) = delete;
    RigidBodyStore& operator=(const RigidBodyStore&) = delete;

    RigidBodyStore(RigidBodyStore&& other) noexcept
        : size_(other.size_), capacity_(other.capacity_),
          transforms_(other.transforms_),
          linearVelocities_(other.linearVelocities_),
          angularVelocities_(other.angularVelocities_),
          forces_(other.forces_),
          torques_(other.torques_),
          invMasses_(other.invMasses_),
          localInvInertias_(other.localInvInertias_),
          worldInvInertias_(other.worldInvInertias_),
          aabbs_(other.aabbs_),
          restitutions_(other.restitutions_),
          frictions_(other.frictions_),
          linearDampings_(other.linearDampings_),
          angularDampings_(other.angularDampings_),
          gravityScales_(other.gravityScales_),
          bodyTypes_(other.bodyTypes_),
          bodyFlags_(other.bodyFlags_),
          islandIds_(other.islandIds_),
          shapeTypes_(other.shapeTypes_),
          collisionLayers_(other.collisionLayers_),
          collisionMasks_(other.collisionMasks_),
          sleepTimers_(other.sleepTimers_)
    {
        other.nullifyPointers();
        other.size_ = 0;
        other.capacity_ = 0;
    }

    RigidBodyStore& operator=(RigidBodyStore&& other) noexcept {
        if (this != &other) {
            freeArrays();
            size_ = other.size_;
            capacity_ = other.capacity_;
            transforms_ = other.transforms_;
            linearVelocities_ = other.linearVelocities_;
            angularVelocities_ = other.angularVelocities_;
            forces_ = other.forces_;
            torques_ = other.torques_;
            invMasses_ = other.invMasses_;
            localInvInertias_ = other.localInvInertias_;
            worldInvInertias_ = other.worldInvInertias_;
            aabbs_ = other.aabbs_;
            restitutions_ = other.restitutions_;
            frictions_ = other.frictions_;
            linearDampings_ = other.linearDampings_;
            angularDampings_ = other.angularDampings_;
            gravityScales_ = other.gravityScales_;
            bodyTypes_ = other.bodyTypes_;
            bodyFlags_ = other.bodyFlags_;
            islandIds_ = other.islandIds_;
            shapeTypes_ = other.shapeTypes_;
            collisionLayers_ = other.collisionLayers_;
            collisionMasks_ = other.collisionMasks_;
            sleepTimers_ = other.sleepTimers_;
            other.nullifyPointers();
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    // ── Add / Remove ─────────────────────────────────────────────────────

    /// Add a body from a BodyDef. Returns the dense index.
    std::size_t add(const BodyDef& def) noexcept {
        if (size_ >= capacity_) {
            grow(capacity_ * 2);
        }

        std::size_t idx = size_;

        transforms_[idx] = def.initialTransform;
        linearVelocities_[idx] = def.linearVelocity;
        angularVelocities_[idx] = def.angularVelocity;
        forces_[idx] = Vec3::zero();
        torques_[idx] = Vec3::zero();

        // Compute inverse mass
        if (def.type == BodyType::Static || def.type == BodyType::Kinematic) {
            invMasses_[idx] = 0.0f;
        } else {
            invMasses_[idx] = (def.mass > math::Epsilon) ? 1.0f / def.mass : 0.0f;
        }

        // Compute inverse inertia
        if (def.type == BodyType::Static || def.type == BodyType::Kinematic || def.fixedRotation) {
            localInvInertias_[idx] = Mat3(0.0f, 0.0f, 0.0f,
                                           0.0f, 0.0f, 0.0f,
                                           0.0f, 0.0f, 0.0f);
        } else {
            localInvInertias_[idx] = def.localInertia.inversed();
        }

        worldInvInertias_[idx] = localInvInertias_[idx]; // Updated by updateWorldInertia()
        aabbs_[idx] = AABB(); // Placeholder — set by broadphase
        restitutions_[idx] = def.restitution;
        frictions_[idx] = def.friction;
        linearDampings_[idx] = def.linearDamping;
        angularDampings_[idx] = def.angularDamping;
        gravityScales_[idx] = def.gravityScale;
        bodyTypes_[idx] = def.type;
        shapeTypes_[idx] = def.shapeType;
        collisionLayers_[idx] = def.collisionLayer;
        collisionMasks_[idx] = def.collisionMask;
        sleepTimers_[idx] = 0.0f;

        // Build flags
        BodyFlags flags = BodyFlags::Active;
        if (def.enableCCD)     flags |= BodyFlags::EnableCCD;
        if (def.fixedRotation) flags |= BodyFlags::FixedRotation;
        if (def.isBullet)      flags |= BodyFlags::Bullet;
        if (def.isSensor)      flags |= BodyFlags::Sensor;
        if (def.type != BodyType::Static) {
            flags |= BodyFlags::EnableGravity;
        }
        if (!def.startAwake) {
            flags |= BodyFlags::Sleeping;
        }
        bodyFlags_[idx] = flags;
        islandIds_[idx] = 0xFFFF;

        size_++;
        return idx;
    }

    /// Remove body at dense index using swap-and-pop (O(1), unordered).
    /// Returns the index of the element that was swapped in (== index if it was the last).
    std::size_t remove(std::size_t index) noexcept {
        PULSE_ASSERT(index < size_);
        std::size_t lastIdx = size_ - 1;
        if (index != lastIdx) {
            swapElements(index, lastIdx);
        }
        size_--;
        return (index != lastIdx) ? index : lastIdx;
    }

    // ── Component accessors (by dense index) ─────────────────────────────

    // Transform
    [[nodiscard]] PULSE_FORCE_INLINE Transform& transform(std::size_t i) noexcept { return transforms_[i]; }
    [[nodiscard]] PULSE_FORCE_INLINE const Transform& transform(std::size_t i) const noexcept { return transforms_[i]; }

    // Position (convenience)
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 position(std::size_t i) const noexcept { return transforms_[i].position; }
    PULSE_FORCE_INLINE void setPosition(std::size_t i, Vec3 pos) noexcept { transforms_[i].position = pos; }

    // Rotation (convenience)
    [[nodiscard]] PULSE_FORCE_INLINE Quat rotation(std::size_t i) const noexcept { return transforms_[i].rotation; }
    PULSE_FORCE_INLINE void setRotation(std::size_t i, Quat rot) noexcept { transforms_[i].rotation = rot; }

    // Linear velocity
    [[nodiscard]] PULSE_FORCE_INLINE Vec3& linearVelocity(std::size_t i) noexcept { return linearVelocities_[i]; }
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 linearVelocity(std::size_t i) const noexcept { return linearVelocities_[i]; }

    // Angular velocity
    [[nodiscard]] PULSE_FORCE_INLINE Vec3& angularVelocity(std::size_t i) noexcept { return angularVelocities_[i]; }
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 angularVelocity(std::size_t i) const noexcept { return angularVelocities_[i]; }

    // Force
    [[nodiscard]] PULSE_FORCE_INLINE Vec3& force(std::size_t i) noexcept { return forces_[i]; }
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 force(std::size_t i) const noexcept { return forces_[i]; }

    // Torque
    [[nodiscard]] PULSE_FORCE_INLINE Vec3& torque(std::size_t i) noexcept { return torques_[i]; }
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 torque(std::size_t i) const noexcept { return torques_[i]; }

    // Inverse mass
    [[nodiscard]] PULSE_FORCE_INLINE float invMass(std::size_t i) const noexcept { return invMasses_[i]; }
    PULSE_FORCE_INLINE void setInvMass(std::size_t i, float inv) noexcept { invMasses_[i] = inv; }

    // Local inverse inertia
    [[nodiscard]] PULSE_FORCE_INLINE const Mat3& localInvInertia(std::size_t i) const noexcept { return localInvInertias_[i]; }
    PULSE_FORCE_INLINE void setLocalInvInertia(std::size_t i, const Mat3& m) noexcept { localInvInertias_[i] = m; }

    // World inverse inertia
    [[nodiscard]] PULSE_FORCE_INLINE const Mat3& worldInvInertia(std::size_t i) const noexcept { return worldInvInertias_[i]; }

    // AABB
    [[nodiscard]] PULSE_FORCE_INLINE AABB& aabb(std::size_t i) noexcept { return aabbs_[i]; }
    [[nodiscard]] PULSE_FORCE_INLINE const AABB& aabb(std::size_t i) const noexcept { return aabbs_[i]; }

    // Material
    [[nodiscard]] PULSE_FORCE_INLINE float restitution(std::size_t i) const noexcept { return restitutions_[i]; }
    [[nodiscard]] PULSE_FORCE_INLINE float friction(std::size_t i) const noexcept { return frictions_[i]; }
    [[nodiscard]] PULSE_FORCE_INLINE float linearDamping(std::size_t i) const noexcept { return linearDampings_[i]; }
    [[nodiscard]] PULSE_FORCE_INLINE float angularDamping(std::size_t i) const noexcept { return angularDampings_[i]; }
    [[nodiscard]] PULSE_FORCE_INLINE float gravityScale(std::size_t i) const noexcept { return gravityScales_[i]; }

    // Type and flags
    [[nodiscard]] PULSE_FORCE_INLINE BodyType bodyType(std::size_t i) const noexcept { return bodyTypes_[i]; }
    [[nodiscard]] PULSE_FORCE_INLINE BodyFlags flags(std::size_t i) const noexcept { return bodyFlags_[i]; }
    PULSE_FORCE_INLINE void setFlags(std::size_t i, BodyFlags f) noexcept { bodyFlags_[i] = f; }
    PULSE_FORCE_INLINE void addFlags(std::size_t i, BodyFlags f) noexcept { bodyFlags_[i] |= f; }
    PULSE_FORCE_INLINE void removeFlags(std::size_t i, BodyFlags f) noexcept { bodyFlags_[i] &= ~f; }

    // Island
    [[nodiscard]] PULSE_FORCE_INLINE uint16_t islandId(std::size_t i) const noexcept { return islandIds_[i]; }
    PULSE_FORCE_INLINE void setIslandId(std::size_t i, uint16_t id) noexcept { islandIds_[i] = id; }

    // Shape type
    [[nodiscard]] PULSE_FORCE_INLINE ShapeType shapeType(std::size_t i) const noexcept { return shapeTypes_[i]; }

    // Collision layer/mask
    [[nodiscard]] PULSE_FORCE_INLINE uint16_t collisionLayer(std::size_t i) const noexcept { return collisionLayers_[i]; }
    [[nodiscard]] PULSE_FORCE_INLINE uint16_t collisionMask(std::size_t i) const noexcept { return collisionMasks_[i]; }

    // Sleep timer
    [[nodiscard]] PULSE_FORCE_INLINE float sleepTimer(std::size_t i) const noexcept { return sleepTimers_[i]; }
    PULSE_FORCE_INLINE void setSleepTimer(std::size_t i, float t) noexcept { sleepTimers_[i] = t; }

    // ── Batch array access (for SIMD / bulk processing) ──────────────────

    [[nodiscard]] Transform*      transforms()         noexcept { return transforms_; }
    [[nodiscard]] Vec3*            linearVelocities()   noexcept { return linearVelocities_; }
    [[nodiscard]] Vec3*            angularVelocities()  noexcept { return angularVelocities_; }
    [[nodiscard]] Vec3*            forces()             noexcept { return forces_; }
    [[nodiscard]] Vec3*            torques()            noexcept { return torques_; }
    [[nodiscard]] float*           invMasses()          noexcept { return invMasses_; }
    [[nodiscard]] Mat3*            localInvInertias()   noexcept { return localInvInertias_; }
    [[nodiscard]] Mat3*            worldInvInertias()   noexcept { return worldInvInertias_; }
    [[nodiscard]] AABB*            aabbs()              noexcept { return aabbs_; }
    [[nodiscard]] BodyType*        bodyTypes()          noexcept { return bodyTypes_; }
    [[nodiscard]] BodyFlags*       bodyFlagsArray()     noexcept { return bodyFlags_; }

    [[nodiscard]] const Transform* transforms()        const noexcept { return transforms_; }
    [[nodiscard]] const Vec3*      linearVelocities()  const noexcept { return linearVelocities_; }
    [[nodiscard]] const Vec3*      angularVelocities() const noexcept { return angularVelocities_; }
    [[nodiscard]] const float*     invMasses()         const noexcept { return invMasses_; }

    // ── Bulk operations ──────────────────────────────────────────────────

    /// Recompute world-space inverse inertia for body i.
    /// worldInvI = R * localInvI * Rᵀ
    PULSE_FORCE_INLINE void updateWorldInertia(std::size_t i) noexcept {
        Mat3 R = Mat3::fromQuat(transforms_[i].rotation);
        Mat3 Rt = R.transposed();
        worldInvInertias_[i] = R * localInvInertias_[i] * Rt;
    }

    /// Recompute world inertias for all bodies.
    void updateAllWorldInertias() noexcept {
        for (std::size_t i = 0; i < size_; ++i) {
            if (bodyTypes_[i] == BodyType::Dynamic) {
                updateWorldInertia(i);
            }
        }
    }

    /// Clear all accumulated forces and torques.
    void clearForces() noexcept {
        for (std::size_t i = 0; i < size_; ++i) {
            forces_[i] = Vec3::zero();
            torques_[i] = Vec3::zero();
        }
    }

    /// Reset all island IDs to unassigned (0xFFFF).
    void clearIslandIds() noexcept {
        for (std::size_t i = 0; i < size_; ++i) {
            islandIds_[i] = 0xFFFF;
            removeFlags(i, BodyFlags::InIsland);
        }
    }

    /// Build a SolverBody view from dense index.
    [[nodiscard]] PULSE_FORCE_INLINE SolverBody toSolverBody(std::size_t i) const noexcept {
        SolverBody sb;
        sb.position = transforms_[i].position;
        sb.linearVelocity = linearVelocities_[i];
        sb.angularVelocity = angularVelocities_[i];
        sb.invMass = invMasses_[i];

        // Extract diagonal of world inverse inertia for SolverBody's Vec3 representation.
        const Mat3& wii = worldInvInertias_[i];
        sb.invInertia = Vec3(wii(0, 0), wii(1, 1), wii(2, 2));

        sb.restitution = restitutions_[i];
        sb.friction = frictions_[i];
        sb.bodyId = static_cast<uint32_t>(i);
        return sb;
    }

    /// Write solver results back to body state from a SolverBody.
    PULSE_FORCE_INLINE void fromSolverBody(std::size_t i, const SolverBody& sb) noexcept {
        transforms_[i].position = sb.position;
        linearVelocities_[i] = sb.linearVelocity;
        angularVelocities_[i] = sb.angularVelocity;
    }

    // ── Queries ──────────────────────────────────────────────────────────

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /// Check if body at index is dynamic.
    [[nodiscard]] PULSE_FORCE_INLINE bool isDynamic(std::size_t i) const noexcept {
        return bodyTypes_[i] == BodyType::Dynamic;
    }

    /// Check if body at index is static.
    [[nodiscard]] PULSE_FORCE_INLINE bool isStatic(std::size_t i) const noexcept {
        return bodyTypes_[i] == BodyType::Static;
    }

    /// Check if body at index is kinematic.
    [[nodiscard]] PULSE_FORCE_INLINE bool isKinematic(std::size_t i) const noexcept {
        return bodyTypes_[i] == BodyType::Kinematic;
    }

    /// Check if body at index is sleeping.
    [[nodiscard]] PULSE_FORCE_INLINE bool isSleeping(std::size_t i) const noexcept {
        return hasFlag(bodyFlags_[i], BodyFlags::Sleeping);
    }

    /// Check if body at index is awake and dynamic.
    [[nodiscard]] PULSE_FORCE_INLINE bool isActive(std::size_t i) const noexcept {
        return isDynamic(i) && !isSleeping(i);
    }

private:
    std::size_t size_     = 0;
    std::size_t capacity_ = 0;

    // ── SoA arrays ──────────────────────────────────────────────────────
    Transform* transforms_         = nullptr;
    Vec3*      linearVelocities_   = nullptr;
    Vec3*      angularVelocities_  = nullptr;
    Vec3*      forces_             = nullptr;
    Vec3*      torques_            = nullptr;
    float*     invMasses_          = nullptr;
    Mat3*      localInvInertias_   = nullptr;
    Mat3*      worldInvInertias_   = nullptr;
    AABB*      aabbs_              = nullptr;
    float*     restitutions_       = nullptr;
    float*     frictions_          = nullptr;
    float*     linearDampings_     = nullptr;
    float*     angularDampings_    = nullptr;
    float*     gravityScales_      = nullptr;
    BodyType*  bodyTypes_          = nullptr;
    BodyFlags* bodyFlags_          = nullptr;
    uint16_t*  islandIds_          = nullptr;
    ShapeType* shapeTypes_         = nullptr;
    uint16_t*  collisionLayers_    = nullptr;
    uint16_t*  collisionMasks_     = nullptr;
    float*     sleepTimers_        = nullptr;

    // ── Internal helpers ─────────────────────────────────────────────────

    void nullifyPointers() noexcept {
        transforms_ = nullptr;
        linearVelocities_ = nullptr;
        angularVelocities_ = nullptr;
        forces_ = nullptr;
        torques_ = nullptr;
        invMasses_ = nullptr;
        localInvInertias_ = nullptr;
        worldInvInertias_ = nullptr;
        aabbs_ = nullptr;
        restitutions_ = nullptr;
        frictions_ = nullptr;
        linearDampings_ = nullptr;
        angularDampings_ = nullptr;
        gravityScales_ = nullptr;
        bodyTypes_ = nullptr;
        bodyFlags_ = nullptr;
        islandIds_ = nullptr;
        shapeTypes_ = nullptr;
        collisionLayers_ = nullptr;
        collisionMasks_ = nullptr;
        sleepTimers_ = nullptr;
    }

    template <typename T>
    static T* allocArray(std::size_t count) noexcept {
        auto* ptr = static_cast<T*>(detail::bodyAlignedAlloc(count * sizeof(T), PULSE_CACHE_LINE));
        PULSE_ASSERT_MSG(ptr != nullptr, "RigidBodyStore: allocation failed");
        return ptr;
    }

    template <typename T>
    static T* reallocArray(T* old, std::size_t oldCount, std::size_t newCount) noexcept {
        T* newArr = allocArray<T>(newCount);
        if (old && oldCount > 0) {
            std::memcpy(newArr, old, oldCount * sizeof(T));
        }
        detail::bodyAlignedFree(old);
        return newArr;
    }

    void grow(std::size_t newCap) noexcept {
        if (newCap <= capacity_) return;

        transforms_        = reallocArray(transforms_,        size_, newCap);
        linearVelocities_  = reallocArray(linearVelocities_,  size_, newCap);
        angularVelocities_ = reallocArray(angularVelocities_, size_, newCap);
        forces_            = reallocArray(forces_,            size_, newCap);
        torques_           = reallocArray(torques_,           size_, newCap);
        invMasses_         = reallocArray(invMasses_,         size_, newCap);
        localInvInertias_  = reallocArray(localInvInertias_,  size_, newCap);
        worldInvInertias_  = reallocArray(worldInvInertias_,  size_, newCap);
        aabbs_             = reallocArray(aabbs_,             size_, newCap);
        restitutions_      = reallocArray(restitutions_,      size_, newCap);
        frictions_         = reallocArray(frictions_,         size_, newCap);
        linearDampings_    = reallocArray(linearDampings_,    size_, newCap);
        angularDampings_   = reallocArray(angularDampings_,   size_, newCap);
        gravityScales_     = reallocArray(gravityScales_,     size_, newCap);
        bodyTypes_         = reallocArray(bodyTypes_,         size_, newCap);
        bodyFlags_         = reallocArray(bodyFlags_,         size_, newCap);
        islandIds_         = reallocArray(islandIds_,         size_, newCap);
        shapeTypes_        = reallocArray(shapeTypes_,        size_, newCap);
        collisionLayers_   = reallocArray(collisionLayers_,   size_, newCap);
        collisionMasks_    = reallocArray(collisionMasks_,    size_, newCap);
        sleepTimers_       = reallocArray(sleepTimers_,       size_, newCap);

        capacity_ = newCap;
    }

    void freeArrays() noexcept {
        detail::bodyAlignedFree(transforms_);
        detail::bodyAlignedFree(linearVelocities_);
        detail::bodyAlignedFree(angularVelocities_);
        detail::bodyAlignedFree(forces_);
        detail::bodyAlignedFree(torques_);
        detail::bodyAlignedFree(invMasses_);
        detail::bodyAlignedFree(localInvInertias_);
        detail::bodyAlignedFree(worldInvInertias_);
        detail::bodyAlignedFree(aabbs_);
        detail::bodyAlignedFree(restitutions_);
        detail::bodyAlignedFree(frictions_);
        detail::bodyAlignedFree(linearDampings_);
        detail::bodyAlignedFree(angularDampings_);
        detail::bodyAlignedFree(gravityScales_);
        detail::bodyAlignedFree(bodyTypes_);
        detail::bodyAlignedFree(bodyFlags_);
        detail::bodyAlignedFree(islandIds_);
        detail::bodyAlignedFree(shapeTypes_);
        detail::bodyAlignedFree(collisionLayers_);
        detail::bodyAlignedFree(collisionMasks_);
        detail::bodyAlignedFree(sleepTimers_);
        nullifyPointers();
    }

    /// Swap elements at indices a and b across all arrays.
    void swapElements(std::size_t a, std::size_t b) noexcept {
        auto swapVal = [](auto* arr, std::size_t i, std::size_t j) {
            auto tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        };

        swapVal(transforms_,        a, b);
        swapVal(linearVelocities_,   a, b);
        swapVal(angularVelocities_,  a, b);
        swapVal(forces_,             a, b);
        swapVal(torques_,            a, b);
        swapVal(invMasses_,          a, b);
        swapVal(localInvInertias_,   a, b);
        swapVal(worldInvInertias_,   a, b);
        swapVal(aabbs_,              a, b);
        swapVal(restitutions_,       a, b);
        swapVal(frictions_,          a, b);
        swapVal(linearDampings_,     a, b);
        swapVal(angularDampings_,    a, b);
        swapVal(gravityScales_,      a, b);
        swapVal(bodyTypes_,          a, b);
        swapVal(bodyFlags_,          a, b);
        swapVal(islandIds_,          a, b);
        swapVal(shapeTypes_,         a, b);
        swapVal(collisionLayers_,    a, b);
        swapVal(collisionMasks_,     a, b);
        swapVal(sleepTimers_,        a, b);
    }
};

} // namespace pulse
