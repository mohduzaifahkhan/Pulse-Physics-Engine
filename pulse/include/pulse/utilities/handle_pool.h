/**
 * @file handle_pool.h
 * @brief Pool that manages allocation, freeing, and validation of generational handles.
 *
 * Maintains an array of generation counters and a free list of available indices.
 * Allocation pops an index from the free list and returns a Handle with the
 * current generation. Freeing pushes the index back and increments its generation,
 * invalidating all outstanding handles to that slot.
 *
 * All operations are O(1). The pool pre-allocates a fixed capacity; if more
 * handles are needed, it grows by doubling (using std::realloc or engine allocators).
 *
 * Thread safety: NOT thread-safe. External synchronization required.
 */

#pragma once

#include <pulse/utilities/handle.h>
#include <pulse/utilities/assert.h>
#include <pulse/math/math_common.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>

namespace pulse {
namespace util {

/**
 * @class HandlePool
 * @brief Manages generational handle allocation, freeing, and validation.
 *
 * @tparam Tag Type tag matching the Handle<Tag> this pool manages.
 */
template <typename Tag = DefaultTag>
class HandlePool {
public:
    using HandleType = Handle<Tag>;

    // ── Construction / Destruction ───────────────────────────────────────

    /// Construct with initial capacity (number of handles pre-allocated).
    explicit HandlePool(std::size_t initialCapacity = 256) noexcept
        : capacity_(initialCapacity),
          count_(0),
          freeListHead_(0) {
        PULSE_ASSERT(initialCapacity > 0);

        generations_ = static_cast<uint32_t*>(
            std::malloc(capacity_ * sizeof(uint32_t)));
        freeList_ = static_cast<uint32_t*>(
            std::malloc(capacity_ * sizeof(uint32_t)));

        PULSE_ASSERT_MSG(generations_ != nullptr, "HandlePool: allocation failed");
        PULSE_ASSERT_MSG(freeList_ != nullptr, "HandlePool: allocation failed");

        // Initialize: all slots free, generation 0.
        for (std::size_t i = 0; i < capacity_; ++i) {
            generations_[i] = 0;
            freeList_[i] = static_cast<uint32_t>(i);
        }
        freeCount_ = static_cast<uint32_t>(capacity_);
    }

    ~HandlePool() noexcept {
        std::free(generations_);
        std::free(freeList_);
    }

    // Non-copyable, movable.
    HandlePool(const HandlePool&) = delete;
    HandlePool& operator=(const HandlePool&) = delete;

    HandlePool(HandlePool&& other) noexcept
        : generations_(other.generations_),
          freeList_(other.freeList_),
          capacity_(other.capacity_),
          count_(other.count_),
          freeCount_(other.freeCount_),
          freeListHead_(other.freeListHead_) {
        other.generations_ = nullptr;
        other.freeList_ = nullptr;
        other.capacity_ = 0;
        other.count_ = 0;
        other.freeCount_ = 0;
        other.freeListHead_ = 0;
    }

    HandlePool& operator=(HandlePool&& other) noexcept {
        if (this != &other) {
            std::free(generations_);
            std::free(freeList_);
            generations_ = other.generations_;
            freeList_ = other.freeList_;
            capacity_ = other.capacity_;
            count_ = other.count_;
            freeCount_ = other.freeCount_;
            freeListHead_ = other.freeListHead_;
            other.generations_ = nullptr;
            other.freeList_ = nullptr;
            other.capacity_ = 0;
            other.count_ = 0;
            other.freeCount_ = 0;
            other.freeListHead_ = 0;
        }
        return *this;
    }

    // ── Allocation ───────────────────────────────────────────────────────

    /// Allocate a new handle. Returns a handle with the current generation
    /// for the assigned slot. Grows the pool if necessary.
    [[nodiscard]] HandleType allocate() noexcept {
        if (freeCount_ == 0) {
            grow();
        }

        // Pop from the free list stack.
        uint32_t index = freeList_[freeListHead_];
        freeListHead_++;
        freeCount_--;
        count_++;

        return HandleType(index, generations_[index]);
    }

    /// Free a handle. Increments the generation for that slot (invalidating
    /// all outstanding handles) and pushes the index back onto the free list.
    void free(HandleType handle) noexcept {
        PULSE_ASSERT(!handle.isNull());
        PULSE_ASSERT_MSG(isValid(handle), "HandlePool::free: stale or invalid handle");

        uint32_t index = handle.index();

        // Increment generation to invalidate outstanding handles.
        generations_[index]++;

        // Push back onto free list.
        PULSE_ASSERT(freeListHead_ > 0);
        freeListHead_--;
        freeList_[freeListHead_] = index;
        freeCount_++;
        count_--;
    }

    // ── Validation ───────────────────────────────────────────────────────

    /// Check if a handle is currently valid (matches the current generation).
    [[nodiscard]] PULSE_FORCE_INLINE bool isValid(HandleType handle) const noexcept {
        if (handle.isNull()) return false;
        uint32_t index = handle.index();
        if (index >= capacity_) return false;
        return handle.generation() == generations_[index];
    }

    // ── Queries ──────────────────────────────────────────────────────────

    /// Number of currently active (allocated) handles.
    [[nodiscard]] std::size_t count() const noexcept { return count_; }

    /// Total capacity (active + free slots).
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

    /// Number of free slots available.
    [[nodiscard]] std::size_t freeSlots() const noexcept { return freeCount_; }

    // ── Iteration ────────────────────────────────────────────────────────

    /// Iterate over all currently active handles. Calls func(HandleType)
    /// for each active slot.
    ///
    /// This iterates all capacity slots and checks which are active (their
    /// generation matches what would be returned by allocate). An active
    /// slot is one where the index is NOT in the free list. We track this
    /// by checking if the generation is even (allocated) vs odd (freed),
    /// since each free increments the generation.
    ///
    /// Actually, a simpler approach: a slot is active if its index does not
    /// appear in the free list. Rather than searching the free list (O(n)),
    /// we use the fact that allocated slots have an even number of free()
    /// calls (generation matches the allocate-time generation). We maintain
    /// a separate active-tracking mechanism.
    ///
    /// Simplest correct approach: scan all indices, skip those whose current
    /// generation doesn't match any outstanding handle. Since we can't know
    /// the "allocated generation" without extra storage, we use a dense
    /// active list or bitmap.
    ///
    /// For simplicity and correctness, we maintain a boolean active array.
    template <typename Func>
    void forEach(Func&& func) const {
        // We need to determine which slots are active. A slot is active
        // if it was allocated and not yet freed. We can determine this
        // by checking if the index appears in the free list portion
        // [freeListHead_, freeListHead_ + freeCount_).
        // For efficiency, we build a temporary bitset. For pools up to
        // a few thousand entries, this is fast enough.
        //
        // Alternative: iterate all slots and call the function for slots
        // where the generation is what was given at allocation time.
        // Since allocate() gives the current generation and free() increments
        // it, a slot is active iff its current generation == the generation
        // it had when last allocated. But after free(), generation is bumped,
        // so an active slot has generation == generations_[index], which is
        // tautologically true for all slots. We need external tracking.
        //
        // Practical solution: maintain a simple boolean/bitfield. But to
        // keep the implementation lean, we'll check the free list.

        // Build a set of free indices for O(1) lookup.
        // Use stack allocation for small pools, heap for large.
        constexpr std::size_t StackThreshold = 4096;
        bool stackBuf[StackThreshold];
        bool* isActive;

        if (capacity_ <= StackThreshold) {
            isActive = stackBuf;
        } else {
            isActive = static_cast<bool*>(std::malloc(capacity_ * sizeof(bool)));
            PULSE_ASSERT(isActive != nullptr);
        }

        // Mark all as active.
        std::memset(isActive, 1, capacity_ * sizeof(bool));

        // Mark free slots as inactive.
        for (uint32_t i = 0; i < freeCount_; ++i) {
            uint32_t freeIdx = freeList_[freeListHead_ + i];
            isActive[freeIdx] = false;
        }

        // Iterate active slots.
        for (std::size_t i = 0; i < capacity_; ++i) {
            if (isActive[i]) {
                func(HandleType(static_cast<uint32_t>(i), generations_[i]));
            }
        }

        if (capacity_ > StackThreshold) {
            std::free(isActive);
        }
    }

    // ── Reset ────────────────────────────────────────────────────────────

    /// Reset the pool — all handles become invalid. Does NOT reduce capacity.
    void reset() noexcept {
        // Increment all generations to invalidate outstanding handles.
        for (std::size_t i = 0; i < capacity_; ++i) {
            generations_[i]++;
            freeList_[i] = static_cast<uint32_t>(i);
        }
        freeListHead_ = 0;
        freeCount_ = static_cast<uint32_t>(capacity_);
        count_ = 0;
    }

private:
    /// Grow the pool by doubling capacity.
    void grow() noexcept {
        std::size_t newCapacity = capacity_ * 2;

        auto* newGenerations = static_cast<uint32_t*>(
            std::realloc(generations_, newCapacity * sizeof(uint32_t)));
        PULSE_ASSERT_MSG(newGenerations != nullptr, "HandlePool::grow: realloc failed");
        generations_ = newGenerations;

        // Build a completely new free list with all slots.
        auto* newFreeList = static_cast<uint32_t*>(
            std::realloc(freeList_, newCapacity * sizeof(uint32_t)));
        PULSE_ASSERT_MSG(newFreeList != nullptr, "HandlePool::grow: realloc failed");
        freeList_ = newFreeList;

        // Initialize new slots.
        for (std::size_t i = capacity_; i < newCapacity; ++i) {
            generations_[i] = 0;
        }

        // Rebuild free list: existing free entries stay, add new entries.
        // Current free entries are at [freeListHead_, freeListHead_ + freeCount_).
        // Move them to the front, then append new entries.
        if (freeListHead_ > 0 && freeCount_ > 0) {
            std::memmove(freeList_, freeList_ + freeListHead_,
                         freeCount_ * sizeof(uint32_t));
        }
        freeListHead_ = 0;

        // Append newly created slots to the free list.
        for (std::size_t i = capacity_; i < newCapacity; ++i) {
            freeList_[freeCount_] = static_cast<uint32_t>(i);
            freeCount_++;
        }

        capacity_ = newCapacity;
    }

    uint32_t* generations_ = nullptr;  ///< Generation counter per slot.
    uint32_t* freeList_    = nullptr;  ///< Stack of free indices.
    std::size_t capacity_  = 0;        ///< Total number of slots.
    std::size_t count_     = 0;        ///< Number of active handles.
    uint32_t freeCount_    = 0;        ///< Number of free slots.
    uint32_t freeListHead_ = 0;        ///< Index into freeList_ for the next free slot.
};

} // namespace util
} // namespace pulse
