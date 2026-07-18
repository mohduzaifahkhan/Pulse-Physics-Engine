/**
 * @file stack_allocator.h
 * @brief LIFO stack allocator with O(1) alloc/dealloc.
 *
 * Allocates linearly like an arena, but supports LIFO (last-in, first-out)
 * deallocation. Each allocation records its previous offset in a small header,
 * allowing deallocate() to roll back to the previous state.
 *
 * Constraint: Deallocations MUST happen in reverse order of allocations
 * (LIFO). The allocator asserts this in debug mode.
 *
 * Ideal for: recursive algorithms (BVH traversal, island detection),
 * nested scope allocations, solver temporaries.
 *
 * Memory layout:
 * ┌──────────────────────────────────────────────────────────┐
 * │ [hdr|data0] [hdr|data1] [hdr|data2] ... [free space]   │
 * │                                          ^ offset       │
 * └──────────────────────────────────────────────────────────┘
 *
 * Header per allocation: 16 bytes (previous offset + size + padding).
 *
 * Thread safety: NOT thread-safe.
 */

#pragma once

#include "allocator_base.h"

namespace pulse {
namespace memory {

/**
 * @class StackAllocator
 * @brief LIFO allocator with O(1) push/pop operations.
 */
class StackAllocator : public AllocatorBase<StackAllocator> {
    friend class AllocatorBase<StackAllocator>;

public:
    explicit StackAllocator(std::size_t capacityBytes) noexcept
        : memory_(nullptr),
          capacity_(capacityBytes),
          offset_(0),
          ownsMemory_(true)
    {
        memory_ = static_cast<uint8_t*>(
            platformAlignedAlloc(capacityBytes, CacheLineSize)
        );
        assert(memory_ != nullptr && "StackAllocator: failed to allocate backing memory");
    }

    StackAllocator(void* memory, std::size_t capacityBytes) noexcept
        : memory_(static_cast<uint8_t*>(memory)),
          capacity_(capacityBytes),
          offset_(0),
          ownsMemory_(false)
    {
        assert(memory != nullptr);
    }

    ~StackAllocator() noexcept {
        if (ownsMemory_ && memory_) {
            platformAlignedFree(memory_);
        }
    }

    StackAllocator(const StackAllocator&) = delete;
    StackAllocator& operator=(const StackAllocator&) = delete;

    StackAllocator(StackAllocator&& other) noexcept
        : memory_(other.memory_),
          capacity_(other.capacity_),
          offset_(other.offset_),
          ownsMemory_(other.ownsMemory_)
    {
        this->stats_ = other.stats_;
        other.memory_ = nullptr;
        other.ownsMemory_ = false;
    }

    StackAllocator& operator=(StackAllocator&& other) noexcept {
        if (this != &other) {
            if (ownsMemory_ && memory_) platformAlignedFree(memory_);
            memory_ = other.memory_;
            capacity_ = other.capacity_;
            offset_ = other.offset_;
            ownsMemory_ = other.ownsMemory_;
            this->stats_ = other.stats_;
            other.memory_ = nullptr;
            other.ownsMemory_ = false;
        }
        return *this;
    }

    /// Current offset into the buffer.
    [[nodiscard]] PULSE_FORCE_INLINE std::size_t currentOffset() const noexcept {
        return offset_;
    }

    /// Remaining capacity.
    [[nodiscard]] PULSE_FORCE_INLINE std::size_t remaining() const noexcept {
        return capacity_ - offset_;
    }

    /// Get a scope marker for scoped allocations.
    struct ScopeMarker {
        std::size_t offset;
        AllocatorStats stats;
    };

    /// Save current state. All allocations after this can be rolled back.
    [[nodiscard]] PULSE_FORCE_INLINE ScopeMarker mark() const noexcept {
        return {offset_, stats_};
    }

    /// Roll back to a previous scope marker, freeing everything allocated after it.
    PULSE_FORCE_INLINE void rollback(const ScopeMarker& marker) noexcept {
        assert(marker.offset <= offset_ && "Invalid scope marker");
        offset_ = marker.offset;
        stats_ = marker.stats;
    }

private:
    /// Header stored before each allocation for LIFO deallocation.
    struct AllocationHeader {
        std::size_t prevOffset; ///< Offset before this allocation (including header).
        std::size_t size;       ///< User-requested size (for stats).
    };
    static constexpr std::size_t HeaderSize = alignUp(sizeof(AllocationHeader), DefaultAlignment);

    [[nodiscard]] PULSE_FORCE_INLINE void* doAllocate(
        std::size_t size, std::size_t alignment
    ) noexcept {
        // Calculate aligned position for the header
        std::size_t headerStart = alignUp(offset_, alignof(AllocationHeader));
        std::size_t dataStart = alignUp(headerStart + HeaderSize, alignment);
        std::size_t newOffset = dataStart + size;

        if (PULSE_UNLIKELY(newOffset > capacity_)) {
            return nullptr;
        }

        // Write header
        auto* header = reinterpret_cast<AllocationHeader*>(memory_ + headerStart);
        header->prevOffset = offset_;
        header->size = size;

        std::size_t waste = dataStart - offset_ - size;
        offset_ = newOffset;
        stats_.recordAllocation(size, waste);

        return memory_ + dataStart;
    }

    PULSE_FORCE_INLINE void doDeallocate(void* ptr) noexcept {
        assert(doOwns(ptr) && "Pointer does not belong to this stack");

        // Find the header before this allocation
        auto headerAddr = reinterpret_cast<std::uintptr_t>(ptr) - HeaderSize;
        // Align down to header alignment
        headerAddr = headerAddr & ~(alignof(AllocationHeader) - 1);
        auto* header = reinterpret_cast<AllocationHeader*>(headerAddr);

        // Verify LIFO order: this deallocation should be for the most recent allocation
        assert(reinterpret_cast<std::uintptr_t>(ptr) + header->size <=
               reinterpret_cast<std::uintptr_t>(memory_) + offset_ &&
               "Stack deallocations must be in LIFO order");

        stats_.recordDeallocation(header->size);
        offset_ = header->prevOffset;
    }

    PULSE_FORCE_INLINE void doDeallocate(void* ptr, std::size_t /*size*/) noexcept {
        doDeallocate(ptr);
    }

    PULSE_FORCE_INLINE void doReset() noexcept {
        offset_ = 0;
        stats_.reset();
    }

    [[nodiscard]] PULSE_FORCE_INLINE bool doOwns(const void* ptr) const noexcept {
        const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
        const auto start = reinterpret_cast<std::uintptr_t>(memory_);
        return addr >= start && addr < (start + capacity_);
    }

    [[nodiscard]] PULSE_FORCE_INLINE std::size_t doCapacity() const noexcept {
        return capacity_;
    }

    uint8_t*    memory_;
    std::size_t capacity_;
    std::size_t offset_;
    bool        ownsMemory_;
};

/**
 * @class ScopedStack
 * @brief RAII guard that automatically rolls back a StackAllocator on scope exit.
 *
 * Usage:
 * @code
 *   StackAllocator stack(1024);
 *   {
 *       ScopedStack scope(stack);
 *       auto* temp = stack.allocateArray<float>(100);
 *       // ... use temp ...
 *   } // temp is automatically freed here
 * @endcode
 */
class ScopedStack {
public:
    explicit ScopedStack(StackAllocator& stack) noexcept
        : stack_(stack), marker_(stack.mark())
    {}

    ~ScopedStack() noexcept {
        stack_.rollback(marker_);
    }

    ScopedStack(const ScopedStack&) = delete;
    ScopedStack& operator=(const ScopedStack&) = delete;

private:
    StackAllocator& stack_;
    StackAllocator::ScopeMarker marker_;
};

} // namespace memory
} // namespace pulse
