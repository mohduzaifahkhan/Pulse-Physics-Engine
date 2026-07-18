/**
 * @file arena_allocator.h
 * @brief Linear (arena/bump) allocator — the fastest possible allocator.
 *
 * Allocates by simply bumping a pointer forward. Deallocation of individual
 * blocks is not supported — you can only free everything at once via reset().
 * This makes it O(1) per allocation with zero bookkeeping overhead.
 *
 * Ideal for: per-frame temporary allocations (contact points, narrow-phase
 * results, temporary arrays). Pair with ScopedAllocatorReset for RAII cleanup.
 *
 * Memory layout:
 * ┌────────────────────────────────────────────────────────┐
 * │ [alloc 0] [alloc 1] [alloc 2] ... [free space]        │
 * │                                    ^ offset            │
 * └────────────────────────────────────────────────────────┘
 *
 * Thread safety: NOT thread-safe. Designed for single-thread use. For
 * multi-threaded usage, give each thread its own arena.
 */

#pragma once

#include "allocator_base.h"

namespace pulse {
namespace memory {

/**
 * @class ArenaAllocator
 * @brief Ultra-fast linear allocator. O(1) alloc, no individual free.
 */
class ArenaAllocator : public AllocatorBase<ArenaAllocator> {
    friend class AllocatorBase<ArenaAllocator>;

public:
    /**
     * @brief Construct an arena allocator with the given capacity.
     * @param capacityBytes Total size of the backing memory.
     */
    explicit ArenaAllocator(std::size_t capacityBytes) noexcept
        : memory_(nullptr),
          capacity_(capacityBytes),
          offset_(0),
          ownsMemory_(true)
    {
        memory_ = static_cast<uint8_t*>(
            platformAlignedAlloc(capacityBytes, CacheLineSize)
        );
        assert(memory_ != nullptr && "ArenaAllocator: failed to allocate backing memory");
    }

    /**
     * @brief Construct from externally provided memory.
     * @param memory Pre-allocated memory block.
     * @param capacityBytes Size of the memory block.
     */
    ArenaAllocator(void* memory, std::size_t capacityBytes) noexcept
        : memory_(static_cast<uint8_t*>(memory)),
          capacity_(capacityBytes),
          offset_(0),
          ownsMemory_(false)
    {
        assert(memory != nullptr);
    }

    ~ArenaAllocator() noexcept {
        if (ownsMemory_ && memory_) {
            platformAlignedFree(memory_);
        }
    }

    // Non-copyable, movable
    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    ArenaAllocator(ArenaAllocator&& other) noexcept
        : memory_(other.memory_),
          capacity_(other.capacity_),
          offset_(other.offset_),
          ownsMemory_(other.ownsMemory_)
    {
        this->stats_ = other.stats_;
        other.memory_ = nullptr;
        other.ownsMemory_ = false;
    }

    ArenaAllocator& operator=(ArenaAllocator&& other) noexcept {
        if (this != &other) {
            if (ownsMemory_ && memory_) {
                platformAlignedFree(memory_);
            }
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

    // ── Accessors ─────────────────────────────────────────────────────────

    /// Current allocation offset (bytes used).
    [[nodiscard]] PULSE_FORCE_INLINE std::size_t currentOffset() const noexcept {
        return offset_;
    }

    /// Remaining bytes available.
    [[nodiscard]] PULSE_FORCE_INLINE std::size_t remaining() const noexcept {
        return capacity_ - offset_;
    }

    /// Save the current offset for later restoration (manual stack-like behavior).
    [[nodiscard]] PULSE_FORCE_INLINE std::size_t saveState() const noexcept {
        return offset_;
    }

    /// Restore to a previously saved offset (frees everything allocated after it).
    PULSE_FORCE_INLINE void restoreState(std::size_t savedOffset) noexcept {
        assert(savedOffset <= offset_ && "Cannot restore to a future state");
        std::size_t freed = offset_ - savedOffset;
        offset_ = savedOffset;
        stats_.recordDeallocation(freed);
    }

private:
    // ── CRTP interface ────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE void* doAllocate(
        std::size_t size, std::size_t alignment
    ) noexcept {
        // Align the current offset
        std::size_t alignedOffset = alignUp(offset_, alignment);
        std::size_t newOffset = alignedOffset + size;

        if (PULSE_UNLIKELY(newOffset > capacity_)) {
            return nullptr; // Out of memory
        }

        void* ptr = memory_ + alignedOffset;
        std::size_t waste = alignedOffset - offset_;
        offset_ = newOffset;

        stats_.recordAllocation(size, waste);
        return ptr;
    }

    PULSE_FORCE_INLINE void doDeallocate(void* /*ptr*/) noexcept {
        // Individual deallocation is not supported by arena allocators.
        // Memory is only freed via reset() or restoreState().
    }

    PULSE_FORCE_INLINE void doDeallocate(void* /*ptr*/, std::size_t /*size*/) noexcept {
        // Same — no-op for arena allocator.
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

} // namespace memory
} // namespace pulse
