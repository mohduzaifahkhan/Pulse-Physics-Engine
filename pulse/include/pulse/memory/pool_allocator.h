/**
 * @file pool_allocator.h
 * @brief Fixed-size block pool allocator.
 *
 * Pre-allocates a contiguous block of memory and divides it into equally-sized
 * chunks. Allocation and deallocation are O(1) — allocate pops from a free list,
 * deallocate pushes back. No fragmentation is possible because all blocks are
 * the same size.
 *
 * Ideal for: rigid bodies, contacts, constraints, broadphase proxies — any
 * object type where you allocate/free many instances of the same size.
 *
 * Memory layout:
 * ┌─────────┬─────────┬─────────┬─────────┬─────────┐
 * │ Block 0 │ Block 1 │ Block 2 │  ...    │ Block N │
 * └─────────┴─────────┴─────────┴─────────┴─────────┘
 *
 * Free blocks form an intrusive singly-linked list (the first 8 bytes of each
 * free block store a pointer to the next free block). This means the minimum
 * block size is sizeof(void*) = 8 bytes.
 *
 * Thread safety: NOT thread-safe. Use one pool per thread, or wrap with a lock.
 */

#pragma once

#include "allocator_base.h"

namespace pulse {
namespace memory {

/**
 * @class PoolAllocator
 * @brief O(1) fixed-size block allocator with zero fragmentation.
 *
 * @tparam BlockSize Size of each block in bytes (will be rounded up to alignment).
 * @tparam Alignment Alignment of each block (default: 16 for SIMD).
 */
template <std::size_t BlockSize, std::size_t Alignment = DefaultAlignment>
class PoolAllocator : public AllocatorBase<PoolAllocator<BlockSize, Alignment>> {
    friend class AllocatorBase<PoolAllocator<BlockSize, Alignment>>;

    static_assert(isPowerOf2(Alignment), "Alignment must be a power of 2");
    static_assert(Alignment >= alignof(void*), "Alignment must be >= pointer size");

public:
    /// Actual block size after alignment padding.
    static constexpr std::size_t AlignedBlockSize =
        alignUp(BlockSize < sizeof(void*) ? sizeof(void*) : BlockSize, Alignment);

    /**
     * @brief Construct a pool allocator.
     * @param maxBlocks Maximum number of blocks the pool can hold.
     *
     * Allocates (maxBlocks * AlignedBlockSize) bytes of backing memory
     * and initializes the free list.
     */
    explicit PoolAllocator(std::size_t maxBlocks) noexcept
        : maxBlocks_(maxBlocks),
          memory_(nullptr),
          freeList_(nullptr),
          ownsMemory_(true)
    {
        totalSize_ = maxBlocks * AlignedBlockSize;
        memory_ = static_cast<uint8_t*>(platformAlignedAlloc(totalSize_, Alignment));
        assert(memory_ != nullptr && "PoolAllocator: failed to allocate backing memory");
        initFreeList();
    }

    /**
     * @brief Construct a pool allocator from externally provided memory.
     * @param memory Pointer to pre-allocated memory.
     * @param memorySize Size of the memory block in bytes.
     *
     * The memory is NOT owned by this allocator — the caller is responsible
     * for freeing it after the allocator is destroyed.
     */
    PoolAllocator(void* memory, std::size_t memorySize) noexcept
        : maxBlocks_(memorySize / AlignedBlockSize),
          totalSize_(maxBlocks_ * AlignedBlockSize),
          memory_(static_cast<uint8_t*>(memory)),
          freeList_(nullptr),
          ownsMemory_(false)
    {
        assert(memory != nullptr);
        assert(isAligned(memory, Alignment));
        initFreeList();
    }

    ~PoolAllocator() noexcept {
        if (ownsMemory_ && memory_) {
            platformAlignedFree(memory_);
        }
    }

    // Non-copyable, movable
    PoolAllocator(const PoolAllocator&) = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;

    PoolAllocator(PoolAllocator&& other) noexcept
        : maxBlocks_(other.maxBlocks_),
          totalSize_(other.totalSize_),
          memory_(other.memory_),
          freeList_(other.freeList_),
          ownsMemory_(other.ownsMemory_)
    {
        this->stats_ = other.stats_;
        other.memory_ = nullptr;
        other.freeList_ = nullptr;
        other.ownsMemory_ = false;
    }

    PoolAllocator& operator=(PoolAllocator&& other) noexcept {
        if (this != &other) {
            if (ownsMemory_ && memory_) {
                platformAlignedFree(memory_);
            }
            maxBlocks_ = other.maxBlocks_;
            totalSize_ = other.totalSize_;
            memory_ = other.memory_;
            freeList_ = other.freeList_;
            ownsMemory_ = other.ownsMemory_;
            this->stats_ = other.stats_;
            other.memory_ = nullptr;
            other.freeList_ = nullptr;
            other.ownsMemory_ = false;
        }
        return *this;
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    /// Number of blocks currently allocated.
    [[nodiscard]] PULSE_FORCE_INLINE std::size_t allocatedBlocks() const noexcept {
        return this->stats_.totalAllocations - this->stats_.totalDeallocations;
    }

    /// Number of free blocks remaining.
    [[nodiscard]] PULSE_FORCE_INLINE std::size_t freeBlocks() const noexcept {
        return maxBlocks_ - allocatedBlocks();
    }

    /// Maximum number of blocks.
    [[nodiscard]] PULSE_FORCE_INLINE std::size_t maxBlockCount() const noexcept {
        return maxBlocks_;
    }

    /// Is the pool full?
    [[nodiscard]] PULSE_FORCE_INLINE bool isFull() const noexcept {
        return freeList_ == nullptr;
    }

private:
    // ── CRTP interface implementation ─────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE void* doAllocate(
        std::size_t size, std::size_t /*alignment*/
    ) noexcept {
        assert(size <= AlignedBlockSize && "Requested size exceeds pool block size");
        (void)size;

        if (PULSE_UNLIKELY(freeList_ == nullptr)) {
            return nullptr; // Pool exhausted
        }

        // Pop from free list — O(1)
        void* block = freeList_;
        freeList_ = *reinterpret_cast<void**>(freeList_);

        this->stats_.recordAllocation(AlignedBlockSize, AlignedBlockSize - size);
        return block;
    }

    PULSE_FORCE_INLINE void doDeallocate(void* ptr) noexcept {
        assert(doOwns(ptr) && "Pointer does not belong to this pool");

        // Push onto free list — O(1)
        *reinterpret_cast<void**>(ptr) = freeList_;
        freeList_ = ptr;

        this->stats_.recordDeallocation(AlignedBlockSize);
    }

    PULSE_FORCE_INLINE void doDeallocate(void* ptr, std::size_t /*size*/) noexcept {
        doDeallocate(ptr);
    }

    PULSE_FORCE_INLINE void doReset() noexcept {
        this->stats_.reset();
        initFreeList();
    }

    [[nodiscard]] PULSE_FORCE_INLINE bool doOwns(const void* ptr) const noexcept {
        const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
        const auto start = reinterpret_cast<std::uintptr_t>(memory_);
        return addr >= start && addr < (start + totalSize_);
    }

    [[nodiscard]] PULSE_FORCE_INLINE std::size_t doCapacity() const noexcept {
        return totalSize_;
    }

    // ── Internal ──────────────────────────────────────────────────────────

    void initFreeList() noexcept {
        freeList_ = nullptr;
        // Build free list from last block to first so that the first allocation
        // returns the first block (cache-friendly sequential access).
        for (std::size_t i = maxBlocks_; i > 0; --i) {
            uint8_t* block = memory_ + (i - 1) * AlignedBlockSize;
            *reinterpret_cast<void**>(block) = freeList_;
            freeList_ = block;
        }
    }

    std::size_t maxBlocks_;
    std::size_t totalSize_;
    uint8_t*    memory_;
    void*       freeList_;
    bool        ownsMemory_;
};

} // namespace memory
} // namespace pulse
