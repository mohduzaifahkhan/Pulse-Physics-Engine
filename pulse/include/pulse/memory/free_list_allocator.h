/**
 * @file free_list_allocator.h
 * @brief General-purpose free-list allocator with variable-size allocations.
 *
 * Manages a contiguous memory block by maintaining a sorted linked list of
 * free regions. Supports arbitrary-sized allocations and deallocations with
 * automatic coalescing of adjacent free blocks to minimize fragmentation.
 *
 * Allocation strategies:
 * - First-Fit:  Fast — picks the first block that fits.
 * - Best-Fit:   Less fragmentation — picks the smallest block that fits.
 *
 * Each allocated block has a small header (16 bytes) storing the block size
 * and alignment adjustment. Free blocks use an intrusive linked list.
 *
 * Ideal for: dynamic shape creation, convex hull storage, triangle mesh data,
 * and any allocation pattern where objects have variable sizes and lifetimes.
 *
 * Memory layout:
 * ┌──────────────────────────────────────────────────────────┐
 * │ [H|used] [H|free──────] [H|used] [H|free] [H|used]     │
 * └──────────────────────────────────────────────────────────┘
 * H = header (16 bytes), linked list threads through free blocks.
 *
 * Thread safety: NOT thread-safe.
 */

#pragma once

#include "allocator_base.h"

namespace pulse {
namespace memory {

/// Allocation search strategy.
enum class FitStrategy : uint8_t {
    FirstFit, ///< Pick the first free block large enough (faster).
    BestFit   ///< Pick the smallest free block large enough (less fragmentation).
};

/**
 * @class FreeListAllocator
 * @brief Variable-size allocator with free block coalescing.
 */
class FreeListAllocator : public AllocatorBase<FreeListAllocator> {
    friend class AllocatorBase<FreeListAllocator>;

public:
    /**
     * @brief Construct a free list allocator.
     * @param capacityBytes Total backing memory size.
     * @param strategy Allocation strategy (FirstFit or BestFit).
     */
    explicit FreeListAllocator(
        std::size_t capacityBytes,
        FitStrategy strategy = FitStrategy::FirstFit
    ) noexcept
        : memory_(nullptr),
          capacity_(capacityBytes),
          freeList_(nullptr),
          strategy_(strategy),
          ownsMemory_(true)
    {
        memory_ = static_cast<uint8_t*>(
            platformAlignedAlloc(capacityBytes, CacheLineSize)
        );
        assert(memory_ != nullptr && "FreeListAllocator: failed to allocate backing memory");
        initFreeList();
    }

    FreeListAllocator(
        void* memory,
        std::size_t capacityBytes,
        FitStrategy strategy = FitStrategy::FirstFit
    ) noexcept
        : memory_(static_cast<uint8_t*>(memory)),
          capacity_(capacityBytes),
          freeList_(nullptr),
          strategy_(strategy),
          ownsMemory_(false)
    {
        assert(memory != nullptr);
        initFreeList();
    }

    ~FreeListAllocator() noexcept {
        if (ownsMemory_ && memory_) {
            platformAlignedFree(memory_);
        }
    }

    FreeListAllocator(const FreeListAllocator&) = delete;
    FreeListAllocator& operator=(const FreeListAllocator&) = delete;

    FreeListAllocator(FreeListAllocator&& other) noexcept
        : memory_(other.memory_),
          capacity_(other.capacity_),
          freeList_(other.freeList_),
          strategy_(other.strategy_),
          ownsMemory_(other.ownsMemory_)
    {
        this->stats_ = other.stats_;
        other.memory_ = nullptr;
        other.freeList_ = nullptr;
        other.ownsMemory_ = false;
    }

    FreeListAllocator& operator=(FreeListAllocator&& other) noexcept {
        if (this != &other) {
            if (ownsMemory_ && memory_) platformAlignedFree(memory_);
            memory_ = other.memory_;
            capacity_ = other.capacity_;
            freeList_ = other.freeList_;
            strategy_ = other.strategy_;
            ownsMemory_ = other.ownsMemory_;
            this->stats_ = other.stats_;
            other.memory_ = nullptr;
            other.freeList_ = nullptr;
            other.ownsMemory_ = false;
        }
        return *this;
    }

    /// Change allocation strategy at runtime.
    PULSE_FORCE_INLINE void setStrategy(FitStrategy s) noexcept {
        strategy_ = s;
    }

    /// Get current strategy.
    [[nodiscard]] PULSE_FORCE_INLINE FitStrategy getStrategy() const noexcept {
        return strategy_;
    }

    /// Count number of free blocks (O(n), for diagnostics only).
    [[nodiscard]] std::size_t freeBlockCount() const noexcept {
        std::size_t count = 0;
        FreeBlock* node = freeList_;
        while (node) {
            count++;
            node = node->next;
        }
        return count;
    }

    /// Total free bytes (O(n), for diagnostics only).
    [[nodiscard]] std::size_t totalFreeBytes() const noexcept {
        std::size_t total = 0;
        FreeBlock* node = freeList_;
        while (node) {
            total += node->size;
            node = node->next;
        }
        return total;
    }

private:
    // ── Internal types ────────────────────────────────────────────────────

    /// Header prepended to every allocation (both used and free).
    struct BlockHeader {
        std::size_t size;       ///< Total block size including header.
        std::size_t adjustment; ///< Alignment adjustment in bytes.
    };
    static constexpr std::size_t HeaderSize = sizeof(BlockHeader);

    /// Free block node (intrusive linked list, stored inside free memory).
    struct FreeBlock {
        std::size_t size; ///< Total size of this free region (including this struct).
        FreeBlock*  next; ///< Next free block (sorted by address).
    };
    static constexpr std::size_t MinFreeBlockSize =
        sizeof(FreeBlock) > HeaderSize ? sizeof(FreeBlock) : HeaderSize;

    // ── CRTP interface ────────────────────────────────────────────────────

    [[nodiscard]] void* doAllocate(std::size_t size, std::size_t alignment) noexcept {
        // Total size needed: header + alignment padding + requested size
        const std::size_t totalNeeded = HeaderSize + size;

        FreeBlock* prev = nullptr;
        FreeBlock* best = nullptr;
        FreeBlock* bestPrev = nullptr;

        if (strategy_ == FitStrategy::FirstFit) {
            // First-fit: find the first block that can satisfy the request
            FreeBlock* node = freeList_;
            while (node) {
                std::size_t adjustment = calculateAdjustment(
                    reinterpret_cast<std::uintptr_t>(node) + HeaderSize, alignment
                );
                std::size_t neededWithAdj = totalNeeded + adjustment;

                if (node->size >= neededWithAdj) {
                    best = node;
                    bestPrev = prev;
                    break;
                }
                prev = node;
                node = node->next;
            }
        } else {
            // Best-fit: find the smallest block that satisfies the request
            std::size_t bestSize = ~std::size_t(0);
            FreeBlock* node = freeList_;
            while (node) {
                std::size_t adjustment = calculateAdjustment(
                    reinterpret_cast<std::uintptr_t>(node) + HeaderSize, alignment
                );
                std::size_t neededWithAdj = totalNeeded + adjustment;

                if (node->size >= neededWithAdj && node->size < bestSize) {
                    best = node;
                    bestPrev = prev;
                    bestSize = node->size;
                    if (node->size == neededWithAdj) break; // Perfect fit
                }
                prev = node;
                node = node->next;
            }
        }

        if (PULSE_UNLIKELY(best == nullptr)) {
            return nullptr; // No suitable block found
        }

        // Calculate alignment adjustment for the data pointer
        std::size_t adjustment = calculateAdjustment(
            reinterpret_cast<std::uintptr_t>(best) + HeaderSize, alignment
        );
        std::size_t allocSize = totalNeeded + adjustment;
        std::size_t remainder = best->size - allocSize;

        // If the remainder is too small to hold a free block, absorb it
        if (remainder < MinFreeBlockSize + HeaderSize) {
            allocSize = best->size;
            remainder = 0;
        }

        // Remove this block from the free list
        if (bestPrev) {
            bestPrev->next = best->next;
        } else {
            freeList_ = best->next;
        }

        // If there's a remainder, create a new free block
        if (remainder > 0) {
            auto* newFreeBlock = reinterpret_cast<FreeBlock*>(
                reinterpret_cast<uint8_t*>(best) + allocSize
            );
            newFreeBlock->size = remainder;
            newFreeBlock->next = nullptr;
            insertFreeBlock(newFreeBlock);
        }

        // Write the allocation header
        auto headerAddr = reinterpret_cast<std::uintptr_t>(best) + adjustment;
        auto* header = reinterpret_cast<BlockHeader*>(headerAddr);
        header->size = allocSize;
        header->adjustment = adjustment;

        void* dataPtr = reinterpret_cast<void*>(headerAddr + HeaderSize);
        std::size_t waste = allocSize - size - HeaderSize;
        stats_.recordAllocation(size, waste);

        return dataPtr;
    }

    void doDeallocate(void* ptr) noexcept {
        assert(doOwns(ptr) && "Pointer does not belong to this allocator");

        // Get the header
        auto* header = reinterpret_cast<BlockHeader*>(
            reinterpret_cast<uint8_t*>(ptr) - HeaderSize
        );
        std::size_t blockSize = header->size;
        std::size_t adjustment = header->adjustment;
        std::size_t userSize = blockSize - HeaderSize - adjustment;

        // Create a free block at the original position (before adjustment)
        auto* freeBlock = reinterpret_cast<FreeBlock*>(
            reinterpret_cast<uint8_t*>(header) - adjustment
        );
        freeBlock->size = blockSize;
        freeBlock->next = nullptr;

        insertFreeBlock(freeBlock);
        coalesce(freeBlock);

        stats_.recordDeallocation(userSize);
    }

    void doDeallocate(void* ptr, std::size_t /*size*/) noexcept {
        doDeallocate(ptr);
    }

    void doReset() noexcept {
        stats_.reset();
        initFreeList();
    }

    [[nodiscard]] bool doOwns(const void* ptr) const noexcept {
        const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
        const auto start = reinterpret_cast<std::uintptr_t>(memory_);
        return addr >= start && addr < (start + capacity_);
    }

    [[nodiscard]] std::size_t doCapacity() const noexcept {
        return capacity_;
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    void initFreeList() noexcept {
        freeList_ = reinterpret_cast<FreeBlock*>(memory_);
        freeList_->size = capacity_;
        freeList_->next = nullptr;
    }

    /// Calculate the number of bytes needed to align `addr` to `alignment`.
    static std::size_t calculateAdjustment(
        std::uintptr_t addr, std::size_t alignment
    ) noexcept {
        std::size_t mask = alignment - 1;
        std::size_t misalignment = addr & mask;
        return (misalignment == 0) ? 0 : (alignment - misalignment);
    }

    /// Insert a free block into the sorted free list (sorted by address).
    void insertFreeBlock(FreeBlock* block) noexcept {
        auto blockAddr = reinterpret_cast<std::uintptr_t>(block);

        // Find insertion point (maintain address-sorted order for coalescing)
        FreeBlock* prev = nullptr;
        FreeBlock* curr = freeList_;
        while (curr && reinterpret_cast<std::uintptr_t>(curr) < blockAddr) {
            prev = curr;
            curr = curr->next;
        }

        block->next = curr;
        if (prev) {
            prev->next = block;
        } else {
            freeList_ = block;
        }
    }

    /// Coalesce adjacent free blocks to reduce fragmentation.
    void coalesce(FreeBlock* block) noexcept {
        // Merge with the next block if adjacent
        if (block->next) {
            auto blockEnd = reinterpret_cast<std::uintptr_t>(block) + block->size;
            auto nextAddr = reinterpret_cast<std::uintptr_t>(block->next);
            if (blockEnd == nextAddr) {
                block->size += block->next->size;
                block->next = block->next->next;
            }
        }

        // Find the previous block and merge if adjacent
        FreeBlock* prev = nullptr;
        FreeBlock* curr = freeList_;
        while (curr && curr != block) {
            prev = curr;
            curr = curr->next;
        }

        if (prev) {
            auto prevEnd = reinterpret_cast<std::uintptr_t>(prev) + prev->size;
            auto blockAddr = reinterpret_cast<std::uintptr_t>(block);
            if (prevEnd == blockAddr) {
                prev->size += block->size;
                prev->next = block->next;
            }
        }
    }

    uint8_t*    memory_;
    std::size_t capacity_;
    FreeBlock*  freeList_;
    FitStrategy strategy_;
    bool        ownsMemory_;
};

} // namespace memory
} // namespace pulse
