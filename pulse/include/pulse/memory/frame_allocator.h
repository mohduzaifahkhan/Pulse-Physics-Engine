/**
 * @file frame_allocator.h
 * @brief Double-buffered frame allocator for per-frame temporaries.
 *
 * Maintains two arena allocators and swaps between them each frame. While the
 * current frame allocates from arena A, the previous frame's data in arena B
 * is still valid (useful for interpolation, debug rendering). At the start of
 * each frame, the "old" arena is reset and becomes the new current arena.
 *
 * This eliminates ALL per-frame allocations from hitting the heap. Contact
 * points, broadphase pair buffers, narrow-phase results, solver temporaries
 * — everything can be allocated from the frame allocator and automatically
 * reclaimed next frame.
 *
 * Memory layout:
 * ┌─────────────────────┐  ┌─────────────────────┐
 * │  Arena A (current)  │  │  Arena B (previous)  │
 * │  [alloc][alloc]...  │  │  [prev frame data]   │
 * └─────────────────────┘  └─────────────────────┘
 *
 * Thread safety: NOT thread-safe. Each thread should have its own frame allocator,
 * or use the per-thread frame allocators from the job system.
 */

#pragma once

#include "arena_allocator.h"

namespace pulse {
namespace memory {

/**
 * @class FrameAllocator
 * @brief Double-buffered arena allocator that resets each frame.
 */
class FrameAllocator {
public:
    /**
     * @brief Construct a frame allocator with the given per-arena capacity.
     * @param arenaCapacity Capacity for each of the two internal arenas.
     *
     * Total memory usage is 2 * arenaCapacity.
     */
    explicit FrameAllocator(std::size_t arenaCapacity) noexcept
        : arenas_{ArenaAllocator(arenaCapacity), ArenaAllocator(arenaCapacity)},
          currentIndex_(0),
          frameNumber_(0)
    {}

    // Non-copyable
    FrameAllocator(const FrameAllocator&) = delete;
    FrameAllocator& operator=(const FrameAllocator&) = delete;

    // Movable
    FrameAllocator(FrameAllocator&& other) noexcept
        : arenas_{std::move(other.arenas_[0]), std::move(other.arenas_[1])},
          currentIndex_(other.currentIndex_),
          frameNumber_(other.frameNumber_)
    {}

    FrameAllocator& operator=(FrameAllocator&& other) noexcept {
        if (this != &other) {
            arenas_[0] = std::move(other.arenas_[0]);
            arenas_[1] = std::move(other.arenas_[1]);
            currentIndex_ = other.currentIndex_;
            frameNumber_ = other.frameNumber_;
        }
        return *this;
    }

    // ── Frame lifecycle ───────────────────────────────────────────────────

    /**
     * @brief Begin a new frame.
     *
     * Swaps the current and previous arenas, then resets the new current
     * arena. Must be called once at the start of each simulation step.
     * Previous frame's data remains valid in the other arena until the
     * NEXT call to beginFrame().
     */
    PULSE_FORCE_INLINE void beginFrame() noexcept {
        // Swap arenas
        currentIndex_ = 1 - currentIndex_;
        // Reset the arena that will be used this frame
        arenas_[currentIndex_].reset();
        frameNumber_++;
    }

    // ── Allocation ────────────────────────────────────────────────────────

    /// Allocate from the current frame's arena.
    [[nodiscard]] PULSE_FORCE_INLINE void* allocate(
        std::size_t size,
        std::size_t alignment = DefaultAlignment
    ) noexcept {
        return arenas_[currentIndex_].allocate(size, alignment);
    }

    /// Typed allocation: allocate and construct.
    template <typename T, typename... Args>
    [[nodiscard]] PULSE_FORCE_INLINE T* create(Args&&... args) noexcept {
        return arenas_[currentIndex_].create<T>(std::forward<Args>(args)...);
    }

    /// Allocate an array of N objects (no construction).
    template <typename T>
    [[nodiscard]] PULSE_FORCE_INLINE T* allocateArray(std::size_t count) noexcept {
        return arenas_[currentIndex_].allocateArray<T>(count);
    }

    /// Deallocate is a no-op (memory is freed by beginFrame).
    PULSE_FORCE_INLINE void deallocate(void* /*ptr*/) noexcept {
        // No-op: frame allocators free everything at once via beginFrame().
    }

    // ── Queries ───────────────────────────────────────────────────────────

    /// Get the current frame's arena.
    [[nodiscard]] PULSE_FORCE_INLINE ArenaAllocator& currentArena() noexcept {
        return arenas_[currentIndex_];
    }

    /// Get the previous frame's arena (data is still valid until next beginFrame).
    [[nodiscard]] PULSE_FORCE_INLINE ArenaAllocator& previousArena() noexcept {
        return arenas_[1 - currentIndex_];
    }

    /// Remaining bytes in the current frame's arena.
    [[nodiscard]] PULSE_FORCE_INLINE std::size_t remaining() const noexcept {
        return arenas_[currentIndex_].remaining();
    }

    /// Per-arena capacity.
    [[nodiscard]] PULSE_FORCE_INLINE std::size_t arenaCapacity() const noexcept {
        return arenas_[0].capacity();
    }

    /// Total capacity (both arenas).
    [[nodiscard]] PULSE_FORCE_INLINE std::size_t totalCapacity() const noexcept {
        return arenas_[0].capacity() + arenas_[1].capacity();
    }

    /// Current frame number (starts at 0, increments per beginFrame call).
    [[nodiscard]] PULSE_FORCE_INLINE uint64_t frameNumber() const noexcept {
        return frameNumber_;
    }

    /// Get stats for the current frame's arena.
    [[nodiscard]] PULSE_FORCE_INLINE const AllocatorStats& currentStats() const noexcept {
        return arenas_[currentIndex_].getStats();
    }

private:
    ArenaAllocator arenas_[2]; ///< Double-buffered arenas.
    int currentIndex_;         ///< Index of the currently active arena (0 or 1).
    uint64_t frameNumber_;     ///< Frame counter.
};

} // namespace memory
} // namespace pulse
