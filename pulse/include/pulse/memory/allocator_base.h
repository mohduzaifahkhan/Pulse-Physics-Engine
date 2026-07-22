/**
 * @file allocator_base.h
 * @brief Base allocator interface and memory utilities.
 *
 * Provides the foundational types, alignment utilities, and the allocator
 * interface contract that all Pulse allocators implement. This is a CRTP base
 * — no virtual functions, no vtable overhead, fully inlinable.
 *
 * Key design principles:
 * - No virtual dispatch — CRTP compile-time polymorphism
 * - All allocations are aligned to at least 16 bytes (SIMD-ready)
 * - Debug mode tracks allocations and detects leaks
 * - All allocators work with raw memory blocks provided at construction
 */

#pragma once

#include <pulse/math/math_common.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

namespace pulse {
namespace memory {

// ── Constants
// ─────────────────────────────────────────────────────────────────

/// Default alignment for all physics allocations (SIMD-friendly).
constexpr std::size_t DefaultAlignment = 16;

/// Cache line size for padding and alignment.
constexpr std::size_t CacheLineSize = 64;

/// Minimum allocation granularity.
constexpr std::size_t MinAllocationSize = 16;

// ── Alignment utilities
// ───────────────────────────────────────────────────────

/// Align a value up to the given alignment (must be power of 2).
/// Works for both std::size_t and std::uintptr_t (handles platforms where
/// they are the same underlying type).
[[nodiscard]] PULSE_FORCE_INLINE constexpr std::size_t
alignUp(std::size_t size, std::size_t alignment) noexcept {
  return (size + alignment - 1) & ~(alignment - 1);
}

/// Check if a pointer is aligned to the given alignment.
[[nodiscard]] PULSE_FORCE_INLINE bool
isAligned(const void *ptr, std::size_t alignment) noexcept {
  return (reinterpret_cast<std::uintptr_t>(ptr) & (alignment - 1)) == 0;
}

/// Check if a value is a power of 2.
[[nodiscard]] PULSE_FORCE_INLINE constexpr bool
isPowerOf2(std::size_t v) noexcept {
  return v > 0 && (v & (v - 1)) == 0;
}

// ── Platform aligned allocation (for backing memory)
// ──────────────────────────

/// Allocate aligned memory from the OS. Use for allocator backing stores only.
[[nodiscard]] inline void *
platformAlignedAlloc(std::size_t size, std::size_t alignment) noexcept {
  assert(isPowerOf2(alignment));
#if defined(_MSC_VER) && !defined(__MINGW32__)
  return _aligned_malloc(size, alignment);
#elif defined(__MINGW32__) || defined(__MINGW64__)
  // MinGW: use _mm_malloc (from xmmintrin.h / malloc.h) or manual alignment.
  // _aligned_malloc may not be available in all MinGW distributions.
  void *ptr = nullptr;
  // Ensure alignment is at least sizeof(void*) for the header trick
  if (alignment < sizeof(void *))
    alignment = sizeof(void *);
  void *raw = std::malloc(size + alignment + sizeof(void *));
  if (!raw)
    return nullptr;
  auto rawAddr = reinterpret_cast<std::uintptr_t>(raw) + sizeof(void *);
  auto aligned = (rawAddr + alignment - 1) & ~(alignment - 1);
  reinterpret_cast<void **>(aligned)[-1] = raw;
  ptr = reinterpret_cast<void *>(aligned);
  return ptr;
#else
  void *ptr = nullptr;
  if (posix_memalign(&ptr, alignment, size) != 0) {
    return nullptr;
  }
  return ptr;
#endif
}

/// Free memory allocated by platformAlignedAlloc.
inline void platformAlignedFree(void *ptr) noexcept {
#if defined(_MSC_VER) && !defined(__MINGW32__)
  _aligned_free(ptr);
#elif defined(__MINGW32__) || defined(__MINGW64__)
  if (ptr) {
    std::free(reinterpret_cast<void **>(ptr)[-1]);
  }
#else
  free(ptr);
#endif
}

// ── Allocation header (for debug tracking)
// ────────────────────────────────────

#ifdef PULSE_DEBUG_ALLOCATORS
/// Debug header prepended to each allocation for leak detection and validation.
struct AllocationHeader {
  std::size_t size;      ///< Requested allocation size.
  std::size_t alignment; ///< Requested alignment.
  uint32_t guardPattern; ///< Magic pattern for corruption detection.
  uint32_t padding;      ///< Pad to 16 bytes.

  static constexpr uint32_t GuardValue = 0xDEADBEEF;
};
static_assert(sizeof(AllocationHeader) == 24 || sizeof(AllocationHeader) == 16,
              "AllocationHeader should be 16 or 24 bytes");
#endif

// ── Allocator statistics
// ──────────────────────────────────────────────────────

/// Statistics tracked by each allocator (always available, very low overhead).
struct AllocatorStats {
  std::size_t totalAllocations = 0;    ///< Total number of allocate() calls.
  std::size_t totalDeallocations = 0;  ///< Total number of deallocate() calls.
  std::size_t currentBytesUsed = 0;    ///< Currently allocated bytes.
  std::size_t peakBytesUsed = 0;       ///< High-water mark of bytes used.
  std::size_t totalBytesAllocated = 0; ///< Cumulative bytes allocated.
  std::size_t totalBytesFreed = 0;     ///< Cumulative bytes freed.
  std::size_t wastedBytes = 0;         ///< Bytes lost to alignment padding.

  PULSE_FORCE_INLINE void recordAllocation(std::size_t size,
                                           std::size_t waste = 0) noexcept {
    totalAllocations++;
    totalBytesAllocated += size;
    currentBytesUsed += size;
    wastedBytes += waste;
    if (currentBytesUsed > peakBytesUsed) {
      peakBytesUsed = currentBytesUsed;
    }
  }

  PULSE_FORCE_INLINE void recordDeallocation(std::size_t size) noexcept {
    totalDeallocations++;
    totalBytesFreed += size;
    currentBytesUsed -= size;
  }

  PULSE_FORCE_INLINE void reset() noexcept {
    totalAllocations = 0;
    totalDeallocations = 0;
    currentBytesUsed = 0;
    peakBytesUsed = 0;
    totalBytesAllocated = 0;
    totalBytesFreed = 0;
    wastedBytes = 0;
  }
};

// ── CRTP Allocator Base
// ───────────────────────────────────────────────────────

/**
 * @class AllocatorBase
 * @brief CRTP base class for all Pulse allocators.
 *
 * Provides common interface (allocate, deallocate, reset, owns) via
 * compile-time polymorphism. No virtual functions, no vtable.
 *
 * @tparam Derived The concrete allocator class.
 *
 * Usage:
 * @code
 *   class PoolAllocator : public AllocatorBase<PoolAllocator> { ... };
 * @endcode
 */
template <typename Derived> class AllocatorBase {
public:
  /// Allocate `size` bytes with the given alignment.
  [[nodiscard]] PULSE_FORCE_INLINE void *
  allocate(std::size_t size,
           std::size_t alignment = DefaultAlignment) noexcept {
    assert(isPowerOf2(alignment));
    assert(size > 0);
    return static_cast<Derived *>(this)->doAllocate(size, alignment);
  }

  /// Deallocate a previously allocated pointer.
  PULSE_FORCE_INLINE void deallocate(void *ptr) noexcept {
    if (PULSE_LIKELY(ptr != nullptr)) {
      static_cast<Derived *>(this)->doDeallocate(ptr);
    }
  }

  /// Deallocate with known size (some allocators can use this for O(1) free).
  PULSE_FORCE_INLINE void deallocate(void *ptr, std::size_t size) noexcept {
    if (PULSE_LIKELY(ptr != nullptr)) {
      static_cast<Derived *>(this)->doDeallocate(ptr, size);
    }
  }

  /// Reset the allocator to its initial state (free all allocations at once).
  PULSE_FORCE_INLINE void reset() noexcept {
    static_cast<Derived *>(this)->doReset();
  }

  /// Check if this allocator owns the given pointer.
  [[nodiscard]] PULSE_FORCE_INLINE bool owns(const void *ptr) const noexcept {
    return static_cast<const Derived *>(this)->doOwns(ptr);
  }

  /// Get allocator statistics.
  [[nodiscard]] PULSE_FORCE_INLINE const AllocatorStats &
  getStats() const noexcept {
    return stats_;
  }

  /// Get total capacity in bytes.
  [[nodiscard]] PULSE_FORCE_INLINE std::size_t capacity() const noexcept {
    return static_cast<const Derived *>(this)->doCapacity();
  }

  /// Get currently used bytes.
  [[nodiscard]] PULSE_FORCE_INLINE std::size_t usedBytes() const noexcept {
    return stats_.currentBytesUsed;
  }

  /// Get available bytes.
  [[nodiscard]] PULSE_FORCE_INLINE std::size_t availableBytes() const noexcept {
    return capacity() - usedBytes();
  }

  // ── Typed allocation helpers ──────────────────────────────────────────

  /// Allocate and construct an object of type T.
  template <typename T, typename... Args>
  [[nodiscard]] PULSE_FORCE_INLINE T *create(Args &&...args) noexcept {
    void *mem = allocate(sizeof(T), alignof(T));
    if (PULSE_UNLIKELY(mem == nullptr))
      return nullptr;
    return new (mem) T(std::forward<Args>(args)...);
  }

  /// Destroy and deallocate an object of type T.
  template <typename T> PULSE_FORCE_INLINE void destroy(T *obj) noexcept {
    if (obj) {
      obj->~T();
      deallocate(obj, sizeof(T));
    }
  }

  /// Allocate an array of N objects of type T (no construction).
  template <typename T>
  [[nodiscard]] PULSE_FORCE_INLINE T *
  allocateArray(std::size_t count) noexcept {
    return static_cast<T *>(allocate(sizeof(T) * count, alignof(T)));
  }

protected:
  AllocatorStats stats_;
};

// ── Scoped allocator guard
// ────────────────────────────────────────────────────

/**
 * @class ScopedAllocatorReset
 * @brief RAII guard that resets an allocator when it goes out of scope.
 *
 * Useful for frame allocators and stack allocators where you want to
 * automatically reclaim all memory at the end of a scope.
 */
template <typename Allocator> class ScopedAllocatorReset {
public:
  explicit ScopedAllocatorReset(Allocator &alloc) noexcept
      : allocator_(alloc) {}
  ~ScopedAllocatorReset() noexcept { allocator_.reset(); }

  ScopedAllocatorReset(const ScopedAllocatorReset &) = delete;
  ScopedAllocatorReset &operator=(const ScopedAllocatorReset &) = delete;

private:
  Allocator &allocator_;
};

} // namespace memory
} // namespace pulse
