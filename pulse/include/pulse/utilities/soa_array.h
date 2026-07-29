/**
 * @file soa_array.h
 * @brief Structure-of-Arrays container for cache-friendly batch processing.
 *
 * Stores multiple component types as separate flat arrays. Each component
 * array is independently cache-line aligned for maximum SIMD throughput.
 * When iterating over a single component (e.g., all positions), only that
 * array touches the cache — no wasted bandwidth on unneeded fields.
 *
 * Example:
 *   // Position (Vec3) + Velocity (Vec3) + Mass (float) SoA layout
 *   SoAArray<Vec3, Vec3, float> bodies(1024);
 *   size_t idx = bodies.add(Vec3(0,10,0), Vec3(0,0,0), 1.0f);
 *
 *   // Batch iterate positions only (cache-optimal):
 *   Vec3* positions = bodies.getArray<0>();
 *   for (size_t i = 0; i < bodies.size(); ++i) {
 *       positions[i] += gravity * dt;
 *   }
 *
 * Memory layout (conceptual):
 *   [Pos0 Pos1 Pos2 ... PosN] [Vel0 Vel1 Vel2 ... VelN] [M0 M1 M2 ... MN]
 *   ^--- cache-line aligned    ^--- cache-line aligned    ^--- aligned
 *
 * Thread safety: NOT thread-safe. External synchronization required.
 */

#pragma once

#include <pulse/math/math_common.h>
#include <pulse/utilities/assert.h>
#include <pulse/utilities/type_traits.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pulse {
namespace util {

// ── Helpers ──────────────────────────────────────────────────────────────────

namespace detail {

/// Allocate cache-line-aligned memory. Returns nullptr on failure.
inline void* alignedAlloc(std::size_t size, std::size_t alignment) noexcept {
    if (size == 0) return nullptr;
#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#elif defined(_WIN32)
    if (alignment < sizeof(void*)) alignment = sizeof(void*);
    void* raw = std::malloc(size + alignment + sizeof(void*));
    if (!raw) return nullptr;
    std::uintptr_t rawAddr = reinterpret_cast<std::uintptr_t>(raw) + sizeof(void*);
    std::uintptr_t aligned = (rawAddr + alignment - 1) & ~(alignment - 1);
    reinterpret_cast<void**>(aligned)[-1] = raw;
    return reinterpret_cast<void*>(aligned);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) return nullptr;
    return ptr;
#endif
}

/// Free cache-line-aligned memory.
inline void alignedFree(void* ptr) noexcept {
    if (!ptr) return;
#if defined(_MSC_VER)
    _aligned_free(ptr);
#elif defined(_WIN32)
    std::free(reinterpret_cast<void**>(ptr)[-1]);
#else
    std::free(ptr);
#endif
}

/// Compute aligned allocation size for N elements of type T with cache-line alignment.
template <typename T>
constexpr std::size_t alignedArraySize(std::size_t count) noexcept {
    constexpr std::size_t elemAlign = alignment_of_v<T> > PULSE_CACHE_LINE
                                     ? alignment_of_v<T> : PULSE_CACHE_LINE;
    std::size_t rawSize = count * sizeof(T);
    // Round up to alignment boundary.
    return (rawSize + elemAlign - 1) & ~(elemAlign - 1);
}

} // namespace detail

// ── SoAArray ─────────────────────────────────────────────────────────────────

/**
 * @class SoAArray
 * @brief Structure-of-Arrays container with separate flat arrays per component.
 *
 * @tparam Components The types of each component stored in the SoA layout.
 */
template <typename... Components>
class SoAArray {
    static_assert(sizeof...(Components) > 0, "SoAArray requires at least one component type");

public:
    static constexpr std::size_t NumComponents = sizeof...(Components);

    // ── Construction / Destruction ───────────────────────────────────────

    /// Construct with initial capacity. Allocates component arrays.
    explicit SoAArray(std::size_t initialCapacity = 64) noexcept
        : size_(0), capacity_(initialCapacity) {
        allocateArrays(capacity_);
    }

    ~SoAArray() noexcept {
        destroyAll();
        freeArrays();
    }

    // Non-copyable, movable.
    SoAArray(const SoAArray&) = delete;
    SoAArray& operator=(const SoAArray&) = delete;

    SoAArray(SoAArray&& other) noexcept
        : size_(other.size_), capacity_(other.capacity_) {
        moveArrayPtrs(other);
        other.size_ = 0;
        other.capacity_ = 0;
    }

    SoAArray& operator=(SoAArray&& other) noexcept {
        if (this != &other) {
            destroyAll();
            freeArrays();
            size_ = other.size_;
            capacity_ = other.capacity_;
            moveArrayPtrs(other);
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    // ── Add / Remove ─────────────────────────────────────────────────────

    /// Append a new element with the given component values. Returns the index.
    std::size_t add(Components... args) noexcept {
        if (size_ >= capacity_) {
            grow();
        }
        std::size_t idx = size_;
        constructElement(idx, static_cast<Components&&>(args)...);
        size_++;
        return idx;
    }

    /// Remove element at index using swap-and-pop (O(1), unordered).
    /// Returns the index that was swapped in (== index if it was the last element).
    std::size_t remove(std::size_t index) noexcept {
        PULSE_ASSERT(index < size_);
        std::size_t lastIdx = size_ - 1;
        if (index != lastIdx) {
            swapElements(index, lastIdx);
        }
        destroyElement(lastIdx);
        size_--;
        return (index != lastIdx) ? index : lastIdx;
    }

    // ── Component Access ─────────────────────────────────────────────────

    /// Get a reference to component I at the given index.
    template <std::size_t I>
    [[nodiscard]] PULSE_FORCE_INLINE auto& get(std::size_t index) noexcept {
        static_assert(I < NumComponents, "Component index out of range");
        PULSE_ASSERT(index < size_);
        using CompType = std::tuple_element_t<I, std::tuple<Components...>>;
        return static_cast<CompType*>(arrays_[I])[index];
    }

    template <std::size_t I>
    [[nodiscard]] PULSE_FORCE_INLINE const auto& get(std::size_t index) const noexcept {
        static_assert(I < NumComponents, "Component index out of range");
        PULSE_ASSERT(index < size_);
        using CompType = std::tuple_element_t<I, std::tuple<Components...>>;
        return static_cast<const CompType*>(arrays_[I])[index];
    }

    /// Get a raw pointer to component array I for batch processing.
    template <std::size_t I>
    [[nodiscard]] PULSE_FORCE_INLINE auto* getArray() noexcept {
        static_assert(I < NumComponents, "Component index out of range");
        using CompType = std::tuple_element_t<I, std::tuple<Components...>>;
        return static_cast<CompType*>(arrays_[I]);
    }

    template <std::size_t I>
    [[nodiscard]] PULSE_FORCE_INLINE const auto* getArray() const noexcept {
        static_assert(I < NumComponents, "Component index out of range");
        using CompType = std::tuple_element_t<I, std::tuple<Components...>>;
        return static_cast<const CompType*>(arrays_[I]);
    }

    // ── Queries ──────────────────────────────────────────────────────────

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /// Clear all elements (destroys them but retains capacity).
    void clear() noexcept {
        destroyAll();
        size_ = 0;
    }

    // ── Iteration ────────────────────────────────────────────────────────

    /// Iterate all elements, calling func(index, comp0&, comp1&, ...) for each.
    template <typename Func>
    void forEach(Func&& func) {
        forEachImpl(static_cast<Func&&>(func), std::index_sequence_for<Components...>{});
    }

    template <typename Func>
    void forEach(Func&& func) const {
        forEachImpl(static_cast<Func&&>(func), std::index_sequence_for<Components...>{});
    }

private:
    void* arrays_[NumComponents] = {};
    std::size_t size_     = 0;
    std::size_t capacity_ = 0;

    // ── Array management ─────────────────────────────────────────────────

    void allocateArrays(std::size_t cap) noexcept {
        if (cap == 0) {
            for (std::size_t i = 0; i < NumComponents; ++i) arrays_[i] = nullptr;
            return;
        }
        allocateArraysImpl(cap, std::index_sequence_for<Components...>{});
    }

    template <std::size_t... Is>
    void allocateArraysImpl(std::size_t cap, std::index_sequence<Is...>) noexcept {
        ((arrays_[Is] = detail::alignedAlloc(
            cap * sizeof(std::tuple_element_t<Is, std::tuple<Components...>>),
            PULSE_CACHE_LINE)), ...);
    }

    void freeArrays() noexcept {
        for (std::size_t i = 0; i < NumComponents; ++i) {
            detail::alignedFree(arrays_[i]);
            arrays_[i] = nullptr;
        }
    }

    void moveArrayPtrs(SoAArray& other) noexcept {
        for (std::size_t i = 0; i < NumComponents; ++i) {
            arrays_[i] = other.arrays_[i];
            other.arrays_[i] = nullptr;
        }
    }

    // ── Element construction / destruction ────────────────────────────────

    template <std::size_t... Is>
    void constructElementImpl(std::size_t idx, std::index_sequence<Is...>,
                              Components&&... args) noexcept {
        using Tuple = std::tuple<Components...>;
        ((new (static_cast<std::tuple_element_t<Is, Tuple>*>(arrays_[Is]) + idx)
            std::tuple_element_t<Is, Tuple>(
                static_cast<std::tuple_element_t<Is, Tuple>&&>(args))), ...);
        // Suppress unused warning for the fold.
        (void)idx;
    }

    void constructElement(std::size_t idx, Components&&... args) noexcept {
        constructElementImpl(idx, std::index_sequence_for<Components...>{},
                             static_cast<Components&&>(args)...);
    }

    template <std::size_t... Is>
    void destroyElementImpl(std::size_t idx, std::index_sequence<Is...>) noexcept {
        using Tuple = std::tuple<Components...>;
        ((callDestructor<std::tuple_element_t<Is, Tuple>>(
            static_cast<std::tuple_element_t<Is, Tuple>*>(arrays_[Is]) + idx)), ...);
    }

    void destroyElement(std::size_t idx) noexcept {
        destroyElementImpl(idx, std::index_sequence_for<Components...>{});
    }

    template <typename T>
    void callDestructor(T* ptr) noexcept {
        if (!std::is_trivially_destructible<T>::value) {
            ptr->~T();
        }
    }

    void destroyAll() noexcept {
        for (std::size_t i = 0; i < size_; ++i) {
            destroyElement(i);
        }
    }

    // ── Swap elements ────────────────────────────────────────────────────

    template <std::size_t... Is>
    void swapElementsImpl(std::size_t a, std::size_t b,
                          std::index_sequence<Is...>) noexcept {
        using Tuple = std::tuple<Components...>;
        ((swapComponent<std::tuple_element_t<Is, Tuple>>(
            static_cast<std::tuple_element_t<Is, Tuple>*>(arrays_[Is]), a, b)), ...);
    }

    void swapElements(std::size_t a, std::size_t b) noexcept {
        swapElementsImpl(a, b, std::index_sequence_for<Components...>{});
    }

    template <typename T>
    static void swapComponent(T* arr, std::size_t a, std::size_t b) noexcept {
        if (std::is_trivially_copyable<T>::value) {
            unsigned char tmp[sizeof(T)];
            std::memcpy(tmp, arr + a, sizeof(T));
            std::memcpy(arr + a, arr + b, sizeof(T));
            std::memcpy(arr + b, tmp, sizeof(T));
        } else {
            T tmp = static_cast<T&&>(arr[a]);
            arr[a] = static_cast<T&&>(arr[b]);
            arr[b] = static_cast<T&&>(tmp);
        }
    }

    // ── Growth ───────────────────────────────────────────────────────────

    void grow() noexcept {
        std::size_t newCap = capacity_ == 0 ? 64 : capacity_ * 2;
        growImpl(newCap, std::index_sequence_for<Components...>{});
        capacity_ = newCap;
    }

    template <std::size_t... Is>
    void growImpl(std::size_t newCap, std::index_sequence<Is...>) noexcept {
        using Tuple = std::tuple<Components...>;
        ((growArray<Is, std::tuple_element_t<Is, Tuple>>(newCap)), ...);
    }

    template <std::size_t I, typename T>
    void growArray(std::size_t newCap) noexcept {
        void* newArr = detail::alignedAlloc(newCap * sizeof(T), PULSE_CACHE_LINE);
        PULSE_ASSERT_MSG(newArr != nullptr, "SoAArray: allocation failed during grow");

        // Copy existing elements.
        if (size_ > 0 && arrays_[I] != nullptr) {
            if (std::is_trivially_copyable<T>::value) {
                std::memcpy(newArr, arrays_[I], size_ * sizeof(T));
            } else {
                T* dst = static_cast<T*>(newArr);
                T* src = static_cast<T*>(arrays_[I]);
                for (std::size_t i = 0; i < size_; ++i) {
                    new (dst + i) T(static_cast<T&&>(src[i]));
                    src[i].~T();
                }
            }
        }

        detail::alignedFree(arrays_[I]);
        arrays_[I] = newArr;
    }

    // ── forEach implementation ───────────────────────────────────────────

    template <typename Func, std::size_t... Is>
    void forEachImpl(Func&& func, std::index_sequence<Is...>) {
        using Tuple = std::tuple<Components...>;
        for (std::size_t i = 0; i < size_; ++i) {
            func(i,
                 static_cast<std::tuple_element_t<Is, Tuple>*>(arrays_[Is])[i]...);
        }
    }

    template <typename Func, std::size_t... Is>
    void forEachImpl(Func&& func, std::index_sequence<Is...>) const {
        using Tuple = std::tuple<Components...>;
        for (std::size_t i = 0; i < size_; ++i) {
            func(i,
                 static_cast<const std::tuple_element_t<Is, Tuple>*>(arrays_[Is])[i]...);
        }
    }
};

} // namespace util
} // namespace pulse
