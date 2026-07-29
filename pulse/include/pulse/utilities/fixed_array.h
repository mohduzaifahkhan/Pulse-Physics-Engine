/**
 * @file fixed_array.h
 * @brief Fixed-capacity array with push/pop semantics — no heap allocation.
 *
 * Like std::array but with a dynamic size counter. Elements are stored in-place.
 * Maximum capacity is a compile-time constant N. Useful for small, bounded
 * collections: contact manifold points (max 4), broadphase candidates, etc.
 *
 * Automatically aligns to 16 bytes when T is a SIMD type (Vec3, Vec4, etc.).
 *
 * Thread safety: NOT thread-safe. External synchronization required.
 */

#pragma once

#include <pulse/math/math_common.h>
#include <pulse/utilities/assert.h>
#include <pulse/utilities/type_traits.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

namespace pulse {
namespace util {

/**
 * @class FixedArray
 * @brief Stack-allocated array with push_back/pop_back and fixed maximum capacity.
 *
 * @tparam T     Element type.
 * @tparam N     Maximum number of elements.
 */
template <typename T, std::size_t N>
class alignas(is_simd_type_v<T> ? alignment_of_v<T> : alignof(T)) FixedArray {
    static_assert(N > 0, "FixedArray must have capacity > 0");

public:
    using value_type      = T;
    using size_type       = std::size_t;
    using reference       = T&;
    using const_reference = const T&;
    using pointer         = T*;
    using const_pointer   = const T*;
    using iterator        = T*;
    using const_iterator  = const T*;

    // ── Constructors / Destructor ────────────────────────────────────────

    /// Default: empty array.
    FixedArray() noexcept : size_(0) {}

    /// Construct with count copies of value.
    explicit FixedArray(std::size_t count, const T& value = T{}) noexcept : size_(0) {
        PULSE_ASSERT(count <= N);
        for (std::size_t i = 0; i < count; ++i) {
            constructAt(i, value);
        }
        size_ = count;
    }

    /// Construct from initializer list.
    FixedArray(std::initializer_list<T> init) noexcept : size_(0) {
        PULSE_ASSERT(init.size() <= N);
        for (const auto& val : init) {
            constructAt(size_, val);
            size_++;
        }
    }

    /// Copy constructor.
    FixedArray(const FixedArray& other) noexcept : size_(0) {
        for (std::size_t i = 0; i < other.size_; ++i) {
            constructAt(i, other[i]);
        }
        size_ = other.size_;
    }

    /// Move constructor.
    FixedArray(FixedArray&& other) noexcept : size_(0) {
        for (std::size_t i = 0; i < other.size_; ++i) {
            constructAt(i, static_cast<T&&>(other[i]));
        }
        size_ = other.size_;
        other.clear();
    }

    /// Destructor.
    ~FixedArray() noexcept {
        destroyAll();
    }

    /// Copy assignment.
    FixedArray& operator=(const FixedArray& other) noexcept {
        if (this != &other) {
            destroyAll();
            size_ = 0;
            for (std::size_t i = 0; i < other.size_; ++i) {
                constructAt(i, other[i]);
            }
            size_ = other.size_;
        }
        return *this;
    }

    /// Move assignment.
    FixedArray& operator=(FixedArray&& other) noexcept {
        if (this != &other) {
            destroyAll();
            size_ = 0;
            for (std::size_t i = 0; i < other.size_; ++i) {
                constructAt(i, static_cast<T&&>(other[i]));
            }
            size_ = other.size_;
            other.clear();
        }
        return *this;
    }

    // ── Element access ───────────────────────────────────────────────────

    /// Unchecked access by index.
    [[nodiscard]] PULSE_FORCE_INLINE reference operator[](std::size_t i) noexcept {
        PULSE_ASSERT(i < size_);
        return *ptrAt(i);
    }

    [[nodiscard]] PULSE_FORCE_INLINE const_reference operator[](std::size_t i) const noexcept {
        PULSE_ASSERT(i < size_);
        return *ptrAt(i);
    }

    /// Bounds-checked access (asserts in debug, no exception).
    [[nodiscard]] PULSE_FORCE_INLINE reference at(std::size_t i) noexcept {
        PULSE_ASSERT_MSG(i < size_, "FixedArray::at: index out of bounds");
        return *ptrAt(i);
    }

    [[nodiscard]] PULSE_FORCE_INLINE const_reference at(std::size_t i) const noexcept {
        PULSE_ASSERT_MSG(i < size_, "FixedArray::at: index out of bounds");
        return *ptrAt(i);
    }

    /// First element.
    [[nodiscard]] PULSE_FORCE_INLINE reference front() noexcept {
        PULSE_ASSERT(size_ > 0);
        return *ptrAt(0);
    }

    [[nodiscard]] PULSE_FORCE_INLINE const_reference front() const noexcept {
        PULSE_ASSERT(size_ > 0);
        return *ptrAt(0);
    }

    /// Last element.
    [[nodiscard]] PULSE_FORCE_INLINE reference back() noexcept {
        PULSE_ASSERT(size_ > 0);
        return *ptrAt(size_ - 1);
    }

    [[nodiscard]] PULSE_FORCE_INLINE const_reference back() const noexcept {
        PULSE_ASSERT(size_ > 0);
        return *ptrAt(size_ - 1);
    }

    /// Raw data pointer.
    [[nodiscard]] PULSE_FORCE_INLINE pointer data() noexcept {
        return ptrAt(0);
    }

    [[nodiscard]] PULSE_FORCE_INLINE const_pointer data() const noexcept {
        return ptrAt(0);
    }

    // ── Capacity ─────────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE std::size_t size() const noexcept { return size_; }
    [[nodiscard]] static constexpr std::size_t capacity() noexcept { return N; }
    [[nodiscard]] PULSE_FORCE_INLINE bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] PULSE_FORCE_INLINE bool full() const noexcept { return size_ == N; }

    // ── Modifiers ────────────────────────────────────────────────────────

    /// Append a copy of value. Asserts if full.
    PULSE_FORCE_INLINE void push_back(const T& value) noexcept {
        PULSE_ASSERT_MSG(!full(), "FixedArray::push_back: array is full");
        constructAt(size_, value);
        size_++;
    }

    /// Append by move. Asserts if full.
    PULSE_FORCE_INLINE void push_back(T&& value) noexcept {
        PULSE_ASSERT_MSG(!full(), "FixedArray::push_back: array is full");
        constructAt(size_, static_cast<T&&>(value));
        size_++;
    }

    /// Construct an element in-place at the back. Asserts if full.
    template <typename... Args>
    PULSE_FORCE_INLINE reference emplace_back(Args&&... args) noexcept {
        PULSE_ASSERT_MSG(!full(), "FixedArray::emplace_back: array is full");
        auto* ptr = new (ptrAt(size_)) T(static_cast<Args&&>(args)...);
        size_++;
        return *ptr;
    }

    /// Remove the last element. Asserts if empty.
    PULSE_FORCE_INLINE void pop_back() noexcept {
        PULSE_ASSERT_MSG(!empty(), "FixedArray::pop_back: array is empty");
        size_--;
        destroyAt(size_);
    }

    /// Remove element at index by swapping with the last element (O(1), unordered).
    PULSE_FORCE_INLINE void swapRemove(std::size_t i) noexcept {
        PULSE_ASSERT(i < size_);
        if (i != size_ - 1) {
            // Move last element into the gap.
            if (std::is_trivially_copyable<T>::value) {
                std::memcpy(ptrAt(i), ptrAt(size_ - 1), sizeof(T));
            } else {
                (*ptrAt(i)) = static_cast<T&&>(*ptrAt(size_ - 1));
            }
        }
        size_--;
        destroyAt(size_);
    }

    /// Clear all elements.
    PULSE_FORCE_INLINE void clear() noexcept {
        destroyAll();
        size_ = 0;
    }

    /// Resize to newSize. If growing, new elements are value-initialized.
    void resize(std::size_t newSize) noexcept {
        PULSE_ASSERT(newSize <= N);
        if (newSize > size_) {
            for (std::size_t i = size_; i < newSize; ++i) {
                constructAt(i, T{});
            }
        } else if (newSize < size_) {
            for (std::size_t i = newSize; i < size_; ++i) {
                destroyAt(i);
            }
        }
        size_ = newSize;
    }

    // ── Iterators ────────────────────────────────────────────────────────

    [[nodiscard]] iterator begin() noexcept { return ptrAt(0); }
    [[nodiscard]] iterator end() noexcept { return ptrAt(size_); }
    [[nodiscard]] const_iterator begin() const noexcept { return ptrAt(0); }
    [[nodiscard]] const_iterator end() const noexcept { return ptrAt(size_); }
    [[nodiscard]] const_iterator cbegin() const noexcept { return ptrAt(0); }
    [[nodiscard]] const_iterator cend() const noexcept { return ptrAt(size_); }

private:
    // ── Storage ──────────────────────────────────────────────────────────

    /// Aligned uninitialized storage for N elements of type T.
    alignas(alignof(T)) unsigned char storage_[sizeof(T) * N];

    /// Current number of live elements.
    std::size_t size_;

    // ── Helpers ──────────────────────────────────────────────────────────

    [[nodiscard]] PULSE_FORCE_INLINE T* ptrAt(std::size_t i) noexcept {
        return reinterpret_cast<T*>(storage_ + sizeof(T) * i);
    }

    [[nodiscard]] PULSE_FORCE_INLINE const T* ptrAt(std::size_t i) const noexcept {
        return reinterpret_cast<const T*>(storage_ + sizeof(T) * i);
    }

    PULSE_FORCE_INLINE void constructAt(std::size_t i, const T& val) noexcept {
        new (ptrAt(i)) T(val);
    }

    PULSE_FORCE_INLINE void constructAt(std::size_t i, T&& val) noexcept {
        new (ptrAt(i)) T(static_cast<T&&>(val));
    }

    PULSE_FORCE_INLINE void destroyAt(std::size_t i) noexcept {
        if (!std::is_trivially_destructible<T>::value) {
            ptrAt(i)->~T();
        }
    }

    void destroyAll() noexcept {
        if (!std::is_trivially_destructible<T>::value) {
            for (std::size_t i = 0; i < size_; ++i) {
                ptrAt(i)->~T();
            }
        }
    }
};

} // namespace util
} // namespace pulse
