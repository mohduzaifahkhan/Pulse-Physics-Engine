/**
 * @file type_traits.h
 * @brief Engine-specific type traits for compile-time introspection.
 *
 * Provides traits to detect SIMD types, check alignment, and wrap types
 * with cache-line padding. Used by containers (SoAArray, FixedArray) and
 * allocators to make optimal layout decisions at compile time.
 */

#pragma once

#include <pulse/math/math_common.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

// Forward declarations of Pulse math types for trait specialization.
namespace pulse {
    struct Vec2;
    struct Vec3;
    struct Vec4;
    struct Quat;
    struct Mat3;
    struct Mat4;
    struct Transform;
    struct AABB;
    struct Plane;
    struct Ray;
}

namespace pulse {
namespace util {

// ── is_simd_type ─────────────────────────────────────────────────────────────

/// Primary template: most types are NOT SIMD types.
template <typename T>
struct is_simd_type : std::false_type {};

/// Specializations for Pulse SIMD-backed math types.
template <> struct is_simd_type<Vec3>      : std::true_type {};
template <> struct is_simd_type<Vec4>      : std::true_type {};
template <> struct is_simd_type<Quat>      : std::true_type {};
template <> struct is_simd_type<Mat3>      : std::true_type {};
template <> struct is_simd_type<Mat4>      : std::true_type {};
template <> struct is_simd_type<AABB>      : std::true_type {};
template <> struct is_simd_type<Plane>     : std::true_type {};
template <> struct is_simd_type<Ray>       : std::true_type {};
template <> struct is_simd_type<Transform> : std::true_type {};

/// Also match const/volatile/reference-qualified types.
template <typename T> struct is_simd_type<const T>    : is_simd_type<T> {};
template <typename T> struct is_simd_type<volatile T> : is_simd_type<T> {};
template <typename T> struct is_simd_type<const volatile T> : is_simd_type<T> {};

/// Helper variable template.
template <typename T>
inline constexpr bool is_simd_type_v = is_simd_type<T>::value;

// ── is_pod_type ──────────────────────────────────────────────────────────────

/// Stricter than std::is_trivially_copyable — requires trivially copyable,
/// trivially default constructible, and standard layout. This guarantees
/// the type can be safely memcpy'd, memset'd, and has no hidden padding
/// or vtable pointers.
template <typename T>
struct is_pod_type : std::bool_constant<
    std::is_trivially_copyable_v<T> &&
    std::is_trivially_default_constructible_v<T> &&
    std::is_standard_layout_v<T>
> {};

/// Helper variable template.
template <typename T>
inline constexpr bool is_pod_type_v = is_pod_type<T>::value;

// ── is_aligned ───────────────────────────────────────────────────────────────

/// Compile-time check: does type T satisfy alignment N?
template <typename T, std::size_t N>
struct is_aligned : std::bool_constant<(alignof(T) >= N)> {};

/// Helper variable template.
template <typename T, std::size_t N>
inline constexpr bool is_aligned_v = is_aligned<T, N>::value;

// ── alignment_of ─────────────────────────────────────────────────────────────

/// Returns the required alignment for type T. For SIMD types, returns at
/// least 16. For all types, returns at least alignof(T).
template <typename T>
struct alignment_of : std::integral_constant<std::size_t,
    is_simd_type_v<T> ? (alignof(T) > 16 ? alignof(T) : 16) : alignof(T)
> {};

/// Helper variable template.
template <typename T>
inline constexpr std::size_t alignment_of_v = alignment_of<T>::value;

// ── is_power_of_2 ────────────────────────────────────────────────────────────

/// Compile-time check: is N a power of 2?
/// Handles N == 0 (returns false).
template <std::size_t N>
struct is_power_of_2 : std::bool_constant<(N > 0) && ((N & (N - 1)) == 0)> {};

/// Helper variable template.
template <std::size_t N>
inline constexpr bool is_power_of_2_v = is_power_of_2<N>::value;

/// constexpr function form (for runtime values).
[[nodiscard]] PULSE_FORCE_INLINE constexpr bool isPowerOf2(std::size_t n) noexcept {
    return (n > 0) && ((n & (n - 1)) == 0);
}

// ── cache_line_padded ────────────────────────────────────────────────────────

/// Wraps type T with padding to fill an entire cache line (64 bytes).
/// Prevents false sharing when T is accessed from multiple threads.
///
/// Example:
///   CacheLinePadded<std::atomic<int>> counter;
///   // sizeof(counter) == 64, no false sharing with adjacent data
template <typename T>
struct PULSE_CACHE_ALIGN CacheLinePadded {
    T value;

    // Pad to fill the rest of the cache line.
    // If T is already >= 64 bytes, no padding is added (padding size = 0
    // would be valid; we use a conditional to avoid zero-sized arrays).
    static constexpr std::size_t PadSize =
        (sizeof(T) >= PULSE_CACHE_LINE) ? 0 : (PULSE_CACHE_LINE - sizeof(T));

    // We only add the pad array if PadSize > 0, using an empty struct otherwise.
    struct PadStorage {
        char pad[PadSize > 0 ? PadSize : 1];
    };

    // Conditional padding member — optimizer will remove the unused byte
    // when PadSize == 0 due to [[no_unique_address]].
    [[no_unique_address]] std::conditional_t<(PadSize > 0), PadStorage, std::monostate> pad_{};

    /// Default construct.
    CacheLinePadded() = default;

    /// Forwarding constructor.
    template <typename... Args>
    explicit CacheLinePadded(Args&&... args)
        : value(static_cast<Args&&>(args)...) {}

    /// Implicit conversion to T&.
    operator T&() noexcept { return value; }
    operator const T&() const noexcept { return value; }

    /// Arrow operator for convenience.
    T* operator->() noexcept { return &value; }
    const T* operator->() const noexcept { return &value; }
};

// Verify cache line padding works.
static_assert(sizeof(CacheLinePadded<int>) >= PULSE_CACHE_LINE,
              "CacheLinePadded must be at least one cache line");

} // namespace util
} // namespace pulse
