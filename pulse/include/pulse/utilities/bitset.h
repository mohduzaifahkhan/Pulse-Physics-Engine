/**
 * @file bitset.h
 * @brief Fixed-size bitset for flags, component masks, and collision layers.
 *
 * Backed by an array of uint64_t words. All operations use hardware-accelerated
 * bit manipulation intrinsics (popcnt, BSF/CTZ, BSR/CLZ) when available.
 *
 * Fully stack-allocated — no heap usage. The capacity N is a compile-time
 * constant (number of bits). The number of 64-bit words is ceil(N / 64).
 *
 * Thread safety: NOT thread-safe. External synchronization required.
 */

#pragma once

#include <pulse/math/math_common.h>
#include <pulse/utilities/assert.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace pulse {
namespace util {

// ── Bit intrinsics ───────────────────────────────────────────────────────────

/// Population count (number of set bits) for a 64-bit word.
[[nodiscard]] PULSE_FORCE_INLINE int popcount64(uint64_t x) noexcept {
#if defined(PULSE_COMPILER_MSVC)
    #if defined(_M_X64) || defined(_M_AMD64)
        return static_cast<int>(__popcnt64(x));
    #else
        // 32-bit MSVC fallback
        int count = 0;
        while (x) { x &= (x - 1); count++; }
        return count;
    #endif
#elif defined(PULSE_COMPILER_GCC) || defined(PULSE_COMPILER_CLANG)
    return __builtin_popcountll(x);
#else
    int count = 0;
    while (x) { x &= (x - 1); count++; }
    return count;
#endif
}

/// Count trailing zeros (index of lowest set bit) for a 64-bit word.
/// Undefined if x == 0.
[[nodiscard]] PULSE_FORCE_INLINE int ctz64(uint64_t x) noexcept {
    PULSE_ASSERT(x != 0);
#if defined(PULSE_COMPILER_MSVC)
    #if defined(_M_X64) || defined(_M_AMD64)
        unsigned long idx;
        _BitScanForward64(&idx, x);
        return static_cast<int>(idx);
    #else
        int n = 0;
        while ((x & 1) == 0) { x >>= 1; n++; }
        return n;
    #endif
#elif defined(PULSE_COMPILER_GCC) || defined(PULSE_COMPILER_CLANG)
    return __builtin_ctzll(x);
#else
    int n = 0;
    while ((x & 1) == 0) { x >>= 1; n++; }
    return n;
#endif
}

// ── FixedBitset ──────────────────────────────────────────────────────────────

/**
 * @class FixedBitset
 * @brief Fixed-capacity bitset backed by uint64_t words.
 *
 * @tparam N Number of bits. Rounded up to the nearest multiple of 64 internally.
 *
 * Use cases: collision layer masks, component flags, island membership,
 * sleep/wake tracking for large body arrays.
 */
template <std::size_t N>
class FixedBitset {
    static_assert(N > 0, "FixedBitset must have at least 1 bit");

public:
    static constexpr std::size_t BitCount = N;
    static constexpr std::size_t WordCount = (N + 63) / 64;

    // ── Constructors ─────────────────────────────────────────────────────

    /// Default: all bits cleared.
    constexpr FixedBitset() noexcept : words_{} {}

    // ── Single-bit operations ────────────────────────────────────────────

    /// Set bit at index i.
    PULSE_FORCE_INLINE constexpr void set(std::size_t i) noexcept {
        PULSE_ASSERT(i < N);
        words_[i / 64] |= (uint64_t(1) << (i % 64));
    }

    /// Clear bit at index i.
    PULSE_FORCE_INLINE constexpr void clear(std::size_t i) noexcept {
        PULSE_ASSERT(i < N);
        words_[i / 64] &= ~(uint64_t(1) << (i % 64));
    }

    /// Toggle bit at index i.
    PULSE_FORCE_INLINE constexpr void toggle(std::size_t i) noexcept {
        PULSE_ASSERT(i < N);
        words_[i / 64] ^= (uint64_t(1) << (i % 64));
    }

    /// Test bit at index i.
    [[nodiscard]] PULSE_FORCE_INLINE constexpr bool test(std::size_t i) const noexcept {
        PULSE_ASSERT(i < N);
        return (words_[i / 64] & (uint64_t(1) << (i % 64))) != 0;
    }

    /// Set bit at index i to the given value.
    PULSE_FORCE_INLINE constexpr void setTo(std::size_t i, bool value) noexcept {
        if (value) set(i); else clear(i);
    }

    // ── Bulk operations ──────────────────────────────────────────────────

    /// Set all bits.
    PULSE_FORCE_INLINE constexpr void setAll() noexcept {
        for (std::size_t w = 0; w < WordCount; ++w) {
            words_[w] = ~uint64_t(0);
        }
        // Mask out unused bits in the last word.
        if (N % 64 != 0) {
            words_[WordCount - 1] &= (uint64_t(1) << (N % 64)) - 1;
        }
    }

    /// Clear all bits.
    PULSE_FORCE_INLINE constexpr void clearAll() noexcept {
        for (std::size_t w = 0; w < WordCount; ++w) {
            words_[w] = 0;
        }
    }

    // ── Queries ──────────────────────────────────────────────────────────

    /// Count the number of set bits (population count).
    [[nodiscard]] PULSE_FORCE_INLINE constexpr int countSet() const noexcept {
        int total = 0;
        for (std::size_t w = 0; w < WordCount; ++w) {
            total += popcount64(words_[w]);
        }
        return total;
    }

    /// Return the index of the first (lowest) set bit, or -1 if none.
    [[nodiscard]] PULSE_FORCE_INLINE constexpr int firstSet() const noexcept {
        for (std::size_t w = 0; w < WordCount; ++w) {
            if (words_[w] != 0) {
                return static_cast<int>(w * 64 + static_cast<std::size_t>(ctz64(words_[w])));
            }
        }
        return -1;
    }

    /// Check if any bit is set.
    [[nodiscard]] PULSE_FORCE_INLINE constexpr bool any() const noexcept {
        for (std::size_t w = 0; w < WordCount; ++w) {
            if (words_[w] != 0) return true;
        }
        return false;
    }

    /// Check if no bits are set.
    [[nodiscard]] PULSE_FORCE_INLINE constexpr bool none() const noexcept {
        return !any();
    }

    /// Iterate over all set bit indices. Calls func(std::size_t bitIndex)
    /// for each set bit, in ascending order.
    template <typename Func>
    PULSE_FORCE_INLINE constexpr void forEachSet(Func&& func) const {
        for (std::size_t w = 0; w < WordCount; ++w) {
            uint64_t word = words_[w];
            while (word != 0) {
                int bit = ctz64(word);
                func(w * 64 + static_cast<std::size_t>(bit));
                word &= word - 1; // Clear lowest set bit
            }
        }
    }

    // ── Bitwise operators ────────────────────────────────────────────────

    [[nodiscard]] constexpr FixedBitset operator&(const FixedBitset& other) const noexcept {
        FixedBitset result;
        for (std::size_t w = 0; w < WordCount; ++w) {
            result.words_[w] = words_[w] & other.words_[w];
        }
        return result;
    }

    [[nodiscard]] constexpr FixedBitset operator|(const FixedBitset& other) const noexcept {
        FixedBitset result;
        for (std::size_t w = 0; w < WordCount; ++w) {
            result.words_[w] = words_[w] | other.words_[w];
        }
        return result;
    }

    [[nodiscard]] constexpr FixedBitset operator^(const FixedBitset& other) const noexcept {
        FixedBitset result;
        for (std::size_t w = 0; w < WordCount; ++w) {
            result.words_[w] = words_[w] ^ other.words_[w];
        }
        return result;
    }

    [[nodiscard]] constexpr FixedBitset operator~() const noexcept {
        FixedBitset result;
        for (std::size_t w = 0; w < WordCount; ++w) {
            result.words_[w] = ~words_[w];
        }
        // Mask out unused bits in the last word.
        if (N % 64 != 0) {
            result.words_[WordCount - 1] &= (uint64_t(1) << (N % 64)) - 1;
        }
        return result;
    }

    constexpr FixedBitset& operator&=(const FixedBitset& other) noexcept {
        for (std::size_t w = 0; w < WordCount; ++w) words_[w] &= other.words_[w];
        return *this;
    }

    constexpr FixedBitset& operator|=(const FixedBitset& other) noexcept {
        for (std::size_t w = 0; w < WordCount; ++w) words_[w] |= other.words_[w];
        return *this;
    }

    constexpr FixedBitset& operator^=(const FixedBitset& other) noexcept {
        for (std::size_t w = 0; w < WordCount; ++w) words_[w] ^= other.words_[w];
        return *this;
    }

    // ── Comparison ───────────────────────────────────────────────────────

    [[nodiscard]] constexpr bool operator==(const FixedBitset& other) const noexcept {
        for (std::size_t w = 0; w < WordCount; ++w) {
            if (words_[w] != other.words_[w]) return false;
        }
        return true;
    }

    [[nodiscard]] constexpr bool operator!=(const FixedBitset& other) const noexcept {
        return !(*this == other);
    }

    // ── Raw access (for advanced usage) ──────────────────────────────────

    [[nodiscard]] constexpr const uint64_t* data() const noexcept { return words_; }
    [[nodiscard]] constexpr uint64_t* data() noexcept { return words_; }

private:
    uint64_t words_[WordCount];
};

} // namespace util
} // namespace pulse
