/**
 * @file handle.h
 * @brief Generational handle — a safe, lightweight replacement for raw pointers.
 *
 * Packs a 32-bit index and 32-bit generation counter into a single uint64_t.
 * The generation counter is incremented each time a slot is recycled, so any
 * outstanding handles to the old occupant become invalid (stale-pointer
 * detection without garbage collection).
 *
 * The Tag template parameter enables type-safe handles — a Handle<BodyTag>
 * cannot be accidentally passed where a Handle<ShapeTag> is expected.
 *
 * Properties:
 * - 8 bytes (same as a pointer on 64-bit platforms)
 * - Trivially copyable, trivially destructible
 * - Comparable, hashable (suitable as map keys)
 * - O(1) validation via HandlePool generation check
 */

#pragma once

#include <pulse/math/math_common.h>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace pulse {
namespace util {

// ── Tag types for type-safe handles ──────────────────────────────────────────

/// Tags create distinct Handle types at compile time. Define your own:
///   struct BodyTag {};
///   using BodyHandle = Handle<BodyTag>;
struct DefaultTag {};

// ── Handle ───────────────────────────────────────────────────────────────────

/**
 * @struct Handle
 * @brief Generational handle: 32-bit index + 32-bit generation in a uint64_t.
 *
 * @tparam Tag Type tag for compile-time safety. Different tags = incompatible handle types.
 */
template <typename Tag = DefaultTag>
struct Handle {
    uint64_t value;

    // ── Bit layout ───────────────────────────────────────────────────────

    static constexpr uint32_t IndexBits      = 32;
    static constexpr uint32_t GenerationBits = 32;
    static constexpr uint64_t IndexMask      = (uint64_t(1) << IndexBits) - 1;
    static constexpr uint64_t GenerationMask = (uint64_t(1) << GenerationBits) - 1;
    static constexpr uint32_t MaxIndex       = static_cast<uint32_t>(IndexMask);
    static constexpr uint32_t MaxGeneration  = static_cast<uint32_t>(GenerationMask);

    // ── Constructors ─────────────────────────────────────────────────────

    /// Default: null handle.
    constexpr Handle() noexcept : value(~uint64_t(0)) {}

    /// Construct from index and generation.
    constexpr Handle(uint32_t index, uint32_t generation) noexcept
        : value((uint64_t(generation) << IndexBits) | uint64_t(index)) {}

    /// Construct from raw packed value.
    static constexpr Handle fromRaw(uint64_t raw) noexcept {
        Handle h;
        h.value = raw;
        return h;
    }

    // ── Accessors ────────────────────────────────────────────────────────

    /// Extract the 32-bit index.
    [[nodiscard]] constexpr uint32_t index() const noexcept {
        return static_cast<uint32_t>(value & IndexMask);
    }

    /// Extract the 32-bit generation counter.
    [[nodiscard]] constexpr uint32_t generation() const noexcept {
        return static_cast<uint32_t>((value >> IndexBits) & GenerationMask);
    }

    /// Check if this handle is the null sentinel.
    [[nodiscard]] constexpr bool isNull() const noexcept {
        return value == ~uint64_t(0);
    }

    /// Check if this handle is not null (does NOT validate against a pool).
    [[nodiscard]] constexpr bool isValid() const noexcept {
        return !isNull();
    }

    // ── Sentinel ─────────────────────────────────────────────────────────

    /// Return the null sentinel handle.
    [[nodiscard]] static constexpr Handle null() noexcept {
        return Handle{};
    }

    // ── Comparison ───────────────────────────────────────────────────────

    [[nodiscard]] constexpr bool operator==(const Handle& other) const noexcept {
        return value == other.value;
    }

    [[nodiscard]] constexpr bool operator!=(const Handle& other) const noexcept {
        return value != other.value;
    }

    [[nodiscard]] constexpr bool operator<(const Handle& other) const noexcept {
        return value < other.value;
    }
};

} // namespace util
} // namespace pulse

// ── std::hash specialization ─────────────────────────────────────────────────

namespace std {
template <typename Tag>
struct hash<pulse::util::Handle<Tag>> {
    std::size_t operator()(const pulse::util::Handle<Tag>& h) const noexcept {
        // FNV-1a style mixing for good distribution.
        uint64_t v = h.value;
        v ^= v >> 33;
        v *= 0xff51afd7ed558ccdULL;
        v ^= v >> 33;
        v *= 0xc4ceb9fe1a85ec53ULL;
        v ^= v >> 33;
        return static_cast<std::size_t>(v);
    }
};
} // namespace std
