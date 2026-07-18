/**
 * @file assert.h
 * @brief Custom assertion macros for the Pulse physics engine.
 *
 * Replaces raw <cassert> with engine-specific macros that support message
 * formatting, source location capture, and compile-time enable/disable.
 *
 * Controlled by PULSE_ENABLE_ASSERTS:
 * - Defined by default in debug builds (_DEBUG or DEBUG)
 * - Can be explicitly defined/undefined to override
 * - When disabled, PULSE_ASSERT becomes a no-op (zero cost)
 * - PULSE_VERIFY always evaluates its expression, but only asserts in debug
 *
 * Assert failure handler is a replaceable function pointer for custom
 * crash reporters, loggers, or debugger integration.
 */

#pragma once

#include <pulse/math/math_common.h>

#include <cstdio>
#include <cstdlib>

// ── Auto-detect debug mode ───────────────────────────────────────────────────

#if !defined(PULSE_ENABLE_ASSERTS) && !defined(PULSE_DISABLE_ASSERTS)
    #if defined(_DEBUG) || defined(DEBUG) || !defined(NDEBUG)
        #define PULSE_ENABLE_ASSERTS 1
    #endif
#endif

namespace pulse {
namespace util {

// ── Source location ──────────────────────────────────────────────────────────

/// Lightweight source location for assert messages (avoids <source_location>
/// which is not yet universally constexpr).
struct SourceLocation {
    const char* file;
    const char* function;
    int line;
};

// ── Assert failure handler ───────────────────────────────────────────────────

/// Signature for custom assert failure handlers.
/// @param loc  Source location where the assert fired.
/// @param expr Stringified expression that failed.
/// @param msg  Optional message (nullptr if none).
using AssertHandler = void(*)(const SourceLocation& loc,
                              const char* expr,
                              const char* msg);

/// Default assert handler — prints to stderr and aborts.
inline void defaultAssertHandler(const SourceLocation& loc,
                                 const char* expr,
                                 const char* msg) noexcept {
    std::fprintf(stderr,
                 "\n=== PULSE ASSERT FAILED ===\n"
                 "  Expression: %s\n"
                 "  File:       %s\n"
                 "  Line:       %d\n"
                 "  Function:   %s\n",
                 expr, loc.file, loc.line, loc.function);
    if (msg) {
        std::fprintf(stderr, "  Message:    %s\n", msg);
    }
    std::fprintf(stderr, "===========================\n\n");
    std::fflush(stderr);

    // Trigger a debugger break if available, then abort
#if defined(PULSE_COMPILER_MSVC)
    __debugbreak();
#elif defined(PULSE_COMPILER_GCC) || defined(PULSE_COMPILER_CLANG)
    __builtin_trap();
#else
    std::abort();
#endif
}

/// Global assert handler — replace to hook into custom crash reporting.
/// Thread safety: set once at startup before any physics work.
inline AssertHandler& getAssertHandler() noexcept {
    static AssertHandler handler = defaultAssertHandler;
    return handler;
}

/// Set a custom assert failure handler.
inline void setAssertHandler(AssertHandler handler) noexcept {
    getAssertHandler() = handler ? handler : defaultAssertHandler;
}

/// Fire the current assert handler.
PULSE_NO_INLINE inline void fireAssert(const SourceLocation& loc,
                                       const char* expr,
                                       const char* msg) noexcept {
    getAssertHandler()(loc, expr, msg);
}

} // namespace util
} // namespace pulse

// ── Assertion macros ─────────────────────────────────────────────────────────

/// Helper to construct a SourceLocation from the current position.
#define PULSE_SOURCE_LOCATION_ \
    ::pulse::util::SourceLocation{__FILE__, PULSE_CURRENT_FUNCTION_, __LINE__}

/// Portable __func__ / __FUNCTION__
#if defined(PULSE_COMPILER_MSVC)
    #define PULSE_CURRENT_FUNCTION_ __FUNCTION__
#else
    #define PULSE_CURRENT_FUNCTION_ __func__
#endif

#ifdef PULSE_ENABLE_ASSERTS

    /**
     * @brief Debug-only runtime assertion.
     *
     * Evaluates @p expr. If false, fires the assert handler and aborts.
     * Compiles to nothing when PULSE_ENABLE_ASSERTS is not defined.
     */
    #define PULSE_ASSERT(expr) \
        do { \
            if (!(expr)) { \
                ::pulse::util::fireAssert( \
                    PULSE_SOURCE_LOCATION_, #expr, nullptr); \
            } \
        } while (0)

    /**
     * @brief Debug-only runtime assertion with a custom message.
     */
    #define PULSE_ASSERT_MSG(expr, msg) \
        do { \
            if (!(expr)) { \
                ::pulse::util::fireAssert( \
                    PULSE_SOURCE_LOCATION_, #expr, msg); \
            } \
        } while (0)

#else // PULSE_ENABLE_ASSERTS not defined

    #define PULSE_ASSERT(expr)          ((void)0)
    #define PULSE_ASSERT_MSG(expr, msg) ((void)0)

#endif // PULSE_ENABLE_ASSERTS

/**
 * @brief Always-evaluated verify — expression runs in all builds,
 *        but only asserts in debug.
 *
 * Use for expressions with side effects that must always execute.
 */
#ifdef PULSE_ENABLE_ASSERTS
    #define PULSE_VERIFY(expr) \
        do { \
            if (!(expr)) { \
                ::pulse::util::fireAssert( \
                    PULSE_SOURCE_LOCATION_, #expr, "PULSE_VERIFY failed"); \
            } \
        } while (0)
#else
    #define PULSE_VERIFY(expr) ((void)(expr))
#endif

/**
 * @brief Static (compile-time) assertion wrapper for consistency.
 */
#define PULSE_STATIC_ASSERT(expr, msg) static_assert(expr, msg)

/**
 * @brief Marks code as unreachable. In debug builds, fires an assert.
 *        In release, tells the compiler this path is impossible.
 */
#ifdef PULSE_ENABLE_ASSERTS
    #define PULSE_UNREACHABLE() \
        do { \
            ::pulse::util::fireAssert( \
                PULSE_SOURCE_LOCATION_, "UNREACHABLE", "Reached unreachable code"); \
        } while (0)
#else
    #if defined(PULSE_COMPILER_MSVC)
        #define PULSE_UNREACHABLE() __assume(false)
    #elif defined(PULSE_COMPILER_GCC) || defined(PULSE_COMPILER_CLANG)
        #define PULSE_UNREACHABLE() __builtin_unreachable()
    #else
        #define PULSE_UNREACHABLE() ((void)0)
    #endif
#endif
