/**
 * @file math_common.h
 * @brief Core math definitions: SIMD detection, alignment macros, constants, and utility intrinsics.
 *
 * This header is included by every other math header. It detects the available
 * SIMD instruction sets at compile time, defines alignment and force-inline
 * macros, and provides the fundamental math constants used throughout the engine.
 */

#pragma once

// ── Platform & SIMD detection ─────────────────────────────────────────────────

// Detect compiler
#if defined(_MSC_VER)
    #define PULSE_COMPILER_MSVC 1
#elif defined(__clang__)
    #define PULSE_COMPILER_CLANG 1
#elif defined(__GNUC__)
    #define PULSE_COMPILER_GCC 1
#endif

// Detect SIMD support
#if defined(__AVX2__)
    #define PULSE_SIMD_AVX2 1
#endif

#if defined(__SSE4_2__) || defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
    #define PULSE_SIMD_SSE42 1
#endif

// On MSVC, SSE4.2 is always available when AVX2 is enabled via /arch:AVX2
#if defined(_MSC_VER) && !defined(PULSE_SIMD_SSE42)
    // MSVC with /arch:AVX2 defines __AVX2__ but not __SSE4_2__
    // SSE2 is baseline on x64 MSVC. We assume SSE4.2 if compiling for x64.
    #if defined(_M_X64) || defined(_M_AMD64)
        #define PULSE_SIMD_SSE42 1
    #endif
#endif

// SIMD headers
#ifdef PULSE_SIMD_SSE42
    #include <immintrin.h>  // Covers SSE*, AVX, AVX2
    #include <smmintrin.h>  // SSE4.1 (needed for _mm_dp_ps, _mm_blend_ps)
    #include <nmmintrin.h>  // SSE4.2
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

// ── Force inline ──────────────────────────────────────────────────────────────

#if defined(PULSE_COMPILER_MSVC)
    #define PULSE_FORCE_INLINE __forceinline
    #define PULSE_NO_INLINE __declspec(noinline)
#elif defined(PULSE_COMPILER_GCC) || defined(PULSE_COMPILER_CLANG)
    #define PULSE_FORCE_INLINE __attribute__((always_inline)) inline
    #define PULSE_NO_INLINE __attribute__((noinline))
#else
    #define PULSE_FORCE_INLINE inline
    #define PULSE_NO_INLINE
#endif

// ── Alignment ─────────────────────────────────────────────────────────────────

#define PULSE_ALIGN(n) alignas(n)
#define PULSE_CACHE_LINE 64
#define PULSE_SIMD_ALIGN alignas(16)
#define PULSE_AVX_ALIGN alignas(32)
#define PULSE_CACHE_ALIGN alignas(PULSE_CACHE_LINE)

// ── Prefetch ──────────────────────────────────────────────────────────────────

#if defined(PULSE_COMPILER_MSVC)
    #include <intrin.h>
    #define PULSE_PREFETCH(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
    #define PULSE_PREFETCH_NTA(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_NTA)
#elif defined(PULSE_COMPILER_GCC) || defined(PULSE_COMPILER_CLANG)
    #define PULSE_PREFETCH(addr) __builtin_prefetch(addr, 0, 3)
    #define PULSE_PREFETCH_NTA(addr) __builtin_prefetch(addr, 0, 0)
#else
    #define PULSE_PREFETCH(addr) ((void)0)
    #define PULSE_PREFETCH_NTA(addr) ((void)0)
#endif

// ── Branch prediction ─────────────────────────────────────────────────────────

#if defined(PULSE_COMPILER_GCC) || defined(PULSE_COMPILER_CLANG)
    #define PULSE_LIKELY(x)   __builtin_expect(!!(x), 1)
    #define PULSE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define PULSE_LIKELY(x)   (x)
    #define PULSE_UNLIKELY(x) (x)
#endif

// ── Restrict ──────────────────────────────────────────────────────────────────

#if defined(PULSE_COMPILER_MSVC)
    #define PULSE_RESTRICT __restrict
#elif defined(PULSE_COMPILER_GCC) || defined(PULSE_COMPILER_CLANG)
    #define PULSE_RESTRICT __restrict__
#else
    #define PULSE_RESTRICT
#endif

namespace pulse {

// ── Math constants ────────────────────────────────────────────────────────────

namespace math {

    constexpr float Pi        = 3.14159265358979323846f;
    constexpr float TwoPi     = 6.28318530717958647692f;
    constexpr float HalfPi    = 1.57079632679489661923f;
    constexpr float InvPi     = 0.31830988618379067154f;
    constexpr float Epsilon   = 1.0e-6f;
    constexpr float BigEpsilon = 1.0e-4f;
    constexpr float Infinity  = std::numeric_limits<float>::infinity();
    constexpr float NegInfinity = -std::numeric_limits<float>::infinity();
    constexpr float MaxFloat  = std::numeric_limits<float>::max();
    constexpr float MinFloat  = std::numeric_limits<float>::lowest();
    constexpr float DegToRad  = Pi / 180.0f;
    constexpr float RadToDeg  = 180.0f / Pi;

    /// Approximate equality for floating-point comparisons.
    [[nodiscard]] PULSE_FORCE_INLINE constexpr bool approxEqual(float a, float b, float eps = Epsilon) noexcept {
        const float diff = a - b;
        return (diff > -eps) && (diff < eps);
    }

    /// Clamp a value between [lo, hi].
    [[nodiscard]] PULSE_FORCE_INLINE constexpr float clamp(float v, float lo, float hi) noexcept {
        return (v < lo) ? lo : ((v > hi) ? hi : v);
    }

    /// Linear interpolation.
    [[nodiscard]] PULSE_FORCE_INLINE constexpr float lerp(float a, float b, float t) noexcept {
        return a + t * (b - a);
    }

    /// Safe reciprocal (returns 0 if value is near-zero).
    [[nodiscard]] PULSE_FORCE_INLINE constexpr float safeReciprocal(float v) noexcept {
        return (v > Epsilon || v < -Epsilon) ? (1.0f / v) : 0.0f;
    }

    /// Fast inverse square root (SSE rsqrt with Newton-Raphson refinement).
    [[nodiscard]] PULSE_FORCE_INLINE float fastInvSqrt(float x) noexcept {
#ifdef PULSE_SIMD_SSE42
        __m128 val = _mm_set_ss(x);
        __m128 est = _mm_rsqrt_ss(val);
        // One Newton-Raphson iteration: est = est * (1.5 - 0.5 * x * est * est)
        __m128 half = _mm_set_ss(0.5f);
        __m128 three_half = _mm_set_ss(1.5f);
        __m128 hx = _mm_mul_ss(half, val);
        __m128 est2 = _mm_mul_ss(est, est);
        __m128 hxe2 = _mm_mul_ss(hx, est2);
        __m128 sub = _mm_sub_ss(three_half, hxe2);
        est = _mm_mul_ss(est, sub);
        return _mm_cvtss_f32(est);
#else
        return 1.0f / std::sqrt(x);
#endif
    }

    /// Fast square root via SSE.
    [[nodiscard]] PULSE_FORCE_INLINE float fastSqrt(float x) noexcept {
#ifdef PULSE_SIMD_SSE42
        return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)));
#else
        return std::sqrt(x);
#endif
    }

    /// Minimum of two floats (branchless via SSE).
    [[nodiscard]] PULSE_FORCE_INLINE float fastMin(float a, float b) noexcept {
#ifdef PULSE_SIMD_SSE42
        return _mm_cvtss_f32(_mm_min_ss(_mm_set_ss(a), _mm_set_ss(b)));
#else
        return (a < b) ? a : b;
#endif
    }

    /// Maximum of two floats (branchless via SSE).
    [[nodiscard]] PULSE_FORCE_INLINE float fastMax(float a, float b) noexcept {
#ifdef PULSE_SIMD_SSE42
        return _mm_cvtss_f32(_mm_max_ss(_mm_set_ss(a), _mm_set_ss(b)));
#else
        return (a > b) ? a : b;
#endif
    }

    /// Absolute value (branchless via bit mask).
    [[nodiscard]] PULSE_FORCE_INLINE float fastAbs(float x) noexcept {
        uint32_t i;
        std::memcpy(&i, &x, 4);
        i &= 0x7FFFFFFF;
        float result;
        std::memcpy(&result, &i, 4);
        return result;
    }

    /// Sign of a float: -1, 0, or 1.
    [[nodiscard]] PULSE_FORCE_INLINE constexpr float sign(float x) noexcept {
        return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
    }

    /// Convert degrees to radians.
    [[nodiscard]] PULSE_FORCE_INLINE constexpr float toRadians(float degrees) noexcept {
        return degrees * DegToRad;
    }

    /// Convert radians to degrees.
    [[nodiscard]] PULSE_FORCE_INLINE constexpr float toDegrees(float radians) noexcept {
        return radians * RadToDeg;
    }

} // namespace math

// ── SIMD shuffle/swizzle helpers ──────────────────────────────────────────────

#ifdef PULSE_SIMD_SSE42
namespace simd {

    /// Shuffle mask helper for _mm_shuffle_ps.
    template <int X, int Y, int Z, int W>
    constexpr int shuffleMask() { return ((W) << 6) | ((Z) << 4) | ((Y) << 2) | (X); }

    /// Splat a single component across all lanes.
    template <int Index>
    [[nodiscard]] PULSE_FORCE_INLINE __m128 splat(__m128 v) noexcept {
        static_assert(Index >= 0 && Index <= 3, "Splat index must be 0-3");
        return _mm_shuffle_ps(v, v, _MM_SHUFFLE(Index, Index, Index, Index));
    }

    /// Horizontal sum of all 4 components.
    [[nodiscard]] PULSE_FORCE_INLINE float hsum(__m128 v) noexcept {
        __m128 shuf = _mm_movehdup_ps(v);       // {y, y, w, w}
        __m128 sums = _mm_add_ps(v, shuf);      // {x+y, _, z+w, _}
        shuf = _mm_movehl_ps(shuf, sums);       // {z+w, _, _, _}
        sums = _mm_add_ss(sums, shuf);          // {x+y+z+w, _, _, _}
        return _mm_cvtss_f32(sums);
    }

    /// Horizontal sum of first 3 components (ignore w).
    [[nodiscard]] PULSE_FORCE_INLINE float hsum3(__m128 v) noexcept {
        // Use dot product: v . {1,1,1,0}
        __m128 mask = _mm_set_ps(0.0f, 1.0f, 1.0f, 1.0f);
        __m128 prod = _mm_mul_ps(v, mask);
        __m128 shuf = _mm_movehdup_ps(prod);
        __m128 sums = _mm_add_ps(prod, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }

    /// Set w component to zero.
    [[nodiscard]] PULSE_FORCE_INLINE __m128 zeroW(__m128 v) noexcept {
        // Blend: keep x,y,z from v, take w from zero
        return _mm_blend_ps(v, _mm_setzero_ps(), 0b1000);
    }

    /// Cross product for 3-component vectors stored in __m128 (w=0).
    [[nodiscard]] PULSE_FORCE_INLINE __m128 cross3(__m128 a, __m128 b) noexcept {
        // a x b = (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
        __m128 a_yzx = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 b_yzx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 c = _mm_sub_ps(
            _mm_mul_ps(a, b_yzx),
            _mm_mul_ps(a_yzx, b)
        );
        return _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 2, 1));
    }

    /// Dot product of 3-component vectors (SSE4.1 _mm_dp_ps).
    [[nodiscard]] PULSE_FORCE_INLINE float dot3(__m128 a, __m128 b) noexcept {
        // Mask 0x71: multiply x,y,z; store result in x.
        return _mm_cvtss_f32(_mm_dp_ps(a, b, 0x71));
    }

    /// Dot product of 4-component vectors.
    [[nodiscard]] PULSE_FORCE_INLINE float dot4(__m128 a, __m128 b) noexcept {
        return _mm_cvtss_f32(_mm_dp_ps(a, b, 0xF1));
    }

    /// Dot product returning result broadcast to all lanes (useful for chains).
    [[nodiscard]] PULSE_FORCE_INLINE __m128 dot3_broadcast(__m128 a, __m128 b) noexcept {
        return _mm_dp_ps(a, b, 0x7F);
    }

    /// Length squared of a 3-component vector.
    [[nodiscard]] PULSE_FORCE_INLINE float lengthSq3(__m128 v) noexcept {
        return dot3(v, v);
    }

    /// Length of a 3-component vector.
    [[nodiscard]] PULSE_FORCE_INLINE float length3(__m128 v) noexcept {
        return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(v, v, 0x71)));
    }

    /// Normalize a 3-component vector. Returns zero vector if length is near-zero.
    [[nodiscard]] PULSE_FORCE_INLINE __m128 normalize3(__m128 v) noexcept {
        __m128 dp = _mm_dp_ps(v, v, 0x7F); // dot broadcast to all lanes
        // Check for near-zero length
        __m128 eps = _mm_set1_ps(1.0e-12f);
        __m128 mask = _mm_cmpgt_ps(dp, eps);
        __m128 inv_len = _mm_rsqrt_ps(dp);
        // One Newton-Raphson iteration for precision
        __m128 half = _mm_set1_ps(0.5f);
        __m128 three = _mm_set1_ps(3.0f);
        __m128 muls = _mm_mul_ps(_mm_mul_ps(dp, inv_len), inv_len);
        inv_len = _mm_mul_ps(_mm_mul_ps(half, inv_len), _mm_sub_ps(three, muls));

        __m128 result = _mm_mul_ps(v, inv_len);
        return _mm_and_ps(result, mask); // zero out if length was ~0
    }

} // namespace simd
#endif // PULSE_SIMD_SSE42

} // namespace pulse
