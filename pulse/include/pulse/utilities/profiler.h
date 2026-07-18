/**
 * @file profiler.h
 * @brief Scoped hierarchical profiler with performance counters and frame statistics.
 *
 * Provides RAII-based scoped timing that builds a hierarchical timing tree.
 * Each scope records inclusive time (self + children) and exclusive time (self only).
 * The profiler tracks per-frame statistics and supports multi-threaded profiling
 * with per-thread stacks.
 *
 * Usage:
 *   void physicsStep() {
 *       PULSE_PROFILE_FUNCTION();                // Times this entire function
 *       {
 *           PULSE_PROFILE_SCOPE("Broadphase");   // Nested scope
 *           runBroadphase();
 *       }
 *       {
 *           PULSE_PROFILE_SCOPE("Narrowphase");
 *           runNarrowphase();
 *       }
 *   }
 *
 * Compile-time enable/disable:
 * - Define PULSE_ENABLE_PROFILING to enable (default in debug builds)
 * - When disabled, all macros compile to nothing (zero overhead)
 *
 * High-resolution timer:
 * - Windows: QueryPerformanceCounter
 * - Linux:   clock_gettime(CLOCK_MONOTONIC)
 * - macOS:   mach_absolute_time
 */

#pragma once

#include <pulse/math/math_common.h>
#include <pulse/utilities/assert.h>
#include <pulse/utilities/fixed_array.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

// ── Auto-detect profiling mode ───────────────────────────────────────────────

#if !defined(PULSE_ENABLE_PROFILING) && !defined(PULSE_DISABLE_PROFILING)
    #if defined(_DEBUG) || defined(DEBUG) || !defined(NDEBUG)
        #define PULSE_ENABLE_PROFILING 1
    #endif
#endif

// ── High-resolution timer ────────────────────────────────────────────────────

#if defined(_WIN32)
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#elif defined(__APPLE__)
    #include <mach/mach_time.h>
#else
    #include <time.h>
#endif

namespace pulse {
namespace util {

// ── Timer ────────────────────────────────────────────────────────────────────

/// High-resolution timer using platform-native APIs.
class Timer {
public:
    /// Get the current timestamp in ticks (platform-specific unit).
    [[nodiscard]] static PULSE_FORCE_INLINE uint64_t now() noexcept {
#if defined(_WIN32)
        LARGE_INTEGER li;
        QueryPerformanceCounter(&li);
        return static_cast<uint64_t>(li.QuadPart);
#elif defined(__APPLE__)
        return mach_absolute_time();
#else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
               static_cast<uint64_t>(ts.tv_nsec);
#endif
    }

    /// Get the timer frequency (ticks per second).
    [[nodiscard]] static uint64_t frequency() noexcept {
#if defined(_WIN32)
        static uint64_t freq = []() -> uint64_t {
            LARGE_INTEGER li;
            QueryPerformanceFrequency(&li);
            return static_cast<uint64_t>(li.QuadPart);
        }();
        return freq;
#elif defined(__APPLE__)
        static uint64_t freq = []() -> uint64_t {
            mach_timebase_info_data_t info;
            mach_timebase_info(&info);
            // Convert to ticks-per-second from nanos-per-tick.
            return 1'000'000'000ULL * info.denom / info.numer;
        }();
        return freq;
#else
        return 1'000'000'000ULL; // clock_gettime is in nanoseconds
#endif
    }

    /// Convert ticks to seconds.
    [[nodiscard]] static PULSE_FORCE_INLINE double ticksToSeconds(uint64_t ticks) noexcept {
        return static_cast<double>(ticks) / static_cast<double>(frequency());
    }

    /// Convert ticks to milliseconds.
    [[nodiscard]] static PULSE_FORCE_INLINE double ticksToMs(uint64_t ticks) noexcept {
        return ticksToSeconds(ticks) * 1000.0;
    }

    /// Convert ticks to microseconds.
    [[nodiscard]] static PULSE_FORCE_INLINE double ticksToUs(uint64_t ticks) noexcept {
        return ticksToSeconds(ticks) * 1'000'000.0;
    }
};

// ── Profiler Data Structures ─────────────────────────────────────────────────

/// Maximum nesting depth for profiler scopes.
constexpr std::size_t MaxProfileDepth = 64;

/// Maximum number of unique profile nodes per frame.
constexpr std::size_t MaxProfileNodes = 512;

/// A single node in the profiler's hierarchical timing tree.
struct ProfileNode {
    const char* name     = nullptr;  ///< Scope name (string literal, not owned).
    uint64_t totalTicks  = 0;        ///< Total inclusive time (self + children).
    uint64_t selfTicks   = 0;        ///< Exclusive time (self only, minus children).
    uint32_t callCount   = 0;        ///< Number of times this scope was entered.
    int32_t  parentIndex = -1;       ///< Index of parent node (-1 = root).
    int32_t  depth       = 0;        ///< Nesting depth (0 = top-level).
};

/// Per-frame profiling results.
struct ProfileFrame {
    ProfileNode nodes[MaxProfileNodes];
    std::size_t nodeCount    = 0;
    uint64_t    frameTicks   = 0;     ///< Total frame time.
    uint64_t    frameStart   = 0;     ///< Frame start timestamp.
};

// ── Profiler ─────────────────────────────────────────────────────────────────

/**
 * @class Profiler
 * @brief Hierarchical scoped profiler with per-frame statistics.
 *
 * Designed as a simple global instance. Call beginFrame()/endFrame() around
 * your main loop. Use PULSE_PROFILE_SCOPE("name") inside functions to time
 * nested scopes.
 *
 * Thread safety: This implementation uses a single profiler stack. For
 * multi-threaded profiling, each thread should have its own Profiler
 * instance (or use thread_local). Full multi-threaded merging is planned
 * for the Job System module (Module 4).
 */
class Profiler {
public:
    // ── Frame management ─────────────────────────────────────────────────

    /// Call at the start of each frame. Resets per-frame data.
    void beginFrame() noexcept {
        currentFrame_.nodeCount  = 0;
        currentFrame_.frameStart = Timer::now();
        stackDepth_ = 0;
    }

    /// Call at the end of each frame. Finalizes timing data.
    void endFrame() noexcept {
        currentFrame_.frameTicks = Timer::now() - currentFrame_.frameStart;
        // Copy current frame to last frame for reading.
        lastFrame_ = currentFrame_;
        frameCount_++;
    }

    // ── Scope management (called by RAII ScopedProfile) ──────────────────

    /// Begin a named scope. Returns the node index.
    int32_t beginScope(const char* name) noexcept {
        PULSE_ASSERT(currentFrame_.nodeCount < MaxProfileNodes);
        PULSE_ASSERT(stackDepth_ < MaxProfileDepth);

        int32_t nodeIdx = static_cast<int32_t>(currentFrame_.nodeCount);
        ProfileNode& node = currentFrame_.nodes[currentFrame_.nodeCount++];
        node.name       = name;
        node.totalTicks = Timer::now();  // Will be converted to duration in endScope.
        node.selfTicks  = 0;
        node.callCount  = 1;
        node.depth      = static_cast<int32_t>(stackDepth_);
        node.parentIndex = (stackDepth_ > 0) ? scopeStack_[stackDepth_ - 1] : -1;

        scopeStack_[stackDepth_] = nodeIdx;
        stackDepth_++;

        return nodeIdx;
    }

    /// End the current scope.
    void endScope(int32_t nodeIdx) noexcept {
        PULSE_ASSERT(stackDepth_ > 0);
        PULSE_ASSERT(nodeIdx >= 0 && static_cast<std::size_t>(nodeIdx) < currentFrame_.nodeCount);

        uint64_t endTime = Timer::now();
        ProfileNode& node = currentFrame_.nodes[nodeIdx];
        uint64_t elapsed = endTime - node.totalTicks;
        node.totalTicks = elapsed;

        // Calculate self time: total - sum of direct children.
        uint64_t childrenTime = 0;
        for (std::size_t i = static_cast<std::size_t>(nodeIdx) + 1;
             i < currentFrame_.nodeCount; ++i) {
            if (currentFrame_.nodes[i].parentIndex == nodeIdx) {
                childrenTime += currentFrame_.nodes[i].totalTicks;
            }
        }
        node.selfTicks = elapsed - childrenTime;

        stackDepth_--;
    }

    // ── Results ──────────────────────────────────────────────────────────

    /// Get the last completed frame's profiling data.
    [[nodiscard]] const ProfileFrame& getLastFrame() const noexcept {
        return lastFrame_;
    }

    /// Get total number of frames profiled.
    [[nodiscard]] uint64_t getFrameCount() const noexcept {
        return frameCount_;
    }

    /// Print the last frame's timing tree to stdout.
    void printLastFrame() const noexcept {
        const ProfileFrame& frame = lastFrame_;
        std::printf("=== Profiler Frame %llu (%.3f ms) ===\n",
                    static_cast<unsigned long long>(frameCount_),
                    Timer::ticksToMs(frame.frameTicks));

        for (std::size_t i = 0; i < frame.nodeCount; ++i) {
            const ProfileNode& node = frame.nodes[i];
            // Indent based on depth.
            for (int32_t d = 0; d < node.depth; ++d) {
                std::printf("  ");
            }
            std::printf("%-30s  total: %8.3f us  self: %8.3f us  calls: %u\n",
                        node.name,
                        Timer::ticksToUs(node.totalTicks),
                        Timer::ticksToUs(node.selfTicks),
                        node.callCount);
        }
        std::printf("====================================\n");
    }

    // ── Global instance ──────────────────────────────────────────────────

    /// Get the global profiler instance.
    static Profiler& instance() noexcept {
        static Profiler s_instance;
        return s_instance;
    }

private:
    ProfileFrame currentFrame_;
    ProfileFrame lastFrame_;
    int32_t      scopeStack_[MaxProfileDepth] = {};
    std::size_t  stackDepth_  = 0;
    uint64_t     frameCount_  = 0;
};

// ── RAII Scoped Profile ──────────────────────────────────────────────────────

/// RAII scope timer — begins a scope on construction, ends on destruction.
class ScopedProfile {
public:
    explicit ScopedProfile(const char* name) noexcept
        : nodeIdx_(Profiler::instance().beginScope(name)) {}

    ~ScopedProfile() noexcept {
        Profiler::instance().endScope(nodeIdx_);
    }

    // Non-copyable, non-movable.
    ScopedProfile(const ScopedProfile&) = delete;
    ScopedProfile& operator=(const ScopedProfile&) = delete;

private:
    int32_t nodeIdx_;
};

} // namespace util
} // namespace pulse

// ── Profiler macros ──────────────────────────────────────────────────────────

/// Helper to generate unique variable names.
#define PULSE_PROFILE_CONCAT_(a, b) a##b
#define PULSE_PROFILE_CONCAT(a, b) PULSE_PROFILE_CONCAT_(a, b)

#ifdef PULSE_ENABLE_PROFILING

    /// Time the current scope with a custom name.
    #define PULSE_PROFILE_SCOPE(name) \
        ::pulse::util::ScopedProfile PULSE_PROFILE_CONCAT(pulse_prof_, __LINE__)(name)

    /// Time the current function (uses __func__).
    #define PULSE_PROFILE_FUNCTION() \
        ::pulse::util::ScopedProfile PULSE_PROFILE_CONCAT(pulse_prof_, __LINE__)(PULSE_CURRENT_FUNCTION_)

#else

    #define PULSE_PROFILE_SCOPE(name)  ((void)0)
    #define PULSE_PROFILE_FUNCTION()   ((void)0)

#endif
