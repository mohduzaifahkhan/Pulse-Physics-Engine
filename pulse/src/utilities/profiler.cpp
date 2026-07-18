/**
 * @file profiler.cpp
 * @brief Non-inline profiler state and frame reporting.
 *
 * Currently minimal — the Profiler class is mostly header-inline.
 * This translation unit ensures the static Profiler instance has a
 * single definition in the final binary and provides a home for any
 * future non-inline profiler logic (e.g., thread-safe frame merging,
 * file output, Chrome tracing export).
 */

#include <pulse/utilities/profiler.h>

namespace pulse {
namespace util {

// Force instantiation of the Profiler singleton in this TU.
// This ensures exactly one copy across the entire program.
Profiler& getGlobalProfiler() noexcept {
    return Profiler::instance();
}

} // namespace util
} // namespace pulse
