/**
 * @file world_stub.cpp
 * @brief Compilation stub for the World module (Module 13).
 *
 * Ensures the world headers are compiled and linked into the pulse library.
 */

#include <pulse/world/world_common.h>
#include <pulse/world/world.h>

// Instantiation anchor — prevents the linker from discarding the TU.
namespace pulse {
namespace world_detail {
    static volatile int worldStubAnchor = 0;
} // namespace world_detail
} // namespace pulse
