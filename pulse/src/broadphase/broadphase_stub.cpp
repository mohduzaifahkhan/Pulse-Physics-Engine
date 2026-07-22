/**
 * @file broadphase_stub.cpp
 * @brief Compilation stub for the Pulse broadphase module.
 *
 * Ensures all broadphase headers compile cleanly as part of the static library.
 */

#include <pulse/broadphase/broadphase_common.h>
#include <pulse/broadphase/dynamic_aabb_tree.h>
#include <pulse/broadphase/sap.h>
#include <pulse/broadphase/uniform_grid.h>
#include <pulse/broadphase/bvh.h>

namespace pulse {
namespace {
    [[maybe_unused]] void broadphase_compile_check() {
        DynamicAABBTree tree;
        SAP sap;
        UniformGrid grid;
        BVH bvh;
        (void)tree; (void)sap; (void)grid; (void)bvh;
    }
}
}
