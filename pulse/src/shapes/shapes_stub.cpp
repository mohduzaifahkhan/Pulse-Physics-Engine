/**
 * @file shapes_stub.cpp
 * @brief Compilation stub for the Pulse shapes module.
 *
 * Ensures the shapes module has a translation unit in the pulse static library.
 * All shape types are header-only, so this file intentionally does minimal work.
 */

#include <pulse/shapes/shape_common.h>
#include <pulse/shapes/sphere.h>
#include <pulse/shapes/box.h>
#include <pulse/shapes/capsule.h>
#include <pulse/shapes/cylinder.h>
#include <pulse/shapes/convex_hull.h>
#include <pulse/shapes/tri_mesh.h>

// Force instantiation to catch compile errors in all shape headers.
namespace pulse {
namespace {
    [[maybe_unused]] void shapes_compile_check() {
        Sphere s;
        Box b;
        Capsule c;
        Cylinder cy;
        ConvexHull ch;
        TriMesh tm;
        (void)s; (void)b; (void)c; (void)cy; (void)ch; (void)tm;
    }
}
}
