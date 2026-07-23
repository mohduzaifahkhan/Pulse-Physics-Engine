/**
 * @file narrowphase_stub.cpp
 * @brief Compilation stub for the Pulse narrowphase module.
 *
 * Ensures all narrowphase headers compile cleanly as part of the static library.
 */

#include <pulse/narrowphase/narrowphase_common.h>
#include <pulse/narrowphase/sat.h>
#include <pulse/narrowphase/gjk.h>
#include <pulse/narrowphase/epa.h>
#include <pulse/narrowphase/mpr.h>
#include <pulse/narrowphase/ccd.h>
#include <pulse/narrowphase/collision_dispatch.h>

namespace pulse {
namespace {
    [[maybe_unused]] void narrowphase_compile_check() {
        ContactManifold manifold;
        NarrowPhaseConfig config;
        SupportPoint sp;
        Simplex simplex;
        GjkResult gjkResult;
        EpaResult epaResult;
        MprResult mprResult;
        CcdResult ccdResult;
        (void)manifold; (void)config; (void)sp; (void)simplex;
        (void)gjkResult; (void)epaResult; (void)mprResult; (void)ccdResult;
    }
}
}
