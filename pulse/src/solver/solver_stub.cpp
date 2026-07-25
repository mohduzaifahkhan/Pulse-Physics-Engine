/**
 * @file solver_stub.cpp
 * @brief Compilation stub for the Pulse solver module.
 *
 * Ensures all solver headers compile cleanly as part of the static library.
 */

#include <pulse/solver/solver_common.h>
#include <pulse/solver/constraint_base.h>
#include <pulse/solver/contact_solver.h>
#include <pulse/solver/solver.h>

namespace pulse {
namespace {
    [[maybe_unused]] void solver_compile_check() {
        // solver_common.h
        SolverConfig config;
        (void)config.velocityIterations;
        (void)config.baumgarte;

        SolverBody body;
        (void)body.isStatic();
        body.applyLinearImpulse(Vec3::unitX());
        body.applyAngularImpulse(Vec3::unitY());

        VelocityConstraint vc;
        (void)vc.normalMass;

        PositionConstraint pc;
        (void)pc.penetration;

        SolverStats stats;
        (void)stats.positionSolved;

        // constraint_base.h
        ConstraintHeader header(ConstraintType::Contact, 0, 1);
        (void)header.isEnabled();
        header.setEnabled(false);

        ContactConstraintGroup group;
        (void)group.count;

        // contact_solver.h
        ContactSolver solver;
        (void)solver.velocityConstraintCount();
        (void)solver.positionConstraintCount();
        (void)solver.groupCount();

        // solver.h — just verify types exist
        (void)static_cast<void(*)(SolverBody*, uint32_t, float)>(&integratePositions);
        (void)static_cast<void(*)(SolverBody*, uint32_t, Vec3, float)>(&applyGravity);
    }
}
}
