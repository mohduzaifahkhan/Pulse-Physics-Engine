/**
 * @file constraints_stub.cpp
 * @brief Compilation stub for the Pulse constraints module (Module 10).
 *
 * Ensures all constraint headers compile cleanly as part of the static library.
 */

#include <pulse/constraints/constraint_common.h>
#include <pulse/constraints/distance_constraint.h>
#include <pulse/constraints/hinge_constraint.h>
#include <pulse/constraints/slider_constraint.h>
#include <pulse/constraints/cone_twist_constraint.h>
#include <pulse/constraints/six_dof_constraint.h>
#include <pulse/constraints/spring_constraint.h>
#include <pulse/constraints/motor_constraint.h>

namespace pulse {
namespace {
    [[maybe_unused]] void constraints_compile_check() {
        // constraint_common.h
        JointLimit limit(0.0f, 1.0f);
        (void)limit.isLocked();
        (void)limit.classify(0.5f);

        JointMotor motor(1.0f, 10.0f);
        (void)motor.targetVelocity;

        SoftConstraintParams soft(30.0f, 0.7f);
        soft.compute(1.0f / 60.0f);
        (void)soft.gamma;
        (void)soft.beta;

        JointRow row;
        (void)row.effectiveMass;

        SolverBody bodyA, bodyB;
        float em = computeJointEffectiveMass(bodyA, bodyB, row, 0.0f);
        (void)em;

        solveJointRow(row, bodyA, bodyB);

        float rav = relativeAngularVelocity(bodyA, bodyB, Vec3::unitY());
        (void)rav;

        float rlv = relativeLinearVelocity(bodyA, bodyB,
                                            Vec3::zero(), Vec3::zero(), Vec3::unitX());
        (void)rlv;

        setupLinearRow(row, bodyA, bodyB, Vec3::zero(), Vec3::zero(),
                       Vec3::unitX(), 0.0f, 0.2f, 60.0f);
        setupAngularRow(row, bodyA, bodyB, Vec3::unitY(), 0.0f, 0.2f, 60.0f);
        setupMotorRow(row, bodyA, bodyB, Vec3::unitY(), motor, 1.0f / 60.0f);
        setupLimitRow(row, bodyA, bodyB, Vec3::unitY(), 0.0f,
                      JointLimitState::AtLower, 0.2f, 60.0f);

        // distance_constraint.h
        DistanceConstraint dist(Vec3::zero(), Vec3(1, 0, 0), 1.0f, 0, 1);
        (void)dist.rowCount();
        (void)dist.getRow();

        // hinge_constraint.h
        HingeConstraint hinge(Vec3::zero(), Vec3::zero(),
                              Vec3::unitY(), Vec3::unitY(), 0, 1);
        (void)hinge.maxRowCount();
        (void)hinge.activeRowCount();

        // slider_constraint.h
        SliderConstraint slider(Vec3::zero(), Vec3::zero(), Vec3::unitX(), 0, 1);
        (void)slider.maxRowCount();
        (void)slider.getSliderPosition();

        // cone_twist_constraint.h
        ConeTwistConstraint ct(Vec3::zero(), Vec3::zero(),
                               Vec3::unitX(), Vec3::unitX(), 0, 1);
        ct.setSwingLimit(0.5f);
        ct.setTwistLimit(-0.3f, 0.3f);
        (void)ct.getSwingAngle();
        (void)ct.getTwistAngle();

        // six_dof_constraint.h
        SixDofConstraint sixDof(Vec3::zero(), Vec3::zero(), 0, 1);
        sixDof.lockDof(DofIndex::TransX);
        sixDof.freeDof(DofIndex::RotZ);
        sixDof.limitDof(DofIndex::TransY, -1.0f, 1.0f);
        sixDof.setMotor(DofIndex::RotX, 1.0f, 10.0f);
        (void)sixDof.activeRowCount();

        // spring_constraint.h
        SpringConstraint spring(Vec3::zero(), Vec3(1, 0, 0), 1.0f, 100.0f, 5.0f, 0, 1);
        (void)spring.rowCount();

        // motor_constraint.h
        MotorConstraint motorC(Vec3::unitY(), 100.0f, MotorMode::Angular, 0, 1);
        (void)motorC.rowCount();
        (void)motorC.getAccumulatedImpulse();
    }
}
} // namespace pulse
