/**
 * @file integration_stub.cpp
 * @brief Compilation stub for the Pulse integration module (Module 12).
 *
 * Ensures all integration headers compile cleanly as part of the static library.
 */

#include <pulse/integration/integration_common.h>
#include <pulse/integration/semi_implicit_euler.h>
#include <pulse/integration/velocity_verlet.h>
#include <pulse/integration/rk4.h>
#include <pulse/integration/integrator.h>

namespace pulse {
namespace {
    [[maybe_unused]] void integration_compile_check() {
        // integration_common.h
        IntegratorType type = IntegratorType::SemiImplicitEuler;
        (void)type;

        IntegrationConfig config;
        (void)config.type;
        (void)config.gravity;
        (void)config.maxLinearSpeed;
        (void)config.maxAngularSpeed;

        IntegrationConfig custom(IntegratorType::RK4, Vec3(0, -10, 0), 200.0f, 50.0f);
        (void)custom;

        // Utility functions
        Vec3 v = integration_detail::clampSpeed(Vec3(100, 0, 0), 50.0f);
        (void)v;
        v = integration_detail::applyDamping(Vec3(10, 0, 0), 0.1f, 1.0f / 60.0f);
        (void)v;
        Quat q = integration_detail::integrateRotation(
            Quat::identity(), Vec3(0, 1, 0), 1.0f / 60.0f);
        (void)q;

        // All integrators require a RigidBodyStore.
        BodyDef def;
        def.type = BodyType::Dynamic;
        def.mass = 1.0f;

        RigidBodyStore store(4);
        store.add(def);

        Vec3 gravity(0, -9.81f, 0);
        float dt = 1.0f / 60.0f;

        // semi_implicit_euler.h
        semiImplicitEulerIntegrateVelocities(store, gravity, dt);
        semiImplicitEulerIntegratePositions(store, dt);
        semiImplicitEulerIntegrate(store, gravity, dt, 500.0f, 100.0f);

        // velocity_verlet.h
        verletHalfKickAndDrift(store, gravity, dt);
        verletSecondHalfKick(store, gravity, dt, 500.0f, 100.0f);
        verletIntegrate(store, gravity, dt, 500.0f, 100.0f);

        // rk4.h
        rk4Integrate(store, gravity, dt, 500.0f, 100.0f);

        // integrator.h — unified dispatch
        integrateVelocities(store, config, dt);
        integratePositions(store, config, dt);
        integrate(store, config, dt);

        // Dispatch all types
        IntegrationConfig cfgEuler;
        cfgEuler.type = IntegratorType::SemiImplicitEuler;
        integrate(store, cfgEuler, dt);

        IntegrationConfig cfgVerlet;
        cfgVerlet.type = IntegratorType::VelocityVerlet;
        integrate(store, cfgVerlet, dt);

        IntegrationConfig cfgRK4;
        cfgRK4.type = IntegratorType::RK4;
        integrate(store, cfgRK4, dt);
    }
}
} // namespace pulse
