/**
 * @file sleep_manager.h
 * @brief Sleep state management — puts idle bodies to sleep and wakes them on demand.
 *
 * Bodies whose linear and angular velocities remain below configurable
 * thresholds for a sustained period are flagged as Sleeping. Sleeping bodies
 * are excluded from integration and constraint solving, dramatically reducing
 * CPU cost for stable stacks and resting objects.
 *
 * Wake conditions:
 *  - External force or impulse applied.
 *  - Contact with an awake body.
 *  - Explicit wake request (wakeBody / wakeIsland).
 *
 * Island-level sleeping: an entire island can be put to sleep when all bodies
 * in it are below threshold. If any body in a sleeping island is woken, the
 * entire island wakes.
 *
 * Thread safety: NOT thread-safe. External synchronization required.
 */

#pragma once

#include <pulse/rigidbody/rigid_body_common.h>
#include <pulse/rigidbody/rigid_body.h>
#include <pulse/math/math_common.h>
#include <pulse/math/vec3.h>

#include <cstddef>
#include <cstdint>

namespace pulse {

// ── Sleep configuration ──────────────────────────────────────────────────────

/**
 * @struct SleepConfig
 * @brief Configuration parameters for the sleep system.
 */
struct SleepConfig {
    /// Linear velocity threshold (m/s). Below this, body is a sleep candidate.
    float linearSleepThreshold;

    /// Angular velocity threshold (rad/s). Below this, body is a sleep candidate.
    float angularSleepThreshold;

    /// Time (seconds) a body must remain below thresholds before sleeping.
    float timeToSleep;

    /// Minimum time a body must stay awake after being woken (prevents flicker).
    float minAwakeTime;

    /// Default: production-tuned values.
    PULSE_FORCE_INLINE SleepConfig() noexcept
        : linearSleepThreshold(0.08f),    // ~8 cm/s
          angularSleepThreshold(0.1f),     // ~5.7 deg/s
          timeToSleep(0.5f),               // 0.5 seconds at rest
          minAwakeTime(0.2f)               // Stay awake at least 0.2s
    {}

    /// Custom configuration.
    PULSE_FORCE_INLINE SleepConfig(float linThresh, float angThresh,
                                    float sleepTime, float minAwake) noexcept
        : linearSleepThreshold(linThresh),
          angularSleepThreshold(angThresh),
          timeToSleep(sleepTime),
          minAwakeTime(minAwake)
    {}
};

// ── SleepManager ─────────────────────────────────────────────────────────────

/**
 * @class SleepManager
 * @brief Manages sleep/wake state transitions for rigid bodies.
 */
class SleepManager {
public:
    SleepConfig config;

    explicit SleepManager(SleepConfig cfg = SleepConfig()) noexcept
        : config(cfg) {}

    // ── Per-frame update ─────────────────────────────────────────────────

    /**
     * @brief Update sleep state for all bodies in the store.
     *
     * For each dynamic body:
     *  - If velocity is below thresholds, increment sleep timer.
     *  - If timer exceeds timeToSleep, set Sleeping flag.
     *  - If velocity is above thresholds, reset timer and clear Sleeping flag.
     *
     * @param store  The body data store.
     * @param dt     Time step (seconds).
     */
    void updateSleep(RigidBodyStore& store, float dt) noexcept {
        for (std::size_t i = 0; i < store.size(); ++i) {
            // Only dynamic bodies can sleep.
            if (!store.isDynamic(i)) continue;

            float linSpeedSq = store.linearVelocity(i).lengthSq();
            float angSpeedSq = store.angularVelocity(i).lengthSq();

            float linThreshSq = config.linearSleepThreshold * config.linearSleepThreshold;
            float angThreshSq = config.angularSleepThreshold * config.angularSleepThreshold;

            if (linSpeedSq < linThreshSq && angSpeedSq < angThreshSq) {
                // Below thresholds — accumulate sleep timer.
                float timer = store.sleepTimer(i) + dt;
                store.setSleepTimer(i, timer);

                if (timer >= config.timeToSleep) {
                    // Put to sleep.
                    store.addFlags(i, BodyFlags::Sleeping);
                    store.linearVelocity(i) = Vec3::zero();
                    store.angularVelocity(i) = Vec3::zero();
                }
            } else {
                // Above thresholds — wake up.
                store.setSleepTimer(i, 0.0f);
                store.removeFlags(i, BodyFlags::Sleeping);
            }
        }
    }

    // ── Individual body operations ───────────────────────────────────────

    /// Wake a single body by dense index.
    PULSE_FORCE_INLINE void wakeBody(RigidBodyStore& store, std::size_t index) noexcept {
        store.removeFlags(index, BodyFlags::Sleeping);
        store.setSleepTimer(index, 0.0f);
    }

    /// Check if a body is awake.
    [[nodiscard]] PULSE_FORCE_INLINE bool isAwake(const RigidBodyStore& store,
                                                    std::size_t index) const noexcept {
        return !store.isSleeping(index);
    }

    /// Forcefully put a body to sleep.
    PULSE_FORCE_INLINE void sleepBody(RigidBodyStore& store, std::size_t index) noexcept {
        store.addFlags(index, BodyFlags::Sleeping);
        store.linearVelocity(index) = Vec3::zero();
        store.angularVelocity(index) = Vec3::zero();
        store.setSleepTimer(index, config.timeToSleep);
    }

    // ── Island-level operations ──────────────────────────────────────────

    /// Wake all bodies in an island by their dense indices.
    void wakeIsland(RigidBodyStore& store, const uint32_t* bodyIndices,
                    uint32_t count) noexcept {
        for (uint32_t i = 0; i < count; ++i) {
            wakeBody(store, bodyIndices[i]);
        }
    }

    /// Sleep all bodies in an island.
    void sleepIsland(RigidBodyStore& store, const uint32_t* bodyIndices,
                     uint32_t count) noexcept {
        for (uint32_t i = 0; i < count; ++i) {
            sleepBody(store, bodyIndices[i]);
        }
    }

    /// Check if all bodies in an island are below sleep thresholds.
    [[nodiscard]] bool canIslandSleep(const RigidBodyStore& store,
                                      const uint32_t* bodyIndices,
                                      uint32_t count) const noexcept {
        float linThreshSq = config.linearSleepThreshold * config.linearSleepThreshold;
        float angThreshSq = config.angularSleepThreshold * config.angularSleepThreshold;

        for (uint32_t i = 0; i < count; ++i) {
            uint32_t idx = bodyIndices[i];
            if (!store.isDynamic(idx)) continue;

            float linSq = store.linearVelocity(idx).lengthSq();
            float angSq = store.angularVelocity(idx).lengthSq();

            if (linSq >= linThreshSq || angSq >= angThreshSq) {
                return false;
            }

            if (store.sleepTimer(idx) < config.timeToSleep) {
                return false;
            }
        }
        return true;
    }
};

} // namespace pulse
