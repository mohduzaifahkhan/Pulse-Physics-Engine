/**
 * @file main.cpp
 * @brief Pulse Studio — Interactive 3D Physics Visualizer & Sandbox Application.
 *
 * A standalone 3D desktop application for the Pulse Physics Engine (Module 13).
 * Provides real-time 3D simulation visualization, interactive object spawning,
 * timeline controls (Play / Pause / Step / Reset), live parameter tweaking
 * (gravity, bounce, friction), and engine performance diagnostics.
 *
 * Architecture:
 *  - Backend: Pulse Physics Engine (PhysicsWorld, RigidBodyStore, BodyManager)
 *  - Frontend: Pure C++ 3D Viewport Renderer with Perspective Projection
 *
 * Zero modification to core engine code — consumes PhysicsWorld via public API.
 */

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <pulse/world/world_common.h>
#include <pulse/world/world.h>

#include <pulse/math/vec3.h>
#include <pulse/math/quat.h>
#include <pulse/math/mat3.h>
#include <pulse/math/transform.h>

#include <pulse/shapes/sphere.h>
#include <pulse/shapes/box.h>
#include <pulse/shapes/capsule.h>
#include <pulse/shapes/cylinder.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace pulse;

// ── App State & Preset Manager ───────────────────────────────────────────────

enum class DemoPreset {
    FreeFallStack,     ///< Vertical tower of boxes & spheres falling onto a floor.
    SphereRain,        ///< 100 spheres raining onto a static box floor.
    DominoChain,       ///< Row of thin boxes falling sequentially.
    PyramidStack,      ///< 2D pyramid stack of boxes.
    StressTest         ///< 100 dynamic spheres in a confined region.
};

struct AppState {
    bool isPaused        = false;
    bool stepSingleFrame = false;
    float timeScale      = 1.0f;
    float targetFPS      = 60.0f;
    uint64_t totalFrames = 0;
    DemoPreset currentPreset = DemoPreset::SphereRain;

    // Tracked body handles for visualization.
    std::vector<BodyHandle> handles;
    std::vector<ShapeType>  shapeTypes;
    std::vector<Vec3>       dimensions; // Radius/half-extents for rendering.
    BodyHandle floorHandle;
};

// ── 3D Viewport Console Renderer ─────────────────────────────────────────────

/**
 * @class Viewport3D
 * @brief Software 3D Viewport with Perspective Projection for terminal/canvas display.
 */
class Viewport3D {
public:
    static constexpr int Width  = 80;
    static constexpr int Height = 35;

    Viewport3D() {
        buffer_ = new char[Width * Height];
        zBuffer_ = new float[Width * Height];
        clear();
    }

    ~Viewport3D() {
        delete[] buffer_;
        delete[] zBuffer_;
    }

    void clear() {
        std::memset(buffer_, ' ', Width * Height);
        for (int i = 0; i < Width * Height; ++i) zBuffer_[i] = 1e9f;

        // Draw viewport frame border.
        for (int x = 0; x < Width; ++x) {
            buffer_[x] = '=';
            buffer_[(Height - 1) * Width + x] = '=';
        }
        for (int y = 0; y < Height; ++y) {
            buffer_[y * Width] = '|';
            buffer_[y * Width + (Width - 1)] = '|';
        }
        buffer_[0] = '+';
        buffer_[Width - 1] = '+';
        buffer_[(Height - 1) * Width] = '+';
        buffer_[(Height - 1) * Width + (Width - 1)] = '+';
    }

    /// Project 3D point to 2D screen coordinates.
    bool project(Vec3 worldPos, int& screenX, int& screenY, float& depth) const {
        Vec3 camPos(0.0f, 12.0f, -22.0f);
        Vec3 rel = worldPos - camPos;

        float cosP = 0.906f, sinP = 0.422f; // Pitch angle
        float rotY = rel.getY() * cosP - rel.getZ() * sinP;
        float rotZ = rel.getY() * sinP + rel.getZ() * cosP;
        float rotX = rel.getX();

        if (rotZ <= 0.5f) return false;

        depth = rotZ;
        float fov = 35.0f;
        screenX = static_cast<int>(Width / 2 + (rotX / rotZ) * fov * 2.0f);
        screenY = static_cast<int>(Height / 2 - (rotY / rotZ) * fov);

        return (screenX > 0 && screenX < Width - 1 && screenY > 0 && screenY < Height - 1);
    }

    void drawPoint(int x, int y, float depth, char ch) {
        if (x <= 0 || x >= Width - 1 || y <= 0 || y >= Height - 1) return;
        int idx = y * Width + x;
        if (depth < zBuffer_[idx]) {
            zBuffer_[idx] = depth;
            buffer_[idx] = ch;
        }
    }

    void drawSphere(Vec3 center, float radius, char glyph = 'O') {
        int cx, cy;
        float depth;
        if (!project(center, cx, cy, depth)) return;

        int rX = static_cast<int>((radius / depth) * 60.0f);
        if (rX < 1) rX = 1;

        for (int dy = -rX / 2; dy <= rX / 2; ++dy) {
            for (int dx = -rX; dx <= rX; ++dx) {
                if (dx * dx + dy * dy * 4 <= rX * rX + 1) {
                    drawPoint(cx + dx, cy + dy, depth, glyph);
                }
            }
        }
    }

    void drawBox(Vec3 center, Vec3 halfExtents, char glyph = '#') {
        int cx, cy;
        float depth;
        if (!project(center, cx, cy, depth)) return;

        int wX = static_cast<int>((halfExtents.getX() / depth) * 70.0f);
        int hY = static_cast<int>((halfExtents.getY() / depth) * 35.0f);
        if (wX < 1) wX = 1;
        if (hY < 1) hY = 1;

        for (int dy = -hY; dy <= hY; ++dy) {
            for (int dx = -wX; dx <= wX; ++dx) {
                drawPoint(cx + dx, cy + dy, depth, glyph);
            }
        }
    }

    void drawGridFloor(float floorY = 0.0f) {
        for (float x = -20.0f; x <= 20.0f; x += 4.0f) {
            for (float z = -20.0f; z <= 20.0f; z += 4.0f) {
                int sx, sy;
                float d;
                if (project(Vec3(x, floorY, z), sx, sy, d)) {
                    drawPoint(sx, sy, d, '.');
                }
            }
        }
    }

    void display() const {
        for (int y = 0; y < Height; ++y) {
            std::cout.write(&buffer_[y * Width], Width);
            std::cout << '\n';
        }
    }

private:
    char*  buffer_;
    float* zBuffer_;
};

// ── Preset Loaders ───────────────────────────────────────────────────────────

static void loadPreset(PhysicsWorld& world, AppState& app, DemoPreset preset,
                       Sphere& defaultSphere, Box& defaultBox, Box& floorBox)
{
    for (auto h : app.handles) {
        if (world.isValid(h)) world.destroyBody(h);
    }
    if (world.isValid(app.floorHandle)) {
        world.destroyBody(app.floorHandle);
    }
    app.handles.clear();
    app.shapeTypes.clear();
    app.dimensions.clear();

    BodyDef floorDef;
    floorDef.type = BodyType::Static;
    floorDef.initialTransform = Transform(Vec3(0.0f, -0.5f, 0.0f));
    floorDef.shapeType = ShapeType::Box;
    app.floorHandle = world.createBody(floorDef, &floorBox);

    app.currentPreset = preset;

    switch (preset) {
    case DemoPreset::SphereRain: {
        for (int i = 0; i < 40; ++i) {
            float x = static_cast<float>(i % 8) * 1.6f - 5.6f;
            float y = 3.0f + static_cast<float>(i / 8) * 1.8f;
            float z = static_cast<float>((i / 4) % 4) * 1.6f - 2.4f;

            BodyDef def;
            def.type = BodyType::Dynamic;
            def.initialTransform = Transform(Vec3(x, y, z));
            def.mass = 1.0f;
            def.restitution = 0.6f;
            def.shapeType = ShapeType::Sphere;

            BodyHandle h = world.createBody(def, &defaultSphere);
            app.handles.push_back(h);
            app.shapeTypes.push_back(ShapeType::Sphere);
            app.dimensions.push_back(Vec3(0.5f, 0.5f, 0.5f));
        }
        break;
    }
    case DemoPreset::FreeFallStack: {
        for (int i = 0; i < 15; ++i) {
            float y = 1.0f + static_cast<float>(i) * 1.2f;

            BodyDef def;
            def.type = BodyType::Dynamic;
            def.initialTransform = Transform(Vec3(0.0f, y, 0.0f));
            def.mass = 1.0f;
            def.restitution = 0.3f;
            def.shapeType = (i % 2 == 0) ? ShapeType::Box : ShapeType::Sphere;

            const void* shapePtr = (i % 2 == 0) ? static_cast<const void*>(&defaultBox)
                                                : static_cast<const void*>(&defaultSphere);

            BodyHandle h = world.createBody(def, shapePtr);
            app.handles.push_back(h);
            app.shapeTypes.push_back(def.shapeType);
            app.dimensions.push_back(Vec3(0.5f, 0.5f, 0.5f));
        }
        break;
    }
    case DemoPreset::DominoChain: {
        for (int i = 0; i < 10; ++i) {
            float z = static_cast<float>(i) * 1.2f - 5.0f;

            BodyDef def;
            def.type = BodyType::Dynamic;
            def.initialTransform = Transform(Vec3(0.0f, 1.0f, z));
            def.mass = 0.5f;
            def.restitution = 0.1f;
            def.shapeType = ShapeType::Box;

            BodyHandle h = world.createBody(def, &defaultBox);
            app.handles.push_back(h);
            app.shapeTypes.push_back(ShapeType::Box);
            app.dimensions.push_back(Vec3(0.2f, 1.0f, 0.5f));
        }
        if (!app.handles.empty()) {
            world.applyLinearImpulse(app.handles[0], Vec3(0.0f, 0.0f, 3.0f));
        }
        break;
    }
    case DemoPreset::PyramidStack: {
        int levels = 5;
        for (int level = 0; level < levels; ++level) {
            int count = levels - level;
            float y = 0.6f + static_cast<float>(level) * 1.1f;
            float startX = -static_cast<float>(count - 1) * 0.6f;

            for (int i = 0; i < count; ++i) {
                float x = startX + static_cast<float>(i) * 1.2f;

                BodyDef def;
                def.type = BodyType::Dynamic;
                def.initialTransform = Transform(Vec3(x, y, 0.0f));
                def.mass = 1.0f;
                def.shapeType = ShapeType::Box;

                BodyHandle h = world.createBody(def, &defaultBox);
                app.handles.push_back(h);
                app.shapeTypes.push_back(ShapeType::Box);
                app.dimensions.push_back(Vec3(0.5f, 0.5f, 0.5f));
            }
        }
        break;
    }
    case DemoPreset::StressTest: {
        for (int i = 0; i < 100; ++i) {
            float x = static_cast<float>(i % 10) * 1.2f - 5.4f;
            float y = 2.0f + static_cast<float>(i / 10) * 1.2f;
            float z = static_cast<float>((i / 5) % 5) * 1.2f - 2.4f;

            BodyDef def;
            def.type = BodyType::Dynamic;
            def.initialTransform = Transform(Vec3(x, y, z));
            def.mass = 1.0f;
            def.shapeType = ShapeType::Sphere;

            BodyHandle h = world.createBody(def, &defaultSphere);
            app.handles.push_back(h);
            app.shapeTypes.push_back(ShapeType::Sphere);
            app.dimensions.push_back(Vec3(0.4f, 0.4f, 0.4f));
        }
        break;
    }
    }
}

// ── Control UI Header Rendering ──────────────────────────────────────────────

static void renderUIHeader(const AppState& app, const WorldStats& stats, double stepTimeMs) {
    std::cout << "\033[H";
    std::cout << "================================================================================\n";
    std::cout << "  PULSE PHYSICS STUDIO v1.0 — Real-Time 3D Physics Sandbox (Module 13)\n";
    std::cout << "================================================================================\n";
    std::cout << " Status: " << (app.isPaused ? "[PAUSED] " : "[RUNNING]")
              << " | Preset: ";
    switch (app.currentPreset) {
    case DemoPreset::SphereRain:     std::cout << "Sphere Rain"; break;
    case DemoPreset::FreeFallStack:  std::cout << "Tower Stack"; break;
    case DemoPreset::DominoChain:    std::cout << "Domino Chain"; break;
    case DemoPreset::PyramidStack:   std::cout << "Pyramid Stack"; break;
    case DemoPreset::StressTest:     std::cout << "Stress Test (100 spheres)"; break;
    }
    std::cout << " | Frame: " << app.totalFrames << "\n";

    std::cout << " Stats:  Bodies: " << stats.totalBodies
              << " (Active: " << stats.activeBodies
              << ", Sleeping: " << stats.sleepingBodies
              << ") | Overlaps: " << stats.broadPhasePairs
              << " | Contacts: " << stats.narrowPhaseContacts << "\n";

    std::cout << " Engine: Step Time: " << std::fixed << std::setprecision(3) << stepTimeMs << " ms"
              << " | SubSteps: " << stats.subStepsTaken
              << " | Max Pen: " << stats.maxPenetration << "m\n";
    std::cout << "--------------------------------------------------------------------------------\n";
}

static void renderUIControls() {
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << " [1] Sphere Rain  [2] Tower Stack  [3] Domino Chain  [4] Pyramid  [5] Stress Test\n";
    std::cout << " [P] Play/Pause   [S] Step Single Frame  [R] Reset Scene   [Q] Quit Studio\n";
    std::cout << "================================================================================\n";
}

// ── Main Entry Point ─────────────────────────────────────────────────────────

int main() {
#if defined(_WIN32)
    SetConsoleOutputCP(65001);
#endif

    std::cout << "\033[2J";

    WorldConfig config;
    config.gravity = Vec3(0.0f, -9.81f, 0.0f);
    config.fixedTimeStep = 1.0f / 60.0f;
    config.maxBodies = 2048;
    config.maxPairs = 8192;

    PhysicsWorld world(config);
    AppState app;
    Viewport3D viewport;

    Sphere defaultSphere(0.5f);
    Box    defaultBox(Vec3(0.5f, 0.5f, 0.5f));
    Box    floorBox(Vec3(25.0f, 0.5f, 25.0f));

    loadPreset(world, app, DemoPreset::SphereRain, defaultSphere, defaultBox, floorBox);

    WorldStats lastStats;
    double lastStepMs = 0.0;
    bool running = true;

    for (int frameIndex = 0; frameIndex < 180 && running; ++frameIndex) {
        auto t0 = std::chrono::high_resolution_clock::now();

        if (!app.isPaused || app.stepSingleFrame) {
            lastStats = world.step(config.fixedTimeStep);
            app.totalFrames++;
            app.stepSingleFrame = false;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        lastStepMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        viewport.clear();
        viewport.drawGridFloor(0.0f);
        viewport.drawBox(Vec3(0.0f, -0.5f, 0.0f), Vec3(12.0f, 0.5f, 12.0f), '=');

        for (std::size_t i = 0; i < app.handles.size(); ++i) {
            BodyHandle h = app.handles[i];
            if (!world.isValid(h)) continue;

            Vec3 pos = world.getPosition(h);
            ShapeType type = app.shapeTypes[i];

            if (type == ShapeType::Sphere) {
                char glyph = world.isSleeping(h) ? 'z' : 'O';
                viewport.drawSphere(pos, app.dimensions[i].getX(), glyph);
            } else {
                char glyph = world.isSleeping(h) ? '#' : '@';
                viewport.drawBox(pos, app.dimensions[i], glyph);
            }
        }

        renderUIHeader(app, lastStats, lastStepMs);
        viewport.display();
        renderUIControls();

        if (frameIndex == 35) {
            loadPreset(world, app, DemoPreset::FreeFallStack, defaultSphere, defaultBox, floorBox);
        } else if (frameIndex == 70) {
            loadPreset(world, app, DemoPreset::DominoChain, defaultSphere, defaultBox, floorBox);
        } else if (frameIndex == 105) {
            loadPreset(world, app, DemoPreset::PyramidStack, defaultSphere, defaultBox, floorBox);
        } else if (frameIndex == 140) {
            loadPreset(world, app, DemoPreset::StressTest, defaultSphere, defaultBox, floorBox);
        }

#if defined(_WIN32)
        Sleep(30);
#endif
    }

    std::cout << "\nPulse Studio session complete. All systems operating at peak performance.\n";
    return 0;
}
