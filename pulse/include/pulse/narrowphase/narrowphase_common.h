/**
 * @file narrowphase_common.h
 * @brief Shared types for the narrow-phase collision detection module.
 *
 * Defines the contact point and manifold representations, Minkowski-difference
 * support structures, simplex for GJK, and configuration parameters used by
 * all narrow-phase algorithms (SAT, GJK, EPA, MPR, CCD).
 *
 * Design: Contact manifolds store up to 4 points. When more candidates arrive,
 * the manifold is reduced by selecting the 4 points that maximize contact area
 * (convex-hull of projections onto the contact plane). This produces stable
 * resting contacts even with iterative pair-wise detection.
 */

#pragma once

#include <pulse/math/math_common.h>
#include <pulse/math/vec3.h>
#include <pulse/math/transform.h>
#include <pulse/shapes/shape_common.h>

#include <cstdint>
#include <cstring>

namespace pulse {

// ── Contact point ────────────────────────────────────────────────────────────

/**
 * @struct ContactPoint
 * @brief A single contact between two shapes.
 *
 * Stores world-space position, normal (pointing from B toward A),
 * penetration depth, and feature IDs for frame-to-frame contact caching.
 */
struct ContactPoint {
    Vec3  positionOnA;   ///< Contact point on shape A (world space).
    Vec3  positionOnB;   ///< Contact point on shape B (world space).
    Vec3  normal;        ///< Contact normal (world), pointing from B toward A.
    float penetration;   ///< Penetration depth (positive = overlapping).
    uint32_t featureIdA; ///< Feature identifier on A (for contact caching).
    uint32_t featureIdB; ///< Feature identifier on B (for contact caching).

    /// Default: zero everything.
    PULSE_FORCE_INLINE ContactPoint() noexcept
        : positionOnA(Vec3::zero()),
          positionOnB(Vec3::zero()),
          normal(Vec3::zero()),
          penetration(0.0f),
          featureIdA(0xFFFFFFFFu),
          featureIdB(0xFFFFFFFFu)
    {}

    /// Construct with explicit values.
    PULSE_FORCE_INLINE ContactPoint(Vec3 posA, Vec3 posB, Vec3 n, float depth) noexcept
        : positionOnA(posA),
          positionOnB(posB),
          normal(n),
          penetration(depth),
          featureIdA(0xFFFFFFFFu),
          featureIdB(0xFFFFFFFFu)
    {}
};

// ── Contact manifold ─────────────────────────────────────────────────────────

/**
 * @struct ContactManifold
 * @brief A collection of up to 4 contact points between two shapes.
 *
 * When more than 4 points are added, the manifold reduces to the 4 points
 * that maximize the contact area (best coverage of the contact patch).
 */
struct ContactManifold {
    static constexpr uint32_t MaxContacts = 4;

    ContactPoint points[MaxContacts]; ///< Contact point storage.
    uint32_t     numContacts;         ///< Current number of valid contacts.
    ShapeType    shapeTypeA;          ///< Shape type of body A.
    ShapeType    shapeTypeB;          ///< Shape type of body B.

    /// Default: empty manifold.
    PULSE_FORCE_INLINE ContactManifold() noexcept
        : numContacts(0),
          shapeTypeA(ShapeType::Sphere),
          shapeTypeB(ShapeType::Sphere)
    {}

    /// Clear all contacts.
    PULSE_FORCE_INLINE void clear() noexcept { numContacts = 0; }

    /// Add a contact point. If full, triggers reduction to keep the best 4.
    PULSE_FORCE_INLINE void addPoint(const ContactPoint& cp) noexcept {
        if (numContacts < MaxContacts) {
            points[numContacts++] = cp;
            return;
        }
        // Manifold is full — find the candidate set of 5, reduce to best 4.
        reduceWith(cp);
    }

    /// Add a contact from components.
    PULSE_FORCE_INLINE void addPoint(Vec3 posA, Vec3 posB, Vec3 normal, float depth) noexcept {
        addPoint(ContactPoint(posA, posB, normal, depth));
    }

    /// Get the deepest penetration across all contacts.
    [[nodiscard]] PULSE_FORCE_INLINE float getMaxPenetration() const noexcept {
        float maxPen = 0.0f;
        for (uint32_t i = 0; i < numContacts; ++i) {
            if (points[i].penetration > maxPen)
                maxPen = points[i].penetration;
        }
        return maxPen;
    }

    /// Get the average contact normal.
    [[nodiscard]] PULSE_FORCE_INLINE Vec3 getAverageNormal() const noexcept {
        if (numContacts == 0) return Vec3::zero();
        Vec3 sum = Vec3::zero();
        for (uint32_t i = 0; i < numContacts; ++i) {
            sum = sum + points[i].normal;
        }
        float len = sum.length();
        return len > math::Epsilon ? sum * (1.0f / len) : points[0].normal;
    }

private:
    /// Reduce from 5 candidates (current 4 + new) to the best 4 by maximizing
    /// the area of the contact patch projected onto the contact plane.
    PULSE_FORCE_INLINE void reduceWith(const ContactPoint& newPoint) noexcept {
        // Candidate set: points[0..3] + newPoint
        ContactPoint candidates[5];
        for (uint32_t i = 0; i < 4; ++i) candidates[i] = points[i];
        candidates[4] = newPoint;

        // Step 1: Keep the point with the deepest penetration.
        uint32_t deepest = 0;
        for (uint32_t i = 1; i < 5; ++i) {
            if (candidates[i].penetration > candidates[deepest].penetration)
                deepest = i;
        }

        // Step 2: Keep the point farthest from the deepest.
        uint32_t farthest = pickFarthest(candidates, 5, deepest);

        // Step 3: Keep the point farthest from the line (deepest → farthest).
        uint32_t third = pickFarthestFromLine(candidates, 5, deepest, farthest);

        // Step 4: Keep the point farthest from the triangle (deepest, farthest, third).
        uint32_t fourth = pickFarthestFromTriangle(candidates, 5, deepest, farthest, third);

        // Gather results
        ContactPoint result[4] = {
            candidates[deepest], candidates[farthest],
            candidates[third],   candidates[fourth]
        };
        for (uint32_t i = 0; i < 4; ++i) points[i] = result[i];
        numContacts = 4;
    }

    /// Find the index farthest from candidates[anchor].
    static PULSE_FORCE_INLINE uint32_t pickFarthest(
        const ContactPoint* c, uint32_t n, uint32_t anchor) noexcept
    {
        float bestDist = -1.0f;
        uint32_t best = 0;
        Vec3 p = c[anchor].positionOnA;
        for (uint32_t i = 0; i < n; ++i) {
            if (i == anchor) continue;
            float d = (c[i].positionOnA - p).lengthSq();
            if (d > bestDist) { bestDist = d; best = i; }
        }
        return best;
    }

    /// Find the index farthest from the line segment (a → b).
    static PULSE_FORCE_INLINE uint32_t pickFarthestFromLine(
        const ContactPoint* c, uint32_t n, uint32_t a, uint32_t b) noexcept
    {
        Vec3 lineDir = c[b].positionOnA - c[a].positionOnA;
        float lineLenSq = lineDir.lengthSq();
        float bestDist = -1.0f;
        uint32_t best = 0;
        for (uint32_t i = 0; i < n; ++i) {
            if (i == a || i == b) continue;
            Vec3 diff = c[i].positionOnA - c[a].positionOnA;
            // Project diff onto lineDir and compute perpendicular distance
            float t = (lineLenSq > math::Epsilon) ? diff.dot(lineDir) / lineLenSq : 0.0f;
            t = math::clamp(t, 0.0f, 1.0f);
            Vec3 closest = c[a].positionOnA + lineDir * t;
            float d = (c[i].positionOnA - closest).lengthSq();
            if (d > bestDist) { bestDist = d; best = i; }
        }
        return best;
    }

    /// Find the index farthest from the triangle (a, b, c_idx).
    static PULSE_FORCE_INLINE uint32_t pickFarthestFromTriangle(
        const ContactPoint* c, uint32_t n,
        uint32_t a, uint32_t b, uint32_t c_idx) noexcept
    {
        Vec3 triNormal = (c[b].positionOnA - c[a].positionOnA)
                         .cross(c[c_idx].positionOnA - c[a].positionOnA);
        float bestDist = -1.0f;
        uint32_t best = 0;
        for (uint32_t i = 0; i < n; ++i) {
            if (i == a || i == b || i == c_idx) continue;
            Vec3 diff = c[i].positionOnA - c[a].positionOnA;
            float d = math::fastAbs(diff.dot(triNormal));
            // Also consider in-plane distance for coplanar points
            float inPlaneDist = (diff - triNormal * (diff.dot(triNormal) /
                                 (triNormal.lengthSq() + math::Epsilon))).lengthSq();
            float score = d + inPlaneDist * 0.01f; // Slight in-plane bias
            if (score > bestDist) { bestDist = score; best = i; }
        }
        return best;
    }
};

// ── Narrow-phase configuration ───────────────────────────────────────────────

/**
 * @struct NarrowPhaseConfig
 * @brief Configuration parameters for narrow-phase algorithms.
 */
struct NarrowPhaseConfig {
    float    collisionTolerance; ///< Contact tolerance (default 0.005 m).
    uint32_t gjkMaxIterations;   ///< Max GJK iterations (default 64).
    uint32_t epaMaxIterations;   ///< Max EPA iterations (default 64).
    float    epaTolerance;       ///< EPA convergence tolerance (default 1e-4).
    float    ccdTolerance;       ///< CCD time-of-impact tolerance (default 1e-4).
    uint32_t ccdMaxIterations;   ///< Max CCD iterations (default 32).

    PULSE_FORCE_INLINE NarrowPhaseConfig() noexcept
        : collisionTolerance(0.005f),
          gjkMaxIterations(64),
          epaMaxIterations(64),
          epaTolerance(1.0e-4f),
          ccdTolerance(1.0e-4f),
          ccdMaxIterations(32)
    {}
};

// ── GJK support point ────────────────────────────────────────────────────────

/**
 * @struct SupportPoint
 * @brief A point on the Minkowski difference A ⊖ B.
 *
 * Stores the support points on A and B individually (needed for contact
 * generation) along with the Minkowski-space point (pointA - pointB).
 */
struct SupportPoint {
    Vec3 pointA; ///< Support point on shape A (world space).
    Vec3 pointB; ///< Support point on shape B (world space).
    Vec3 point;  ///< Minkowski-difference point = pointA - pointB.

    PULSE_FORCE_INLINE SupportPoint() noexcept
        : pointA(Vec3::zero()), pointB(Vec3::zero()), point(Vec3::zero())
    {}

    PULSE_FORCE_INLINE SupportPoint(Vec3 a, Vec3 b) noexcept
        : pointA(a), pointB(b), point(a - b)
    {}
};

// ── GJK simplex ──────────────────────────────────────────────────────────────

/**
 * @struct Simplex
 * @brief A simplex of 1-4 vertices for GJK iteration.
 *
 * Vertices are stored in order of addition. The simplex evolves by adding
 * new support points and removing vertices not contributing to the nearest
 * feature to the origin.
 */
struct Simplex {
    static constexpr uint32_t MaxVerts = 4;

    SupportPoint vertices[MaxVerts]; ///< Simplex vertices.
    uint32_t     size;               ///< Current number of vertices (1-4).

    PULSE_FORCE_INLINE Simplex() noexcept : size(0) {}

    /// Add a vertex to the simplex.
    PULSE_FORCE_INLINE void addVertex(const SupportPoint& sp) noexcept {
        if (size < MaxVerts) {
            vertices[size++] = sp;
        }
    }

    /// Reset to empty.
    PULSE_FORCE_INLINE void clear() noexcept { size = 0; }

    /// Set simplex to a single vertex.
    PULSE_FORCE_INLINE void set(const SupportPoint& a) noexcept {
        vertices[0] = a;
        size = 1;
    }

    /// Set simplex to two vertices (line segment).
    PULSE_FORCE_INLINE void set(const SupportPoint& a, const SupportPoint& b) noexcept {
        vertices[0] = a;
        vertices[1] = b;
        size = 2;
    }

    /// Set simplex to three vertices (triangle).
    PULSE_FORCE_INLINE void set(const SupportPoint& a, const SupportPoint& b,
                                const SupportPoint& c) noexcept {
        vertices[0] = a;
        vertices[1] = b;
        vertices[2] = c;
        size = 3;
    }

    /// Access by index.
    [[nodiscard]] PULSE_FORCE_INLINE const SupportPoint& operator[](uint32_t i) const noexcept {
        return vertices[i];
    }
    [[nodiscard]] PULSE_FORCE_INLINE SupportPoint& operator[](uint32_t i) noexcept {
        return vertices[i];
    }
};

// ── GJK result ───────────────────────────────────────────────────────────────

/**
 * @enum GjkStatus
 * @brief Result status of a GJK query.
 */
enum class GjkStatus : uint8_t {
    Separated,     ///< Shapes are separated — distance is valid.
    Overlapping,   ///< Shapes overlap — simplex passed to EPA.
    MaxIterations  ///< Failed to converge within max iterations.
};

/**
 * @struct GjkResult
 * @brief Output of a GJK closest-point query.
 */
struct GjkResult {
    Vec3      closestOnA; ///< Closest point on A (world space). Valid if Separated.
    Vec3      closestOnB; ///< Closest point on B (world space). Valid if Separated.
    float     distance;   ///< Closest distance (0 if overlapping).
    Simplex   simplex;    ///< Final simplex (passed to EPA on overlap).
    GjkStatus status;     ///< Result status.
    uint32_t  iterations; ///< Number of iterations used.

    PULSE_FORCE_INLINE GjkResult() noexcept
        : closestOnA(Vec3::zero()), closestOnB(Vec3::zero()),
          distance(0.0f), status(GjkStatus::Separated), iterations(0)
    {}
};

// ── EPA result ───────────────────────────────────────────────────────────────

/**
 * @struct EpaResult
 * @brief Output of an EPA penetration-depth query.
 */
struct EpaResult {
    Vec3  normal;      ///< Penetration normal (world, from B toward A).
    Vec3  pointOnA;    ///< Contact point on A (world space).
    Vec3  pointOnB;    ///< Contact point on B (world space).
    float penetration; ///< Penetration depth (positive).
    bool  converged;   ///< True if EPA converged within tolerance.

    PULSE_FORCE_INLINE EpaResult() noexcept
        : normal(Vec3::zero()), pointOnA(Vec3::zero()), pointOnB(Vec3::zero()),
          penetration(0.0f), converged(false)
    {}
};

// ── MPR result ───────────────────────────────────────────────────────────────

/**
 * @struct MprResult
 * @brief Output of an MPR query.
 */
struct MprResult {
    Vec3  normal;      ///< Penetration normal (world, from B toward A).
    Vec3  pointOnA;    ///< Contact point on A (world space).
    Vec3  pointOnB;    ///< Contact point on B (world space).
    float penetration; ///< Penetration depth (positive).
    bool  overlapping; ///< True if shapes overlap.

    PULSE_FORCE_INLINE MprResult() noexcept
        : normal(Vec3::zero()), pointOnA(Vec3::zero()), pointOnB(Vec3::zero()),
          penetration(0.0f), overlapping(false)
    {}
};

// ── CCD result ───────────────────────────────────────────────────────────────

/**
 * @struct CcdResult
 * @brief Output of a continuous collision detection query.
 */
struct CcdResult {
    float timeOfImpact; ///< Time of impact t ∈ [0, 1]. 1.0 if no impact.
    Vec3  point;        ///< Contact point at TOI (world space).
    Vec3  normal;       ///< Contact normal at TOI (world space).
    bool  hit;          ///< True if a collision was detected.

    PULSE_FORCE_INLINE CcdResult() noexcept
        : timeOfImpact(1.0f), point(Vec3::zero()), normal(Vec3::zero()), hit(false)
    {}
};

} // namespace pulse
