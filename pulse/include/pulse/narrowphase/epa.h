/**
 * @file epa.h
 * @brief Expanding Polytope Algorithm (EPA) — penetration depth from GJK simplex.
 *
 * When GJK reports that two shapes overlap, the GJK simplex (a tetrahedron
 * enclosing the origin in Minkowski space) is passed to EPA. EPA iteratively
 * expands this polytope by finding the face closest to the origin, computing
 * a new support point in that direction, and re-triangulating the visible
 * horizon. The process converges to the minimum penetration vector.
 *
 * Fixed-capacity polytope (128 faces, 64 vertices) to avoid heap allocation.
 * This is sufficient for game physics; extremely complex concave interactions
 * are not expected as all shapes are convex.
 */

#pragma once

#include "narrowphase_common.h"
#include <pulse/shapes/sphere.h>
#include <pulse/shapes/box.h>
#include <pulse/shapes/capsule.h>
#include <pulse/shapes/cylinder.h>
#include <pulse/shapes/convex_hull.h>

namespace pulse {

namespace epa_detail {

    static constexpr uint32_t MaxFaces    = 128;
    static constexpr uint32_t MaxVertices = 64;
    static constexpr uint32_t MaxEdges    = 128;

    /// A triangle face on the polytope.
    struct Face {
        uint32_t v[3];   ///< Vertex indices.
        Vec3     normal; ///< Outward-facing normal.
        float    dist;   ///< Distance from origin to the face plane.
        bool     alive;  ///< False if removed during expansion.

        PULSE_FORCE_INLINE Face() noexcept : dist(0.0f), alive(false) {
            v[0] = v[1] = v[2] = 0;
        }
    };

    /// An edge on the horizon (used during polytope expansion).
    struct HorizonEdge {
        uint32_t a, b; ///< Vertex indices of the edge.
    };

    /// Compute the normal and distance of a triangle face.
    static PULSE_FORCE_INLINE void computeFaceNormal(
        Face& face, const SupportPoint* verts) noexcept
    {
        Vec3 a = verts[face.v[0]].point;
        Vec3 b = verts[face.v[1]].point;
        Vec3 c = verts[face.v[2]].point;

        Vec3 ab = b - a;
        Vec3 ac = c - a;
        Vec3 n = ab.cross(ac);
        float len = n.length();

        if (len > math::Epsilon) {
            n = n * (1.0f / len);
        } else {
            // Degenerate triangle — use an arbitrary normal
            n = Vec3(1.0f, 0.0f, 0.0f);
        }

        // Ensure normal points away from origin
        if (n.dot(a) < 0.0f) {
            n = -n;
            // Swap winding
            uint32_t tmp = face.v[1];
            face.v[1] = face.v[2];
            face.v[2] = tmp;
        }

        face.normal = n;
        face.dist = n.dot(a);
    }

    /// Find the face closest to the origin.
    static PULSE_FORCE_INLINE uint32_t findClosestFace(
        const Face* faces, uint32_t faceCount) noexcept
    {
        float minDist = math::Infinity;
        uint32_t closest = 0;
        for (uint32_t i = 0; i < faceCount; ++i) {
            if (!faces[i].alive) continue;
            if (faces[i].dist < minDist) {
                minDist = faces[i].dist;
                closest = i;
            }
        }
        return closest;
    }

    /// Check if an edge already exists in the horizon edge list (for dedup).
    static PULSE_FORCE_INLINE bool hasEdge(
        const HorizonEdge* edges, uint32_t count,
        uint32_t a, uint32_t b) noexcept
    {
        for (uint32_t i = 0; i < count; ++i) {
            if ((edges[i].a == a && edges[i].b == b) ||
                (edges[i].a == b && edges[i].b == a))
                return true;
        }
        return false;
    }

} // namespace epa_detail

// ── EPA query ────────────────────────────────────────────────────────────────

/**
 * @brief Run the EPA algorithm to find penetration depth and contact info.
 *
 * @tparam ShapeA  Shape type with supportWorld(Vec3 dir, const Transform&).
 * @tparam ShapeB  Shape type with supportWorld(Vec3 dir, const Transform&).
 * @param shapeA   First shape.
 * @param txA      Transform of shape A.
 * @param shapeB   Second shape.
 * @param txB      Transform of shape B.
 * @param gjkSimplex  The GJK simplex (must be a tetrahedron, size == 4).
 * @param result   Output: penetration normal, depth, contact points.
 * @param maxIter  Maximum iterations (default 64).
 * @param tolerance  Convergence tolerance (default 1e-4).
 * @return True if EPA converged successfully.
 */
template <typename ShapeA, typename ShapeB>
static inline bool epaQuery(
    const ShapeA& shapeA, const Transform& txA,
    const ShapeB& shapeB, const Transform& txB,
    const Simplex& gjkSimplex,
    EpaResult& result,
    uint32_t maxIter = 64,
    float tolerance = 1.0e-4f) noexcept
{
    using namespace epa_detail;

    result = EpaResult();

    if (gjkSimplex.size < 4) {
        // Need a tetrahedron to start EPA.
        // If GJK gave us less, produce a degenerate result from the simplex.
        if (gjkSimplex.size >= 2) {
            Vec3 diff = gjkSimplex[0].point - gjkSimplex[1].point;
            float len = diff.length();
            if (len > math::Epsilon) {
                result.normal = diff * (1.0f / len);
            } else {
                result.normal = Vec3(1.0f, 0.0f, 0.0f);
            }
            result.penetration = math::Epsilon;
            result.pointOnA = gjkSimplex[0].pointA;
            result.pointOnB = gjkSimplex[0].pointB;
            result.converged = false;
        }
        return false;
    }

    // ── Initialize polytope from GJK tetrahedron ──

    SupportPoint verts[MaxVertices];
    Face faces[MaxFaces];
    uint32_t vertCount = 4;
    uint32_t faceCount = 0;

    for (uint32_t i = 0; i < 4; ++i) {
        verts[i] = gjkSimplex[i];
    }

    // Create 4 triangular faces of the tetrahedron
    // Winding: faces point outward from the centroid
    uint32_t faceIndices[4][3] = {
        {0, 1, 2},
        {0, 3, 1},
        {0, 2, 3},
        {1, 3, 2}
    };

    for (int f = 0; f < 4; ++f) {
        faces[faceCount].v[0] = faceIndices[f][0];
        faces[faceCount].v[1] = faceIndices[f][1];
        faces[faceCount].v[2] = faceIndices[f][2];
        faces[faceCount].alive = true;
        computeFaceNormal(faces[faceCount], verts);
        faceCount++;
    }

    // ── Main EPA loop ──

    for (uint32_t iter = 0; iter < maxIter; ++iter) {
        // Find the face closest to the origin
        uint32_t closestIdx = findClosestFace(faces, faceCount);
        const Face& closest = faces[closestIdx];

        // Get new support point in the direction of the closest face's normal
        Vec3 searchDir = closest.normal;
        Vec3 pA = shapeA.supportWorld(searchDir, txA);
        Vec3 pB = shapeB.supportWorld(-searchDir, txB);
        SupportPoint newSP(pA, pB);

        // Check convergence: is the new point significantly beyond the face?
        float newDist = newSP.point.dot(searchDir);
        if (newDist - closest.dist < tolerance) {
            // Converged — extract result
            result.normal = closest.normal;
            result.penetration = closest.dist;

            // Compute contact point via barycentric coords on the closest face
            Vec3 a = verts[closest.v[0]].point;
            Vec3 b = verts[closest.v[1]].point;
            Vec3 c = verts[closest.v[2]].point;

            // Project origin onto the face and compute barycentric coords
            Vec3 v0 = b - a, v1 = c - a, v2 = -a; // v2 = origin - a
            float d00 = v0.dot(v0);
            float d01 = v0.dot(v1);
            float d11 = v1.dot(v1);
            float d20 = v2.dot(v0);
            float d21 = v2.dot(v1);
            float denom = d00 * d11 - d01 * d01;

            float baryV = (d11 * d20 - d01 * d21) / (denom + math::Epsilon);
            float baryW = (d00 * d21 - d01 * d20) / (denom + math::Epsilon);
            float baryU = 1.0f - baryV - baryW;

            // Clamp barycentric coords
            baryU = math::clamp(baryU, 0.0f, 1.0f);
            baryV = math::clamp(baryV, 0.0f, 1.0f);
            baryW = math::clamp(baryW, 0.0f, 1.0f);
            float sum = baryU + baryV + baryW;
            if (sum > math::Epsilon) {
                baryU /= sum; baryV /= sum; baryW /= sum;
            }

            result.pointOnA = verts[closest.v[0]].pointA * baryU +
                              verts[closest.v[1]].pointA * baryV +
                              verts[closest.v[2]].pointA * baryW;
            result.pointOnB = verts[closest.v[0]].pointB * baryU +
                              verts[closest.v[1]].pointB * baryV +
                              verts[closest.v[2]].pointB * baryW;

            result.converged = true;
            return true;
        }

        // ── Expand polytope: remove faces visible from the new point ──

        if (vertCount >= MaxVertices) break; // Capacity exhausted

        uint32_t newVertIdx = vertCount;
        verts[vertCount++] = newSP;

        // Collect horizon edges
        HorizonEdge horizonEdges[MaxEdges];
        uint32_t horizonCount = 0;

        for (uint32_t f = 0; f < faceCount; ++f) {
            if (!faces[f].alive) continue;

            // Check if this face is visible from the new point
            Vec3 faceVert = verts[faces[f].v[0]].point;
            if (faces[f].normal.dot(newSP.point - faceVert) > 0.0f) {
                // Face is visible — mark for removal
                faces[f].alive = false;

                // Add edges to the horizon (edges shared by visible and non-visible faces)
                for (int e = 0; e < 3; ++e) {
                    uint32_t ea = faces[f].v[e];
                    uint32_t eb = faces[f].v[(e + 1) % 3];

                    if (horizonCount < MaxEdges) {
                        // Check if reverse edge exists — if so, it's an internal edge, remove both
                        bool found = false;
                        for (uint32_t h = 0; h < horizonCount; ++h) {
                            if (horizonEdges[h].a == eb && horizonEdges[h].b == ea) {
                                // Internal edge — remove it
                                horizonEdges[h] = horizonEdges[horizonCount - 1];
                                horizonCount--;
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            horizonEdges[horizonCount].a = ea;
                            horizonEdges[horizonCount].b = eb;
                            horizonCount++;
                        }
                    }
                }
            }
        }

        // Create new faces from horizon edges to the new vertex
        for (uint32_t h = 0; h < horizonCount && faceCount < MaxFaces; ++h) {
            Face& newFace = faces[faceCount];
            newFace.v[0] = horizonEdges[h].a;
            newFace.v[1] = horizonEdges[h].b;
            newFace.v[2] = newVertIdx;
            newFace.alive = true;
            computeFaceNormal(newFace, verts);
            faceCount++;
        }
    }

    // Did not converge — return best result so far
    uint32_t closestIdx = findClosestFace(faces, faceCount);
    if (faces[closestIdx].alive) {
        const Face& closest = faces[closestIdx];
        result.normal = closest.normal;
        result.penetration = closest.dist;
        result.pointOnA = verts[closest.v[0]].pointA;
        result.pointOnB = verts[closest.v[0]].pointB;
    }
    result.converged = false;
    return false;
}

} // namespace pulse
