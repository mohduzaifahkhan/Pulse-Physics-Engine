/**
 * @file sat.h
 * @brief Separating Axis Theorem (SAT) — fast Box-vs-Box collision detection.
 *
 * Implements the 15-axis SAT test for oriented boxes: 3 face normals from A,
 * 3 face normals from B, and 9 edge cross products. Returns the minimum
 * penetration axis and depth, then generates contact points by clipping
 * the incident face against the reference face (Sutherland-Hodgman).
 *
 * This is the primary algorithm for Box-Box collisions due to its speed
 * and ability to produce high-quality contact manifolds.
 */

#pragma once

#include "narrowphase_common.h"
#include <pulse/shapes/box.h>
#include <pulse/math/mat3.h>

namespace pulse {

// ── SAT internal helpers ─────────────────────────────────────────────────────

namespace sat_detail {

    /// Project a box onto an axis and return the half-extent of the projection.
    [[nodiscard]] PULSE_FORCE_INLINE float projectBoxOnAxis(
        const Vec3& halfExtents, const Vec3& axisInLocal) noexcept
    {
        return math::fastAbs(halfExtents.getX() * axisInLocal.getX()) +
               math::fastAbs(halfExtents.getY() * axisInLocal.getY()) +
               math::fastAbs(halfExtents.getZ() * axisInLocal.getZ());
    }

    /// Clip a polygon (vertices stored in `input`) against a plane defined by
    /// (normal · point ≤ offset). Output stored in `output`. Returns new count.
    /// Sutherland-Hodgman single-plane clip.
    static PULSE_FORCE_INLINE uint32_t clipPolygonAgainstPlane(
        const Vec3* input, uint32_t inputCount,
        Vec3* output, Vec3 planeNormal, float planeOffset) noexcept
    {
        if (inputCount == 0) return 0;

        uint32_t outCount = 0;
        constexpr uint32_t MaxClipVerts = 8;

        Vec3 prev = input[inputCount - 1];
        float prevDist = prev.dot(planeNormal) - planeOffset;

        for (uint32_t i = 0; i < inputCount && outCount < MaxClipVerts; ++i) {
            Vec3 curr = input[i];
            float currDist = curr.dot(planeNormal) - planeOffset;

            if (prevDist <= 0.0f) {
                // Previous vertex is inside
                if (currDist <= 0.0f) {
                    // Both inside — emit current
                    output[outCount++] = curr;
                } else {
                    // Exiting — emit intersection
                    float t = prevDist / (prevDist - currDist);
                    output[outCount++] = prev + (curr - prev) * t;
                }
            } else {
                // Previous vertex is outside
                if (currDist <= 0.0f) {
                    // Entering — emit intersection, then current
                    float t = prevDist / (prevDist - currDist);
                    if (outCount < MaxClipVerts)
                        output[outCount++] = prev + (curr - prev) * t;
                    if (outCount < MaxClipVerts)
                        output[outCount++] = curr;
                }
                // Both outside — emit nothing
            }

            prev = curr;
            prevDist = currDist;
        }

        return outCount;
    }

    /// Get the 4 vertices of a box face. faceIndex: 0-5 (±X, ±Y, ±Z).
    /// Returns outward face normal.
    static PULSE_FORCE_INLINE Vec3 getBoxFaceVertices(
        const Vec3& halfExtents, uint32_t faceIndex, Vec3* verts) noexcept
    {
        float hx = halfExtents.getX();
        float hy = halfExtents.getY();
        float hz = halfExtents.getZ();

        switch (faceIndex) {
        case 0: // +X face
            verts[0] = Vec3( hx, -hy, -hz);
            verts[1] = Vec3( hx,  hy, -hz);
            verts[2] = Vec3( hx,  hy,  hz);
            verts[3] = Vec3( hx, -hy,  hz);
            return Vec3(1.0f, 0.0f, 0.0f);
        case 1: // -X face
            verts[0] = Vec3(-hx, -hy,  hz);
            verts[1] = Vec3(-hx,  hy,  hz);
            verts[2] = Vec3(-hx,  hy, -hz);
            verts[3] = Vec3(-hx, -hy, -hz);
            return Vec3(-1.0f, 0.0f, 0.0f);
        case 2: // +Y face
            verts[0] = Vec3(-hx,  hy, -hz);
            verts[1] = Vec3( hx,  hy, -hz);
            verts[2] = Vec3( hx,  hy,  hz);
            verts[3] = Vec3(-hx,  hy,  hz);
            return Vec3(0.0f, 1.0f, 0.0f);
        case 3: // -Y face
            verts[0] = Vec3(-hx, -hy,  hz);
            verts[1] = Vec3( hx, -hy,  hz);
            verts[2] = Vec3( hx, -hy, -hz);
            verts[3] = Vec3(-hx, -hy, -hz);
            return Vec3(0.0f, -1.0f, 0.0f);
        case 4: // +Z face
            verts[0] = Vec3(-hx, -hy,  hz);
            verts[1] = Vec3(-hx,  hy,  hz);
            verts[2] = Vec3( hx,  hy,  hz);
            verts[3] = Vec3( hx, -hy,  hz);
            return Vec3(0.0f, 0.0f, 1.0f);
        case 5: // -Z face
        default:
            verts[0] = Vec3( hx, -hy, -hz);
            verts[1] = Vec3( hx,  hy, -hz);
            verts[2] = Vec3(-hx,  hy, -hz);
            verts[3] = Vec3(-hx, -hy, -hz);
            return Vec3(0.0f, 0.0f, -1.0f);
        }
    }

} // namespace sat_detail

// ── SAT Box-Box collision ────────────────────────────────────────────────────

/**
 * @brief Test two oriented boxes for overlap using the 15-axis SAT.
 *
 * If overlapping, generates contact points by clipping the incident face
 * against the reference face. Adds results to the manifold.
 *
 * @param boxA   First box shape.
 * @param txA    World transform of box A.
 * @param boxB   Second box shape.
 * @param txB    World transform of box B.
 * @param manifold  Output contact manifold (contacts are appended).
 * @return True if the boxes overlap.
 */
[[nodiscard]] static inline bool satTestBoxBox(
    const Box& boxA, const Transform& txA,
    const Box& boxB, const Transform& txB,
    ContactManifold& manifold) noexcept
{
    // Get rotation matrices
    Mat3 rotA = txA.toMat3();
    Mat3 rotB = txB.toMat3();

    // Translation vector from A to B in world space
    Vec3 t = txB.position - txA.position;

    // Express t in A's local frame
    Vec3 tInA(t.dot(rotA[0]), t.dot(rotA[1]), t.dot(rotA[2]));

    // Rotation matrix expressing B in A's coordinate frame: C = A^T * B
    // C[i][j] = rotA[i] · rotB[j]
    float C[3][3];
    float absC[3][3]; // Absolute values with epsilon for edge-edge degeneracy
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            C[i][j] = rotA[i].dot(rotB[j]);
            absC[i][j] = math::fastAbs(C[i][j]) + math::Epsilon;
        }
    }

    float heA[3] = { boxA.halfExtents.getX(), boxA.halfExtents.getY(), boxA.halfExtents.getZ() };
    float heB[3] = { boxB.halfExtents.getX(), boxB.halfExtents.getY(), boxB.halfExtents.getZ() };
    float tA[3] = { tInA.getX(), tInA.getY(), tInA.getZ() };

    float minPenetration = math::Infinity;
    int bestAxis = -1;

    // Helper lambda: test one separating axis
    auto testAxis = [&](float projCenter, float projRadiusA, float projRadiusB, int axisId) -> bool {
        float separation = math::fastAbs(projCenter) - projRadiusA - projRadiusB;
        if (separation > 0.0f) return false; // Separated on this axis
        float pen = -separation;
        if (pen < minPenetration) {
            minPenetration = pen;
            bestAxis = axisId;
        }
        return true;
    };

    // ── Face axes of A (3 axes) ──
    // Axis A0: A's local X
    if (!testAxis(tA[0],
                  heA[0],
                  heB[0] * absC[0][0] + heB[1] * absC[0][1] + heB[2] * absC[0][2], 0))
        return false;

    // Axis A1: A's local Y
    if (!testAxis(tA[1],
                  heA[1],
                  heB[0] * absC[1][0] + heB[1] * absC[1][1] + heB[2] * absC[1][2], 1))
        return false;

    // Axis A2: A's local Z
    if (!testAxis(tA[2],
                  heA[2],
                  heB[0] * absC[2][0] + heB[1] * absC[2][1] + heB[2] * absC[2][2], 2))
        return false;

    // ── Face axes of B (3 axes) ──
    // Axis B0: B's local X (expressed in A's frame as column 0 of C)
    if (!testAxis(tA[0] * C[0][0] + tA[1] * C[1][0] + tA[2] * C[2][0],
                  heA[0] * absC[0][0] + heA[1] * absC[1][0] + heA[2] * absC[2][0],
                  heB[0], 3))
        return false;

    // Axis B1
    if (!testAxis(tA[0] * C[0][1] + tA[1] * C[1][1] + tA[2] * C[2][1],
                  heA[0] * absC[0][1] + heA[1] * absC[1][1] + heA[2] * absC[2][1],
                  heB[1], 4))
        return false;

    // Axis B2
    if (!testAxis(tA[0] * C[0][2] + tA[1] * C[1][2] + tA[2] * C[2][2],
                  heA[0] * absC[0][2] + heA[1] * absC[1][2] + heA[2] * absC[2][2],
                  heB[2], 5))
        return false;

    // ── Edge cross-product axes (9 axes) ──
    // Axis A0 × B0
    if (!testAxis(tA[2] * C[1][0] - tA[1] * C[2][0],
                  heA[1] * absC[2][0] + heA[2] * absC[1][0],
                  heB[1] * absC[0][2] + heB[2] * absC[0][1], 6))
        return false;

    // Axis A0 × B1
    if (!testAxis(tA[2] * C[1][1] - tA[1] * C[2][1],
                  heA[1] * absC[2][1] + heA[2] * absC[1][1],
                  heB[0] * absC[0][2] + heB[2] * absC[0][0], 7))
        return false;

    // Axis A0 × B2
    if (!testAxis(tA[2] * C[1][2] - tA[1] * C[2][2],
                  heA[1] * absC[2][2] + heA[2] * absC[1][2],
                  heB[0] * absC[0][1] + heB[1] * absC[0][0], 8))
        return false;

    // Axis A1 × B0
    if (!testAxis(tA[0] * C[2][0] - tA[2] * C[0][0],
                  heA[0] * absC[2][0] + heA[2] * absC[0][0],
                  heB[1] * absC[1][2] + heB[2] * absC[1][1], 9))
        return false;

    // Axis A1 × B1
    if (!testAxis(tA[0] * C[2][1] - tA[2] * C[0][1],
                  heA[0] * absC[2][1] + heA[2] * absC[0][1],
                  heB[0] * absC[1][2] + heB[2] * absC[1][0], 10))
        return false;

    // Axis A1 × B2
    if (!testAxis(tA[0] * C[2][2] - tA[2] * C[0][2],
                  heA[0] * absC[2][2] + heA[2] * absC[0][2],
                  heB[0] * absC[1][1] + heB[1] * absC[1][0], 11))
        return false;

    // Axis A2 × B0
    if (!testAxis(tA[1] * C[0][0] - tA[0] * C[1][0],
                  heA[0] * absC[1][0] + heA[1] * absC[0][0],
                  heB[1] * absC[2][2] + heB[2] * absC[2][1], 12))
        return false;

    // Axis A2 × B1
    if (!testAxis(tA[1] * C[0][1] - tA[0] * C[1][1],
                  heA[0] * absC[1][1] + heA[1] * absC[0][1],
                  heB[0] * absC[2][2] + heB[2] * absC[2][0], 13))
        return false;

    // Axis A2 × B2
    if (!testAxis(tA[1] * C[0][2] - tA[0] * C[1][2],
                  heA[0] * absC[1][2] + heA[1] * absC[0][2],
                  heB[0] * absC[2][1] + heB[1] * absC[2][0], 14))
        return false;

    // ── All 15 axes failed to separate → boxes overlap ──

    // Determine the separating axis normal in world space
    Vec3 normal;
    bool isEdgeContact = (bestAxis >= 6);

    if (bestAxis < 3) {
        // Face of A
        normal = rotA[bestAxis];
        if (normal.dot(t) < 0.0f) normal = -normal;
    } else if (bestAxis < 6) {
        // Face of B
        normal = rotB[bestAxis - 3];
        if (normal.dot(t) < 0.0f) normal = -normal;
    } else {
        // Edge-edge: compute cross product of the two edges
        int edgeA = (bestAxis - 6) / 3;
        int edgeB = (bestAxis - 6) % 3;
        normal = rotA[edgeA].cross(rotB[edgeB]);
        float len = normal.length();
        if (len < math::Epsilon) {
            // Degenerate — edges are parallel. Fall back to face axis.
            normal = rotA[0];
            if (normal.dot(t) < 0.0f) normal = -normal;
        } else {
            normal = normal * (1.0f / len);
            if (normal.dot(t) < 0.0f) normal = -normal;
        }
    }

    // ── Generate contact points ──

    if (isEdgeContact) {
        // Edge-edge: single contact point at the closest point between edges
        int edgeA = (bestAxis - 6) / 3;
        int edgeB = (bestAxis - 6) % 3;

        // Edge midpoints in local space → world
        Vec3 pA = txA.position;
        Vec3 pB = txB.position;

        // Move to the correct edge on each box
        for (int i = 0; i < 3; ++i) {
            if (i == edgeA) continue;
            float s = (rotA[i].dot(normal) > 0.0f) ? -heA[i] : heA[i];
            pA = pA + rotA[i] * s;
        }
        for (int i = 0; i < 3; ++i) {
            if (i == edgeB) continue;
            float s = (rotB[i].dot(normal) > 0.0f) ? heB[i] : -heB[i];
            pB = pB + rotB[i] * s;
        }

        // Closest points between the two edge lines
        Vec3 dA = rotA[edgeA];
        Vec3 dB = rotB[edgeB];
        Vec3 r = pA - pB;

        float a = dA.dot(dA);
        float e = dB.dot(dB);
        float f = dB.dot(r);
        float b = dA.dot(dB);
        float c = dA.dot(r);
        float denom = a * e - b * b;

        float sParam = 0.0f, tParam = 0.0f;
        if (math::fastAbs(denom) > math::Epsilon) {
            sParam = math::clamp((b * f - c * e) / denom, -heA[edgeA], heA[edgeA]);
            tParam = math::clamp((a * f - b * c) / denom, -heB[edgeB], heB[edgeB]);
        }

        Vec3 contactA = pA + dA * sParam;
        Vec3 contactB = pB + dB * tParam;
        Vec3 contactMid = (contactA + contactB) * 0.5f;

        manifold.addPoint(contactA, contactB, normal, minPenetration);
    } else {
        // Face contact: clip the incident face against the reference face

        // Determine reference and incident faces
        uint32_t refFaceIdx;
        const Box* refBox;
        const Transform* refTx;
        const Box* incBox;
        const Transform* incTx;
        Mat3 refRot, incRot;
        bool flip = false;

        if (bestAxis < 3) {
            refBox = &boxA; refTx = &txA; refRot = rotA;
            incBox = &boxB; incTx = &txB; incRot = rotB;
            refFaceIdx = (normal.dot(rotA[bestAxis]) > 0.0f) ? bestAxis * 2 : bestAxis * 2 + 1;
        } else {
            refBox = &boxB; refTx = &txB; refRot = rotB;
            incBox = &boxA; incTx = &txA; incRot = rotA;
            refFaceIdx = ((bestAxis - 3) * 2);
            if (normal.dot(rotB[bestAxis - 3]) < 0.0f) refFaceIdx++;
            flip = true;
        }

        // Get reference face vertices and normal in world space
        Vec3 refFaceVerts[4];
        Vec3 refFaceLocalNormal = sat_detail::getBoxFaceVertices(
            refBox->halfExtents, refFaceIdx, refFaceVerts);

        // Transform ref face vertices to world space
        for (int i = 0; i < 4; ++i) {
            refFaceVerts[i] = refTx->transformPoint(refFaceVerts[i]);
        }
        Vec3 refNormalWorld = refTx->transformDirection(refFaceLocalNormal);

        // Find incident face: the face on incBox most anti-parallel to refNormalWorld
        Vec3 incNormalLocal = incTx->inverseTransformDirection(-refNormalWorld);
        float bestDot = -math::Infinity;
        uint32_t incFaceIdx = 0;
        for (uint32_t f = 0; f < 6; ++f) {
            Vec3 dummyVerts[4];
            Vec3 fn = sat_detail::getBoxFaceVertices(incBox->halfExtents, f, dummyVerts);
            float d = fn.dot(incNormalLocal);
            if (d > bestDot) {
                bestDot = d;
                incFaceIdx = f;
            }
        }

        // Get incident face vertices in world space
        Vec3 incFaceVerts[4];
        sat_detail::getBoxFaceVertices(incBox->halfExtents, incFaceIdx, incFaceVerts);
        for (int i = 0; i < 4; ++i) {
            incFaceVerts[i] = incTx->transformPoint(incFaceVerts[i]);
        }

        // Clip incident face against the 4 side planes of the reference face
        Vec3 buf1[8], buf2[8];
        for (int i = 0; i < 4; ++i) buf1[i] = incFaceVerts[i];
        uint32_t count = 4;

        for (int i = 0; i < 4; ++i) {
            Vec3 edgeStart = refFaceVerts[i];
            Vec3 edgeEnd = refFaceVerts[(i + 1) % 4];
            Vec3 edgeDir = edgeEnd - edgeStart;
            Vec3 clipNormal = edgeDir.cross(refNormalWorld);
            float clipLen = clipNormal.length();
            if (clipLen < math::Epsilon) continue;
            clipNormal = clipNormal * (1.0f / clipLen);
            float clipOffset = clipNormal.dot(edgeStart);

            Vec3* src = (i % 2 == 0) ? buf1 : buf2;
            Vec3* dst = (i % 2 == 0) ? buf2 : buf1;
            count = sat_detail::clipPolygonAgainstPlane(src, count, dst, clipNormal, clipOffset);
        }

        // The last clip writes to dst. For i=3 (odd), dst=buf1.
        Vec3* clipped = buf1;

        // Project clipped points onto the reference face plane and generate contacts
        float refPlaneOffset = refNormalWorld.dot(refFaceVerts[0]);

        for (uint32_t i = 0; i < count && manifold.numContacts < ContactManifold::MaxContacts; ++i) {
            float dist = clipped[i].dot(refNormalWorld) - refPlaneOffset;
            if (dist <= math::BigEpsilon) {
                // Contact point is at or below the reference face
                Vec3 contactOnRef = clipped[i] - refNormalWorld * dist;
                Vec3 contactOnInc = clipped[i];

                Vec3 pointA = flip ? contactOnInc : contactOnRef;
                Vec3 pointB = flip ? contactOnRef : contactOnInc;

                // Use the SAT-computed penetration depth, not the clip distance,
                // since face-face contacts have near-zero clip distance but
                // the actual penetration is the SAT minimum overlap.
                float pen = math::fastMax(minPenetration, -dist);
                manifold.addPoint(pointA, pointB, normal, pen);
            }
        }
    }

    return manifold.numContacts > 0;
}

} // namespace pulse
