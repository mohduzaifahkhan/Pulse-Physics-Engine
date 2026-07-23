#include <pulse/narrowphase/collision_dispatch.h>
#include <pulse/narrowphase/ccd.h>
#include <cstdio>
using namespace pulse;
int main() {
    // Box-Box SAT test
    {
        Box a(1.0f, 1.0f, 1.0f), b(1.0f, 1.0f, 1.0f);
        Transform txA, txB(Vec3(1.5f, 0, 0));
        ContactManifold m;
        bool hit = collide(a, txA, b, txB, m);
        std::printf("Box-Box: hit=%d contacts=%u pen=%.4f\n", hit, m.numContacts,
                    m.numContacts > 0 ? m.points[0].penetration : -1.0f);
        for (uint32_t i = 0; i < m.numContacts; ++i) {
            std::printf("  contact[%u]: pen=%.4f normal=(%.3f,%.3f,%.3f)\n",
                        i, m.points[i].penetration,
                        m.points[i].normal.getX(), m.points[i].normal.getY(), m.points[i].normal.getZ());
        }
    }
    // GJK distance
    {
        Sphere a(1.0f), b(1.0f);
        Transform txA(Vec3(0,0,0)), txB(Vec3(4.0f, 0, 0));
        GjkResult result;
        gjkQuery(a, txA, b, txB, result);
        std::printf("GJK sphere dist: status=%d distance=%.4f iters=%u\n",
                    (int)result.status, result.distance, result.iterations);
        std::printf("  closestA=(%.3f,%.3f,%.3f) closestB=(%.3f,%.3f,%.3f)\n",
                    result.closestOnA.getX(), result.closestOnA.getY(), result.closestOnA.getZ(),
                    result.closestOnB.getX(), result.closestOnB.getY(), result.closestOnB.getZ());
    }
    {
        Box box(1.0f, 1.0f, 1.0f);
        Sphere sphere(0.5f);
        Transform txBox, txSphere(Vec3(3.0f, 0, 0));
        GjkResult result;
        gjkQuery(box, txBox, sphere, txSphere, result);
        std::printf("GJK box-sphere dist: status=%d distance=%.4f iters=%u\n",
                    (int)result.status, result.distance, result.iterations);
    }
    // CCD linear
    {
        Sphere a(0.5f), b(0.5f);
        Transform txA_start(Vec3(-5, 0, 0)), txA_end(Vec3(5, 0, 0));
        Transform txB_start(Vec3(2, 0, 0)), txB_end(Vec3(2, 0, 0));
        CcdResult result;
        bool hit = ccdQuery(a, txA_start, txA_end, b, txB_start, txB_end, result);
        std::printf("CCD linear: hit=%d toi=%.6f\n", hit, result.timeOfImpact);
    }
    // CCD tunneling
    {
        Sphere bullet(0.1f);
        Box wall(0.05f, 1.0f, 1.0f);
        Transform txA_start(Vec3(-10, 0, 0)), txA_end(Vec3(10, 0, 0));
        Transform txB_start(Vec3(0, 0, 0)), txB_end(Vec3(0, 0, 0));
        CcdResult result;
        bool hit = ccdQuery(bullet, txA_start, txA_end, wall, txB_start, txB_end, result);
        std::printf("CCD tunnel: hit=%d toi=%.6f\n", hit, result.timeOfImpact);
    }
    return 0;
}
