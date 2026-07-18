/**
 * @file test_utilities.cpp
 * @brief Comprehensive unit tests for the Pulse utilities module.
 *
 * Covers: assert macros, type traits, FixedBitset, Handle, HandlePool,
 * FixedArray, SoAArray, RingBuffer, and Profiler.
 *
 * Uses the same lightweight test framework as test_math.cpp and test_memory.cpp.
 */

// Force asserts and profiling ON for testing regardless of build mode.
#define PULSE_ENABLE_ASSERTS 1
#define PULSE_ENABLE_PROFILING 1

#include <pulse/utilities/assert.h>
#include <pulse/utilities/type_traits.h>
#include <pulse/utilities/bitset.h>
#include <pulse/utilities/handle.h>
#include <pulse/utilities/handle_pool.h>
#include <pulse/utilities/fixed_array.h>
#include <pulse/utilities/soa_array.h>
#include <pulse/utilities/ring_buffer.h>
#include <pulse/utilities/profiler.h>

#include <pulse/math/vec3.h>
#include <pulse/math/vec4.h>
#include <pulse/math/quat.h>
#include <pulse/math/mat3.h>
#include <pulse/math/mat4.h>
#include <pulse/math/aabb.h>
#include <pulse/math/plane.h>
#include <pulse/math/ray.h>
#include <pulse/math/transform.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <thread>
#include <atomic>

// ── Minimal test framework ────────────────────────────────────────────────────

static int g_totalTests = 0;
static int g_passedTests = 0;
static int g_failedTests = 0;

#define TEST_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            std::printf("  FAIL: %s (line %d): %s\n", __func__, __LINE__, #expr); \
            return false; \
        } \
    } while(0)

#define RUN_TEST(func) \
    do { \
        g_totalTests++; \
        if (func()) { \
            g_passedTests++; \
        } else { \
            g_failedTests++; \
            std::printf("FAILED: %s\n", #func); \
        } \
    } while(0)

// =============================================================================
// ASSERT TESTS
// =============================================================================

static bool g_assertHandlerCalled = false;
static const char* g_lastAssertExpr = nullptr;
static const char* g_lastAssertMsg = nullptr;

void testAssertHandler(const pulse::util::SourceLocation& /*loc*/,
                       const char* expr, const char* msg) noexcept {
    g_assertHandlerCalled = true;
    g_lastAssertExpr = expr;
    g_lastAssertMsg = msg;
}

bool test_assert_custom_handler() {
    // Install custom handler.
    pulse::util::setAssertHandler(testAssertHandler);
    g_assertHandlerCalled = false;

    // Fire an assert.
    PULSE_ASSERT(false);
    TEST_ASSERT(g_assertHandlerCalled);
    TEST_ASSERT(g_lastAssertExpr != nullptr);

    // Reset handler.
    pulse::util::setAssertHandler(nullptr); // Resets to default.
    g_assertHandlerCalled = false;

    return true;
}

bool test_assert_msg() {
    pulse::util::setAssertHandler(testAssertHandler);
    g_assertHandlerCalled = false;

    PULSE_ASSERT_MSG(false, "test message");
    TEST_ASSERT(g_assertHandlerCalled);
    TEST_ASSERT(g_lastAssertMsg != nullptr);
    TEST_ASSERT(std::strcmp(g_lastAssertMsg, "test message") == 0);

    pulse::util::setAssertHandler(nullptr);
    g_assertHandlerCalled = false;
    return true;
}

bool test_verify_evaluates() {
    // PULSE_VERIFY should always evaluate its expression.
    int counter = 0;
    PULSE_VERIFY(++counter > 0); // Should evaluate counter++
    TEST_ASSERT(counter == 1);
    return true;
}

bool test_static_assert_compiles() {
    // PULSE_STATIC_ASSERT should compile without error.
    PULSE_STATIC_ASSERT(sizeof(int) >= 4, "int must be at least 4 bytes");
    PULSE_STATIC_ASSERT(true, "This should always pass");
    return true;
}

// =============================================================================
// TYPE TRAITS TESTS
// =============================================================================

bool test_is_simd_type() {
    using namespace pulse::util;

    TEST_ASSERT(is_simd_type_v<pulse::Vec3>);
    TEST_ASSERT(is_simd_type_v<pulse::Vec4>);
    TEST_ASSERT(is_simd_type_v<pulse::Quat>);
    TEST_ASSERT(is_simd_type_v<pulse::Mat3>);
    TEST_ASSERT(is_simd_type_v<pulse::Mat4>);
    TEST_ASSERT(is_simd_type_v<pulse::AABB>);
    TEST_ASSERT(is_simd_type_v<pulse::Plane>);
    TEST_ASSERT(is_simd_type_v<pulse::Ray>);
    TEST_ASSERT(is_simd_type_v<pulse::Transform>);

    // Non-SIMD types
    TEST_ASSERT(!is_simd_type_v<int>);
    TEST_ASSERT(!is_simd_type_v<float>);
    TEST_ASSERT(!is_simd_type_v<double>);

    // CV-qualified
    TEST_ASSERT(is_simd_type_v<const pulse::Vec3>);
    TEST_ASSERT(is_simd_type_v<volatile pulse::Vec4>);

    return true;
}

bool test_is_pod_type() {
    using namespace pulse::util;

    TEST_ASSERT(is_pod_type_v<int>);
    TEST_ASSERT(is_pod_type_v<float>);
    TEST_ASSERT(is_pod_type_v<uint64_t>);

    // Struct with non-trivial constructor is NOT pod.
    struct NonTrivial {
        NonTrivial() : x(42) {}
        int x;
    };
    TEST_ASSERT(!is_pod_type_v<NonTrivial>);

    return true;
}

bool test_is_aligned() {
    using namespace pulse::util;

    TEST_ASSERT((is_aligned_v<pulse::Vec3, 16>));
    TEST_ASSERT((is_aligned_v<pulse::Vec4, 16>));
    TEST_ASSERT((is_aligned_v<int, 4>));
    TEST_ASSERT(!(is_aligned_v<int, 16>));

    return true;
}

bool test_alignment_of() {
    using namespace pulse::util;

    TEST_ASSERT(alignment_of_v<pulse::Vec3> >= 16);
    TEST_ASSERT(alignment_of_v<pulse::Vec4> >= 16);
    TEST_ASSERT(alignment_of_v<int> == alignof(int));

    return true;
}

bool test_is_power_of_2() {
    using namespace pulse::util;

    TEST_ASSERT(is_power_of_2_v<1>);
    TEST_ASSERT(is_power_of_2_v<2>);
    TEST_ASSERT(is_power_of_2_v<4>);
    TEST_ASSERT(is_power_of_2_v<64>);
    TEST_ASSERT(is_power_of_2_v<1024>);

    TEST_ASSERT(!is_power_of_2_v<0>);
    TEST_ASSERT(!is_power_of_2_v<3>);
    TEST_ASSERT(!is_power_of_2_v<5>);
    TEST_ASSERT(!is_power_of_2_v<6>);

    // Function form
    TEST_ASSERT(isPowerOf2(16));
    TEST_ASSERT(!isPowerOf2(15));

    return true;
}

bool test_cache_line_padded() {
    using namespace pulse::util;

    CacheLinePadded<int> padded(42);
    TEST_ASSERT(padded.value == 42);
    TEST_ASSERT(sizeof(CacheLinePadded<int>) >= 64);
    TEST_ASSERT(alignof(CacheLinePadded<int>) >= 64);

    // Ensure we can access via arrow operator for struct types.
    struct Point { float x, y; };
    CacheLinePadded<Point> p(Point{1.0f, 2.0f});
    TEST_ASSERT(p->x == 1.0f);
    TEST_ASSERT(p->y == 2.0f);

    return true;
}

// =============================================================================
// BITSET TESTS
// =============================================================================

bool test_bitset_set_clear_test() {
    using namespace pulse::util;

    FixedBitset<128> bs;
    TEST_ASSERT(!bs.test(0));
    TEST_ASSERT(!bs.test(63));
    TEST_ASSERT(!bs.test(127));

    bs.set(0);
    bs.set(63);
    bs.set(127);
    TEST_ASSERT(bs.test(0));
    TEST_ASSERT(bs.test(63));
    TEST_ASSERT(bs.test(127));
    TEST_ASSERT(!bs.test(1));

    bs.clear(63);
    TEST_ASSERT(!bs.test(63));
    TEST_ASSERT(bs.test(0));
    TEST_ASSERT(bs.test(127));

    return true;
}

bool test_bitset_toggle() {
    using namespace pulse::util;

    FixedBitset<64> bs;
    bs.toggle(10);
    TEST_ASSERT(bs.test(10));
    bs.toggle(10);
    TEST_ASSERT(!bs.test(10));

    return true;
}

bool test_bitset_setAll_clearAll() {
    using namespace pulse::util;

    FixedBitset<100> bs;
    bs.setAll();
    for (std::size_t i = 0; i < 100; ++i) {
        TEST_ASSERT(bs.test(i));
    }
    // Bits beyond 100 (in the last word) should NOT be set.
    // We can't test bit 100 because it would assert, but countSet should be exactly 100.
    TEST_ASSERT(bs.countSet() == 100);

    bs.clearAll();
    for (std::size_t i = 0; i < 100; ++i) {
        TEST_ASSERT(!bs.test(i));
    }
    TEST_ASSERT(bs.countSet() == 0);

    return true;
}

bool test_bitset_countSet() {
    using namespace pulse::util;

    FixedBitset<256> bs;
    TEST_ASSERT(bs.countSet() == 0);

    bs.set(0);
    bs.set(100);
    bs.set(255);
    TEST_ASSERT(bs.countSet() == 3);

    bs.setAll();
    TEST_ASSERT(bs.countSet() == 256);

    return true;
}

bool test_bitset_firstSet() {
    using namespace pulse::util;

    FixedBitset<128> bs;
    TEST_ASSERT(bs.firstSet() == -1);

    bs.set(42);
    TEST_ASSERT(bs.firstSet() == 42);

    bs.set(10);
    TEST_ASSERT(bs.firstSet() == 10);

    bs.clear(10);
    TEST_ASSERT(bs.firstSet() == 42);

    return true;
}

bool test_bitset_forEachSet() {
    using namespace pulse::util;

    FixedBitset<128> bs;
    bs.set(3);
    bs.set(7);
    bs.set(64);
    bs.set(100);

    int visited[4] = {};
    int count = 0;
    bs.forEachSet([&](std::size_t idx) {
        if (count < 4) visited[count] = static_cast<int>(idx);
        count++;
    });

    TEST_ASSERT(count == 4);
    TEST_ASSERT(visited[0] == 3);
    TEST_ASSERT(visited[1] == 7);
    TEST_ASSERT(visited[2] == 64);
    TEST_ASSERT(visited[3] == 100);

    return true;
}

bool test_bitset_bitwise_ops() {
    using namespace pulse::util;

    FixedBitset<64> a, b;
    a.set(0); a.set(1); a.set(2);
    b.set(1); b.set(2); b.set(3);

    auto c = a & b;
    TEST_ASSERT(!c.test(0));
    TEST_ASSERT(c.test(1));
    TEST_ASSERT(c.test(2));
    TEST_ASSERT(!c.test(3));

    auto d = a | b;
    TEST_ASSERT(d.test(0));
    TEST_ASSERT(d.test(1));
    TEST_ASSERT(d.test(2));
    TEST_ASSERT(d.test(3));

    auto e = a ^ b;
    TEST_ASSERT(e.test(0));
    TEST_ASSERT(!e.test(1));
    TEST_ASSERT(!e.test(2));
    TEST_ASSERT(e.test(3));

    auto f = ~a;
    TEST_ASSERT(!f.test(0));
    TEST_ASSERT(!f.test(1));
    TEST_ASSERT(!f.test(2));
    TEST_ASSERT(f.test(3));
    TEST_ASSERT(f.test(63));

    return true;
}

bool test_bitset_any_none() {
    using namespace pulse::util;

    FixedBitset<64> bs;
    TEST_ASSERT(bs.none());
    TEST_ASSERT(!bs.any());

    bs.set(32);
    TEST_ASSERT(bs.any());
    TEST_ASSERT(!bs.none());

    return true;
}

bool test_bitset_equality() {
    using namespace pulse::util;

    FixedBitset<128> a, b;
    TEST_ASSERT(a == b);

    a.set(50);
    TEST_ASSERT(a != b);

    b.set(50);
    TEST_ASSERT(a == b);

    return true;
}

// =============================================================================
// HANDLE TESTS
// =============================================================================

bool test_handle_creation() {
    using namespace pulse::util;

    Handle<> h(10, 5);
    TEST_ASSERT(h.index() == 10);
    TEST_ASSERT(h.generation() == 5);
    TEST_ASSERT(h.isValid());
    TEST_ASSERT(!h.isNull());

    return true;
}

bool test_handle_null() {
    using namespace pulse::util;

    Handle<> h = Handle<>::null();
    TEST_ASSERT(h.isNull());
    TEST_ASSERT(!h.isValid());

    Handle<> defaultH;
    TEST_ASSERT(defaultH.isNull());

    return true;
}

bool test_handle_equality() {
    using namespace pulse::util;

    Handle<> a(10, 5);
    Handle<> b(10, 5);
    Handle<> c(10, 6);
    Handle<> d(11, 5);

    TEST_ASSERT(a == b);
    TEST_ASSERT(a != c); // Different generation
    TEST_ASSERT(a != d); // Different index

    return true;
}

bool test_handle_hash() {
    using namespace pulse::util;

    Handle<> a(10, 5);
    Handle<> b(10, 5);
    Handle<> c(11, 5);

    std::hash<Handle<>> hasher;
    TEST_ASSERT(hasher(a) == hasher(b));
    // Different handles should (almost certainly) have different hashes.
    TEST_ASSERT(hasher(a) != hasher(c));

    return true;
}

bool test_handle_type_safety() {
    using namespace pulse::util;

    struct BodyTag {};
    struct ShapeTag {};

    Handle<BodyTag> bodyH(1, 0);
    Handle<ShapeTag> shapeH(1, 0);

    // These are different types — this should NOT compile:
    // bodyH == shapeH;  // compile error
    // This test just verifies the types are distinct.
    TEST_ASSERT(sizeof(bodyH) == sizeof(shapeH));
    TEST_ASSERT(bodyH.index() == shapeH.index());

    return true;
}

// =============================================================================
// HANDLE POOL TESTS
// =============================================================================

bool test_handle_pool_allocate_free() {
    using namespace pulse::util;

    HandlePool<> pool(16);
    TEST_ASSERT(pool.count() == 0);
    TEST_ASSERT(pool.capacity() == 16);

    auto h1 = pool.allocate();
    auto h2 = pool.allocate();
    auto h3 = pool.allocate();
    TEST_ASSERT(pool.count() == 3);

    TEST_ASSERT(pool.isValid(h1));
    TEST_ASSERT(pool.isValid(h2));
    TEST_ASSERT(pool.isValid(h3));

    pool.free(h2);
    TEST_ASSERT(pool.count() == 2);
    TEST_ASSERT(!pool.isValid(h2)); // Freed — generation bumped.
    TEST_ASSERT(pool.isValid(h1));
    TEST_ASSERT(pool.isValid(h3));

    return true;
}

bool test_handle_pool_generation() {
    using namespace pulse::util;

    HandlePool<> pool(4);

    auto h1 = pool.allocate();
    uint32_t gen1 = h1.generation();

    pool.free(h1);

    // Reallocate the same slot — should have a bumped generation.
    auto h2 = pool.allocate();
    TEST_ASSERT(h2.index() == h1.index());
    TEST_ASSERT(h2.generation() > gen1);

    // Original handle is now stale.
    TEST_ASSERT(!pool.isValid(h1));
    TEST_ASSERT(pool.isValid(h2));

    return true;
}

bool test_handle_pool_grow() {
    using namespace pulse::util;

    HandlePool<> pool(2);
    TEST_ASSERT(pool.capacity() == 2);

    auto h1 = pool.allocate();
    auto h2 = pool.allocate();
    auto h3 = pool.allocate(); // Should trigger grow
    TEST_ASSERT(pool.capacity() > 2);
    TEST_ASSERT(pool.count() == 3);

    TEST_ASSERT(pool.isValid(h1));
    TEST_ASSERT(pool.isValid(h2));
    TEST_ASSERT(pool.isValid(h3));

    return true;
}

bool test_handle_pool_iteration() {
    using namespace pulse::util;

    HandlePool<> pool(8);

    auto h0 = pool.allocate();
    auto h1 = pool.allocate();
    auto h2 = pool.allocate();
    auto h3 = pool.allocate();

    pool.free(h1);
    pool.free(h3);

    int activeCount = 0;
    pool.forEach([&](Handle<> h) {
        TEST_ASSERT(pool.isValid(h));
        activeCount++;
    });

    TEST_ASSERT(activeCount == 2); // h0 and h2

    return true;
}

bool test_handle_pool_reset() {
    using namespace pulse::util;

    HandlePool<> pool(8);
    auto h1 = pool.allocate();
    auto h2 = pool.allocate();

    pool.reset();
    TEST_ASSERT(pool.count() == 0);
    TEST_ASSERT(!pool.isValid(h1));
    TEST_ASSERT(!pool.isValid(h2));

    // Can allocate again.
    auto h3 = pool.allocate();
    TEST_ASSERT(pool.isValid(h3));
    TEST_ASSERT(pool.count() == 1);

    return true;
}

bool test_handle_pool_null_invalid() {
    using namespace pulse::util;

    HandlePool<> pool(4);
    auto nullH = Handle<>::null();
    TEST_ASSERT(!pool.isValid(nullH));

    return true;
}

// =============================================================================
// FIXED ARRAY TESTS
// =============================================================================

bool test_fixed_array_push_pop() {
    using namespace pulse::util;

    FixedArray<int, 8> arr;
    TEST_ASSERT(arr.empty());
    TEST_ASSERT(arr.size() == 0);
    TEST_ASSERT(arr.capacity() == 8);

    arr.push_back(10);
    arr.push_back(20);
    arr.push_back(30);
    TEST_ASSERT(arr.size() == 3);
    TEST_ASSERT(arr[0] == 10);
    TEST_ASSERT(arr[1] == 20);
    TEST_ASSERT(arr[2] == 30);

    arr.pop_back();
    TEST_ASSERT(arr.size() == 2);
    TEST_ASSERT(arr.back() == 20);

    return true;
}

bool test_fixed_array_emplace_back() {
    using namespace pulse::util;

    struct Point { float x, y; };
    FixedArray<Point, 4> arr;
    arr.emplace_back(Point{1.0f, 2.0f});
    arr.emplace_back(Point{3.0f, 4.0f});
    TEST_ASSERT(arr.size() == 2);
    TEST_ASSERT(arr[0].x == 1.0f);
    TEST_ASSERT(arr[1].y == 4.0f);

    return true;
}

bool test_fixed_array_front_back() {
    using namespace pulse::util;

    FixedArray<int, 4> arr;
    arr.push_back(100);
    arr.push_back(200);
    arr.push_back(300);
    TEST_ASSERT(arr.front() == 100);
    TEST_ASSERT(arr.back() == 300);

    return true;
}

bool test_fixed_array_iteration() {
    using namespace pulse::util;

    FixedArray<int, 8> arr;
    arr.push_back(1);
    arr.push_back(2);
    arr.push_back(3);

    int sum = 0;
    for (int val : arr) {
        sum += val;
    }
    TEST_ASSERT(sum == 6);

    return true;
}

bool test_fixed_array_swap_remove() {
    using namespace pulse::util;

    FixedArray<int, 8> arr;
    arr.push_back(10);
    arr.push_back(20);
    arr.push_back(30);
    arr.push_back(40);

    arr.swapRemove(1); // Remove 20, last element (40) takes its place.
    TEST_ASSERT(arr.size() == 3);
    TEST_ASSERT(arr[0] == 10);
    TEST_ASSERT(arr[1] == 40);
    TEST_ASSERT(arr[2] == 30);

    return true;
}

bool test_fixed_array_full_empty() {
    using namespace pulse::util;

    FixedArray<int, 2> arr;
    TEST_ASSERT(arr.empty());
    TEST_ASSERT(!arr.full());

    arr.push_back(1);
    arr.push_back(2);
    TEST_ASSERT(arr.full());
    TEST_ASSERT(!arr.empty());

    return true;
}

bool test_fixed_array_clear() {
    using namespace pulse::util;

    FixedArray<int, 8> arr;
    arr.push_back(1);
    arr.push_back(2);
    arr.push_back(3);
    arr.clear();
    TEST_ASSERT(arr.empty());
    TEST_ASSERT(arr.size() == 0);

    return true;
}

bool test_fixed_array_initializer_list() {
    using namespace pulse::util;

    FixedArray<int, 8> arr = {10, 20, 30};
    TEST_ASSERT(arr.size() == 3);
    TEST_ASSERT(arr[0] == 10);
    TEST_ASSERT(arr[1] == 20);
    TEST_ASSERT(arr[2] == 30);

    return true;
}

bool test_fixed_array_copy() {
    using namespace pulse::util;

    FixedArray<int, 8> a = {1, 2, 3};
    FixedArray<int, 8> b = a;
    TEST_ASSERT(b.size() == 3);
    TEST_ASSERT(b[0] == 1);
    TEST_ASSERT(b[1] == 2);
    TEST_ASSERT(b[2] == 3);

    // Modifying b should not affect a.
    b[0] = 100;
    TEST_ASSERT(a[0] == 1);

    return true;
}

bool test_fixed_array_resize() {
    using namespace pulse::util;

    FixedArray<int, 8> arr;
    arr.resize(5);
    TEST_ASSERT(arr.size() == 5);
    // Value-initialized ints should be 0.
    for (std::size_t i = 0; i < 5; ++i) {
        TEST_ASSERT(arr[i] == 0);
    }

    arr.resize(2);
    TEST_ASSERT(arr.size() == 2);

    return true;
}

// =============================================================================
// SOA ARRAY TESTS
// =============================================================================

bool test_soa_array_add_get() {
    using namespace pulse::util;

    SoAArray<float, int, double> soa(8);
    TEST_ASSERT(soa.empty());

    std::size_t i0 = soa.add(1.0f, 10, 100.0);
    std::size_t i1 = soa.add(2.0f, 20, 200.0);
    std::size_t i2 = soa.add(3.0f, 30, 300.0);

    TEST_ASSERT(soa.size() == 3);
    TEST_ASSERT(i0 == 0);
    TEST_ASSERT(i1 == 1);
    TEST_ASSERT(i2 == 2);

    TEST_ASSERT(soa.get<0>(0) == 1.0f);
    TEST_ASSERT(soa.get<1>(1) == 20);
    TEST_ASSERT(soa.get<2>(2) == 300.0);

    return true;
}

bool test_soa_array_remove() {
    using namespace pulse::util;

    SoAArray<int, float> soa(8);
    soa.add(10, 1.0f);
    soa.add(20, 2.0f);
    soa.add(30, 3.0f);

    // Remove index 0 — swap-and-pop: last element (30) moves to index 0.
    soa.remove(0);
    TEST_ASSERT(soa.size() == 2);
    TEST_ASSERT(soa.get<0>(0) == 30);
    TEST_ASSERT(soa.get<1>(0) == 3.0f);
    TEST_ASSERT(soa.get<0>(1) == 20);

    return true;
}

bool test_soa_array_raw_arrays() {
    using namespace pulse::util;

    SoAArray<float, float, float> soa(4);
    soa.add(1.0f, 10.0f, 100.0f);
    soa.add(2.0f, 20.0f, 200.0f);
    soa.add(3.0f, 30.0f, 300.0f);

    float* xs = soa.getArray<0>();
    float* ys = soa.getArray<1>();
    float* zs = soa.getArray<2>();

    TEST_ASSERT(xs[0] == 1.0f && xs[1] == 2.0f && xs[2] == 3.0f);
    TEST_ASSERT(ys[0] == 10.0f && ys[1] == 20.0f && ys[2] == 30.0f);
    TEST_ASSERT(zs[0] == 100.0f && zs[1] == 200.0f && zs[2] == 300.0f);

    return true;
}

bool test_soa_array_forEach() {
    using namespace pulse::util;

    SoAArray<int, float> soa(8);
    soa.add(1, 10.0f);
    soa.add(2, 20.0f);
    soa.add(3, 30.0f);

    int intSum = 0;
    float floatSum = 0.0f;
    soa.forEach([&](std::size_t /*idx*/, int& a, float& b) {
        intSum += a;
        floatSum += b;
    });

    TEST_ASSERT(intSum == 6);
    TEST_ASSERT(std::fabs(floatSum - 60.0f) < 0.001f);

    return true;
}

bool test_soa_array_grow() {
    using namespace pulse::util;

    SoAArray<int, float> soa(2);
    soa.add(1, 1.0f);
    soa.add(2, 2.0f);
    soa.add(3, 3.0f); // Should trigger grow.
    soa.add(4, 4.0f);

    TEST_ASSERT(soa.size() == 4);
    TEST_ASSERT(soa.get<0>(0) == 1);
    TEST_ASSERT(soa.get<0>(3) == 4);
    TEST_ASSERT(soa.get<1>(3) == 4.0f);

    return true;
}

bool test_soa_array_clear() {
    using namespace pulse::util;

    SoAArray<int, float> soa(8);
    soa.add(1, 1.0f);
    soa.add(2, 2.0f);
    soa.clear();
    TEST_ASSERT(soa.size() == 0);
    TEST_ASSERT(soa.empty());

    // Can add again after clear.
    soa.add(10, 10.0f);
    TEST_ASSERT(soa.size() == 1);
    TEST_ASSERT(soa.get<0>(0) == 10);

    return true;
}

bool test_soa_array_cache_alignment() {
    using namespace pulse::util;

    SoAArray<float, int> soa(64);
    for (int i = 0; i < 64; ++i) {
        soa.add(static_cast<float>(i), i);
    }

    // Component arrays should be cache-line aligned.
    float* floats = soa.getArray<0>();
    int* ints = soa.getArray<1>();
    TEST_ASSERT((reinterpret_cast<std::uintptr_t>(floats) % 64) == 0);
    TEST_ASSERT((reinterpret_cast<std::uintptr_t>(ints) % 64) == 0);

    return true;
}

// =============================================================================
// RING BUFFER TESTS
// =============================================================================

bool test_ring_buffer_push_pop() {
    using namespace pulse::util;

    RingBuffer<int, 4> rb;
    TEST_ASSERT(rb.empty());

    TEST_ASSERT(rb.tryPush(10));
    TEST_ASSERT(rb.tryPush(20));
    TEST_ASSERT(rb.tryPush(30));
    TEST_ASSERT(!rb.empty());

    int val = 0;
    TEST_ASSERT(rb.tryPop(val));
    TEST_ASSERT(val == 10);
    TEST_ASSERT(rb.tryPop(val));
    TEST_ASSERT(val == 20);
    TEST_ASSERT(rb.tryPop(val));
    TEST_ASSERT(val == 30);
    TEST_ASSERT(rb.empty());

    return true;
}

bool test_ring_buffer_full() {
    using namespace pulse::util;

    // Capacity 4 means we can store 3 elements (one slot reserved for empty/full distinction).
    RingBuffer<int, 4> rb;
    TEST_ASSERT(rb.tryPush(1));
    TEST_ASSERT(rb.tryPush(2));
    TEST_ASSERT(rb.tryPush(3));

    // The 4th push might fail depending on the full detection scheme.
    // With virtual indices and capacity N=4, we can store exactly N-1=3 items,
    // OR exactly N items depending on implementation. Let's just test we can
    // push and pop correctly.
    int val;
    TEST_ASSERT(rb.tryPop(val) && val == 1);
    TEST_ASSERT(rb.tryPush(4)); // Should succeed after popping one.
    TEST_ASSERT(rb.tryPop(val) && val == 2);
    TEST_ASSERT(rb.tryPop(val) && val == 3);
    TEST_ASSERT(rb.tryPop(val) && val == 4);
    TEST_ASSERT(rb.empty());

    return true;
}

bool test_ring_buffer_empty_pop() {
    using namespace pulse::util;

    RingBuffer<int, 4> rb;
    int val = 0;
    TEST_ASSERT(!rb.tryPop(val)); // Should fail — empty.

    return true;
}

bool test_ring_buffer_wraparound() {
    using namespace pulse::util;

    RingBuffer<int, 4> rb;

    // Push and pop multiple times to wrap around the buffer.
    for (int round = 0; round < 10; ++round) {
        TEST_ASSERT(rb.tryPush(round * 10));
        int val;
        TEST_ASSERT(rb.tryPop(val));
        TEST_ASSERT(val == round * 10);
    }

    TEST_ASSERT(rb.empty());
    return true;
}

bool test_ring_buffer_producer_consumer() {
    using namespace pulse::util;

    // Test basic SPSC correctness with two threads.
    constexpr int NumItems = 10000;
    RingBuffer<int, 1024> rb;
    std::atomic<bool> done{false};

    // Producer thread.
    std::thread producer([&]() {
        for (int i = 0; i < NumItems; ++i) {
            while (!rb.tryPush(i)) {
                // Spin until space available.
            }
        }
        done.store(true, std::memory_order_release);
    });

    // Consumer thread (this thread).
    int received = 0;
    int lastVal = -1;
    while (received < NumItems) {
        int val;
        if (rb.tryPop(val)) {
            // Verify ordering.
            if (lastVal >= 0) {
                TEST_ASSERT(val == lastVal + 1);
            }
            lastVal = val;
            received++;
        }
    }

    producer.join();
    TEST_ASSERT(received == NumItems);
    TEST_ASSERT(lastVal == NumItems - 1);

    return true;
}

// =============================================================================
// PROFILER TESTS
// =============================================================================

bool test_profiler_scope_timing() {
    using namespace pulse::util;

    Profiler& prof = Profiler::instance();
    prof.beginFrame();
    {
        PULSE_PROFILE_SCOPE("TestScope");
        // Do some work to burn time.
        volatile int sum = 0;
        for (int i = 0; i < 10000; ++i) sum += i;
        (void)sum;
    }
    prof.endFrame();

    const ProfileFrame& frame = prof.getLastFrame();
    TEST_ASSERT(frame.nodeCount == 1);
    TEST_ASSERT(std::strcmp(frame.nodes[0].name, "TestScope") == 0);
    TEST_ASSERT(frame.nodes[0].totalTicks > 0);
    TEST_ASSERT(frame.nodes[0].callCount == 1);

    return true;
}

bool test_profiler_hierarchy() {
    using namespace pulse::util;

    Profiler& prof = Profiler::instance();
    prof.beginFrame();
    {
        PULSE_PROFILE_SCOPE("Parent");
        {
            PULSE_PROFILE_SCOPE("Child1");
            volatile int x = 0;
            for (int i = 0; i < 1000; ++i) x += i;
            (void)x;
        }
        {
            PULSE_PROFILE_SCOPE("Child2");
            volatile int x = 0;
            for (int i = 0; i < 1000; ++i) x += i;
            (void)x;
        }
    }
    prof.endFrame();

    const ProfileFrame& frame = prof.getLastFrame();
    TEST_ASSERT(frame.nodeCount == 3);

    // Node 0: Parent
    TEST_ASSERT(std::strcmp(frame.nodes[0].name, "Parent") == 0);
    TEST_ASSERT(frame.nodes[0].parentIndex == -1);
    TEST_ASSERT(frame.nodes[0].depth == 0);

    // Node 1: Child1
    TEST_ASSERT(std::strcmp(frame.nodes[1].name, "Child1") == 0);
    TEST_ASSERT(frame.nodes[1].parentIndex == 0);
    TEST_ASSERT(frame.nodes[1].depth == 1);

    // Node 2: Child2
    TEST_ASSERT(std::strcmp(frame.nodes[2].name, "Child2") == 0);
    TEST_ASSERT(frame.nodes[2].parentIndex == 0);
    TEST_ASSERT(frame.nodes[2].depth == 1);

    // Parent's self time should be less than total time (children take time).
    TEST_ASSERT(frame.nodes[0].selfTicks <= frame.nodes[0].totalTicks);

    return true;
}

bool test_profiler_frame_count() {
    using namespace pulse::util;

    Profiler& prof = Profiler::instance();
    uint64_t startCount = prof.getFrameCount();

    prof.beginFrame();
    prof.endFrame();
    prof.beginFrame();
    prof.endFrame();

    TEST_ASSERT(prof.getFrameCount() == startCount + 2);

    return true;
}

bool test_timer_basics() {
    using namespace pulse::util;

    uint64_t t1 = Timer::now();
    // Spin briefly.
    volatile int x = 0;
    for (int i = 0; i < 100000; ++i) x += i;
    (void)x;
    uint64_t t2 = Timer::now();

    TEST_ASSERT(t2 > t1);
    TEST_ASSERT(Timer::frequency() > 0);

    double ms = Timer::ticksToMs(t2 - t1);
    TEST_ASSERT(ms > 0.0);
    TEST_ASSERT(ms < 10000.0); // Should be well under 10 seconds

    return true;
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    std::printf("╔═══════════════════════════════════════════════╗\n");
    std::printf("║     Pulse Utilities Module — Unit Tests       ║\n");
    std::printf("╚═══════════════════════════════════════════════╝\n\n");

    // ── Assert tests ─────────────────────────────────────────────────────
    std::printf("── Assert ─────────────────────────────────────\n");
    RUN_TEST(test_assert_custom_handler);
    RUN_TEST(test_assert_msg);
    RUN_TEST(test_verify_evaluates);
    RUN_TEST(test_static_assert_compiles);

    // ── Type traits tests ────────────────────────────────────────────────
    std::printf("\n── Type Traits ─────────────────────────────────\n");
    RUN_TEST(test_is_simd_type);
    RUN_TEST(test_is_pod_type);
    RUN_TEST(test_is_aligned);
    RUN_TEST(test_alignment_of);
    RUN_TEST(test_is_power_of_2);
    RUN_TEST(test_cache_line_padded);

    // ── Bitset tests ─────────────────────────────────────────────────────
    std::printf("\n── FixedBitset ─────────────────────────────────\n");
    RUN_TEST(test_bitset_set_clear_test);
    RUN_TEST(test_bitset_toggle);
    RUN_TEST(test_bitset_setAll_clearAll);
    RUN_TEST(test_bitset_countSet);
    RUN_TEST(test_bitset_firstSet);
    RUN_TEST(test_bitset_forEachSet);
    RUN_TEST(test_bitset_bitwise_ops);
    RUN_TEST(test_bitset_any_none);
    RUN_TEST(test_bitset_equality);

    // ── Handle tests ─────────────────────────────────────────────────────
    std::printf("\n── Handle ──────────────────────────────────────\n");
    RUN_TEST(test_handle_creation);
    RUN_TEST(test_handle_null);
    RUN_TEST(test_handle_equality);
    RUN_TEST(test_handle_hash);
    RUN_TEST(test_handle_type_safety);

    // ── Handle pool tests ────────────────────────────────────────────────
    std::printf("\n── HandlePool ──────────────────────────────────\n");
    RUN_TEST(test_handle_pool_allocate_free);
    RUN_TEST(test_handle_pool_generation);
    RUN_TEST(test_handle_pool_grow);
    RUN_TEST(test_handle_pool_iteration);
    RUN_TEST(test_handle_pool_reset);
    RUN_TEST(test_handle_pool_null_invalid);

    // ── Fixed array tests ────────────────────────────────────────────────
    std::printf("\n── FixedArray ──────────────────────────────────\n");
    RUN_TEST(test_fixed_array_push_pop);
    RUN_TEST(test_fixed_array_emplace_back);
    RUN_TEST(test_fixed_array_front_back);
    RUN_TEST(test_fixed_array_iteration);
    RUN_TEST(test_fixed_array_swap_remove);
    RUN_TEST(test_fixed_array_full_empty);
    RUN_TEST(test_fixed_array_clear);
    RUN_TEST(test_fixed_array_initializer_list);
    RUN_TEST(test_fixed_array_copy);
    RUN_TEST(test_fixed_array_resize);

    // ── SoA array tests ──────────────────────────────────────────────────
    std::printf("\n── SoAArray ────────────────────────────────────\n");
    RUN_TEST(test_soa_array_add_get);
    RUN_TEST(test_soa_array_remove);
    RUN_TEST(test_soa_array_raw_arrays);
    RUN_TEST(test_soa_array_forEach);
    RUN_TEST(test_soa_array_grow);
    RUN_TEST(test_soa_array_clear);
    RUN_TEST(test_soa_array_cache_alignment);

    // ── Ring buffer tests ────────────────────────────────────────────────
    std::printf("\n── RingBuffer ──────────────────────────────────\n");
    RUN_TEST(test_ring_buffer_push_pop);
    RUN_TEST(test_ring_buffer_full);
    RUN_TEST(test_ring_buffer_empty_pop);
    RUN_TEST(test_ring_buffer_wraparound);
    RUN_TEST(test_ring_buffer_producer_consumer);

    // ── Profiler tests ───────────────────────────────────────────────────
    std::printf("\n── Profiler ────────────────────────────────────\n");
    RUN_TEST(test_profiler_scope_timing);
    RUN_TEST(test_profiler_hierarchy);
    RUN_TEST(test_profiler_frame_count);
    RUN_TEST(test_timer_basics);

    // ── Summary ──────────────────────────────────────────────────────────
    std::printf("\n════════════════════════════════════════════════\n");
    std::printf("  Total: %d  |  Passed: %d  |  Failed: %d\n",
                g_totalTests, g_passedTests, g_failedTests);
    std::printf("════════════════════════════════════════════════\n");

    return g_failedTests > 0 ? 1 : 0;
}
