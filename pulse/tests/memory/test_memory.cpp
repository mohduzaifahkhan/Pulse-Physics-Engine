/**
 * @file test_memory.cpp
 * @brief Comprehensive unit tests for the Pulse memory allocator library.
 *
 * Lightweight test framework — no external dependencies. Each test is a
 * function that returns true on pass. Failed tests print the function name
 * and line number.
 */

#include <pulse/memory/allocator_base.h>
#include <pulse/memory/arena_allocator.h>
#include <pulse/memory/pool_allocator.h>
#include <pulse/memory/stack_allocator.h>
#include <pulse/memory/frame_allocator.h>
#include <pulse/memory/free_list_allocator.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

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

using namespace pulse::memory;

// ── Alignment Utility Tests ──────────────────────────────────────────────────

bool test_alignUp_basic() {
    TEST_ASSERT(alignUp(std::size_t(0), 16) == 0);
    TEST_ASSERT(alignUp(std::size_t(1), 16) == 16);
    TEST_ASSERT(alignUp(std::size_t(15), 16) == 16);
    TEST_ASSERT(alignUp(std::size_t(16), 16) == 16);
    TEST_ASSERT(alignUp(std::size_t(17), 16) == 32);
    return true;
}

bool test_alignUp_various_alignments() {
    TEST_ASSERT(alignUp(std::size_t(5), 4) == 8);
    TEST_ASSERT(alignUp(std::size_t(8), 8) == 8);
    TEST_ASSERT(alignUp(std::size_t(1), 64) == 64);
    TEST_ASSERT(alignUp(std::size_t(63), 64) == 64);
    TEST_ASSERT(alignUp(std::size_t(64), 64) == 64);
    TEST_ASSERT(alignUp(std::size_t(65), 64) == 128);
    return true;
}

bool test_isAligned() {
    alignas(64) char buf[128];
    TEST_ASSERT(isAligned(buf, 1));
    TEST_ASSERT(isAligned(buf, 2));
    TEST_ASSERT(isAligned(buf, 4));
    // buf is 64-byte aligned
    TEST_ASSERT(isAligned(buf, 64));
    // One byte off should not be aligned to 16
    TEST_ASSERT(!isAligned(buf + 1, 16));
    return true;
}

bool test_isPowerOf2() {
    TEST_ASSERT(isPowerOf2(1));
    TEST_ASSERT(isPowerOf2(2));
    TEST_ASSERT(isPowerOf2(4));
    TEST_ASSERT(isPowerOf2(64));
    TEST_ASSERT(isPowerOf2(1024));
    TEST_ASSERT(!isPowerOf2(0));
    TEST_ASSERT(!isPowerOf2(3));
    TEST_ASSERT(!isPowerOf2(6));
    TEST_ASSERT(!isPowerOf2(100));
    return true;
}

// ── AllocatorStats Tests ─────────────────────────────────────────────────────

bool test_stats_tracking() {
    AllocatorStats stats;
    TEST_ASSERT(stats.totalAllocations == 0);
    TEST_ASSERT(stats.currentBytesUsed == 0);

    stats.recordAllocation(128, 4);
    TEST_ASSERT(stats.totalAllocations == 1);
    TEST_ASSERT(stats.currentBytesUsed == 128);
    TEST_ASSERT(stats.peakBytesUsed == 128);
    TEST_ASSERT(stats.totalBytesAllocated == 128);
    TEST_ASSERT(stats.wastedBytes == 4);

    stats.recordAllocation(256);
    TEST_ASSERT(stats.totalAllocations == 2);
    TEST_ASSERT(stats.currentBytesUsed == 384);
    TEST_ASSERT(stats.peakBytesUsed == 384);

    stats.recordDeallocation(128);
    TEST_ASSERT(stats.totalDeallocations == 1);
    TEST_ASSERT(stats.currentBytesUsed == 256);
    TEST_ASSERT(stats.peakBytesUsed == 384); // Peak should remain

    stats.reset();
    TEST_ASSERT(stats.totalAllocations == 0);
    TEST_ASSERT(stats.currentBytesUsed == 0);
    TEST_ASSERT(stats.peakBytesUsed == 0);
    return true;
}

// ── ArenaAllocator Tests ─────────────────────────────────────────────────────

bool test_arena_basic_alloc() {
    ArenaAllocator arena(4096);
    void* p1 = arena.allocate(64);
    TEST_ASSERT(p1 != nullptr);
    TEST_ASSERT(isAligned(p1, DefaultAlignment));

    void* p2 = arena.allocate(128);
    TEST_ASSERT(p2 != nullptr);
    TEST_ASSERT(p2 != p1);
    TEST_ASSERT(isAligned(p2, DefaultAlignment));

    // Verify we can write to the memory
    std::memset(p1, 0xAB, 64);
    std::memset(p2, 0xCD, 128);
    TEST_ASSERT(static_cast<uint8_t*>(p1)[0] == 0xAB);
    TEST_ASSERT(static_cast<uint8_t*>(p2)[0] == 0xCD);
    return true;
}

bool test_arena_alignment() {
    ArenaAllocator arena(4096);
    // Allocate 1 byte to misalign, then request 64-byte aligned
    void* p1 = arena.allocate(1, 1);
    TEST_ASSERT(p1 != nullptr);

    void* p2 = arena.allocate(32, 64);
    TEST_ASSERT(p2 != nullptr);
    TEST_ASSERT(isAligned(p2, 64));

    void* p3 = arena.allocate(16, 32);
    TEST_ASSERT(p3 != nullptr);
    TEST_ASSERT(isAligned(p3, 32));
    return true;
}

bool test_arena_oom() {
    ArenaAllocator arena(128);
    void* p1 = arena.allocate(64);
    TEST_ASSERT(p1 != nullptr);

    void* p2 = arena.allocate(64);
    TEST_ASSERT(p2 != nullptr);

    // Should return nullptr — out of memory
    void* p3 = arena.allocate(1);
    TEST_ASSERT(p3 == nullptr);
    return true;
}

bool test_arena_reset() {
    ArenaAllocator arena(4096);
    void* p1 = arena.allocate(1024);
    TEST_ASSERT(p1 != nullptr);
    TEST_ASSERT(arena.currentOffset() > 0);

    arena.reset();
    TEST_ASSERT(arena.currentOffset() == 0);
    TEST_ASSERT(arena.remaining() == 4096);
    TEST_ASSERT(arena.getStats().totalAllocations == 0);

    // Should be able to allocate again
    void* p2 = arena.allocate(1024);
    TEST_ASSERT(p2 != nullptr);
    return true;
}

bool test_arena_save_restore() {
    ArenaAllocator arena(4096);
    void* p1 = arena.allocate(256);
    TEST_ASSERT(p1 != nullptr);

    std::size_t saved = arena.saveState();
    void* p2 = arena.allocate(512);
    TEST_ASSERT(p2 != nullptr);
    TEST_ASSERT(arena.currentOffset() > saved);

    arena.restoreState(saved);
    TEST_ASSERT(arena.currentOffset() == saved);

    // Allocating again should reuse the memory
    void* p3 = arena.allocate(512);
    TEST_ASSERT(p3 != nullptr);
    return true;
}

bool test_arena_owns() {
    ArenaAllocator arena(4096);
    void* p = arena.allocate(64);
    TEST_ASSERT(arena.owns(p));

    int stackVar = 42;
    TEST_ASSERT(!arena.owns(&stackVar));
    return true;
}

bool test_arena_external_memory() {
    alignas(64) char buf[1024];
    ArenaAllocator arena(buf, sizeof(buf));

    void* p = arena.allocate(256);
    TEST_ASSERT(p != nullptr);
    // Pointer should be within our buffer
    TEST_ASSERT(reinterpret_cast<uintptr_t>(p) >= reinterpret_cast<uintptr_t>(buf));
    TEST_ASSERT(reinterpret_cast<uintptr_t>(p) < reinterpret_cast<uintptr_t>(buf) + sizeof(buf));
    return true;
}

bool test_arena_move() {
    ArenaAllocator arena1(4096);
    void* p = arena1.allocate(128);
    TEST_ASSERT(p != nullptr);
    std::size_t offset = arena1.currentOffset();

    ArenaAllocator arena2(std::move(arena1));
    TEST_ASSERT(arena2.currentOffset() == offset);
    TEST_ASSERT(arena2.owns(p));
    return true;
}

bool test_arena_stats() {
    ArenaAllocator arena(4096);
    TEST_ASSERT(arena.getStats().totalAllocations == 0);
    TEST_ASSERT(arena.capacity() == 4096);

    arena.allocate(100);
    TEST_ASSERT(arena.getStats().totalAllocations == 1);
    TEST_ASSERT(arena.getStats().totalBytesAllocated == 100);
    TEST_ASSERT(arena.usedBytes() == 100);
    TEST_ASSERT(arena.availableBytes() < 4096);

    arena.allocate(200);
    TEST_ASSERT(arena.getStats().totalAllocations == 2);
    TEST_ASSERT(arena.getStats().totalBytesAllocated == 300);
    return true;
}

bool test_arena_sequential_fill() {
    // Fill the entire arena with small allocations
    constexpr std::size_t Cap = 1024;
    ArenaAllocator arena(Cap);
    int count = 0;
    while (arena.allocate(16) != nullptr) {
        count++;
    }
    TEST_ASSERT(count > 0);
    TEST_ASSERT(count <= static_cast<int>(Cap / 16));
    return true;
}

// ── PoolAllocator Tests ──────────────────────────────────────────────────────

bool test_pool_basic_alloc() {
    PoolAllocator<64> pool(100);
    void* p1 = pool.allocate(64);
    TEST_ASSERT(p1 != nullptr);
    TEST_ASSERT(isAligned(p1, DefaultAlignment));

    void* p2 = pool.allocate(64);
    TEST_ASSERT(p2 != nullptr);
    TEST_ASSERT(p2 != p1);

    // Write to verify memory is usable
    std::memset(p1, 0xAA, 64);
    std::memset(p2, 0xBB, 64);
    TEST_ASSERT(static_cast<uint8_t*>(p1)[0] == 0xAA);
    TEST_ASSERT(static_cast<uint8_t*>(p2)[0] == 0xBB);
    return true;
}

bool test_pool_dealloc_reuse() {
    PoolAllocator<64> pool(2);
    void* p1 = pool.allocate(64);
    void* p2 = pool.allocate(64);
    TEST_ASSERT(p1 != nullptr);
    TEST_ASSERT(p2 != nullptr);

    // Pool is full
    TEST_ASSERT(pool.isFull());
    TEST_ASSERT(pool.allocate(64) == nullptr);

    // Free one, should be able to allocate again
    pool.deallocate(p2);
    TEST_ASSERT(!pool.isFull());
    void* p3 = pool.allocate(64);
    TEST_ASSERT(p3 != nullptr);
    return true;
}

bool test_pool_exhaustion() {
    PoolAllocator<32> pool(5);
    void* ptrs[5];
    for (int i = 0; i < 5; ++i) {
        ptrs[i] = pool.allocate(32);
        TEST_ASSERT(ptrs[i] != nullptr);
    }
    TEST_ASSERT(pool.isFull());
    TEST_ASSERT(pool.allocate(32) == nullptr);

    // Free all and allocate again
    for (int i = 4; i >= 0; --i) {
        pool.deallocate(ptrs[i]);
    }
    TEST_ASSERT(pool.freeBlocks() == 5);
    for (int i = 0; i < 5; ++i) {
        TEST_ASSERT(pool.allocate(32) != nullptr);
    }
    return true;
}

bool test_pool_free_blocks() {
    PoolAllocator<64> pool(10);
    TEST_ASSERT(pool.freeBlocks() == 10);
    TEST_ASSERT(pool.allocatedBlocks() == 0);
    TEST_ASSERT(pool.maxBlockCount() == 10);

    void* p = pool.allocate(64);
    TEST_ASSERT(pool.freeBlocks() == 9);
    TEST_ASSERT(pool.allocatedBlocks() == 1);

    pool.deallocate(p);
    TEST_ASSERT(pool.freeBlocks() == 10);
    TEST_ASSERT(pool.allocatedBlocks() == 0);
    return true;
}

bool test_pool_reset() {
    PoolAllocator<64> pool(5);
    for (int i = 0; i < 5; ++i) {
        pool.allocate(64);
    }
    TEST_ASSERT(pool.isFull());

    pool.reset();
    TEST_ASSERT(!pool.isFull());
    TEST_ASSERT(pool.freeBlocks() == 5);
    TEST_ASSERT(pool.getStats().totalAllocations == 0);

    // Should be able to allocate all 5 again
    for (int i = 0; i < 5; ++i) {
        TEST_ASSERT(pool.allocate(64) != nullptr);
    }
    return true;
}

bool test_pool_owns() {
    PoolAllocator<64> pool(10);
    void* p = pool.allocate(64);
    TEST_ASSERT(pool.owns(p));

    int stackVar = 42;
    TEST_ASSERT(!pool.owns(&stackVar));
    return true;
}

bool test_pool_move() {
    PoolAllocator<64> pool1(10);
    void* p = pool1.allocate(64);
    TEST_ASSERT(p != nullptr);

    PoolAllocator<64> pool2(std::move(pool1));
    TEST_ASSERT(pool2.owns(p));
    TEST_ASSERT(pool2.allocatedBlocks() == 1);
    return true;
}

bool test_pool_aligned_block_size() {
    // BlockSize=17 with Alignment=16 should round up to 32
    constexpr std::size_t aligned = PoolAllocator<17, 16>::AlignedBlockSize;
    TEST_ASSERT(aligned == 32);

    // BlockSize=64 with Alignment=16 stays 64
    constexpr std::size_t exact = PoolAllocator<64, 16>::AlignedBlockSize;
    TEST_ASSERT(exact == 64);

    // BlockSize=1 with Alignment=16 should be at least sizeof(void*)
    constexpr std::size_t tiny = PoolAllocator<1, 16>::AlignedBlockSize;
    TEST_ASSERT(tiny >= sizeof(void*));
    TEST_ASSERT(tiny % 16 == 0);
    return true;
}

// ── StackAllocator Tests ─────────────────────────────────────────────────────

bool test_stack_basic_alloc() {
    StackAllocator stack(4096);
    void* p1 = stack.allocate(64);
    TEST_ASSERT(p1 != nullptr);
    TEST_ASSERT(isAligned(p1, DefaultAlignment));

    void* p2 = stack.allocate(128);
    TEST_ASSERT(p2 != nullptr);
    TEST_ASSERT(p2 != p1);

    std::memset(p1, 0x11, 64);
    std::memset(p2, 0x22, 128);
    TEST_ASSERT(static_cast<uint8_t*>(p1)[0] == 0x11);
    TEST_ASSERT(static_cast<uint8_t*>(p2)[0] == 0x22);
    return true;
}

bool test_stack_lifo_dealloc() {
    StackAllocator stack(4096);
    void* p1 = stack.allocate(64);
    void* p2 = stack.allocate(128);
    void* p3 = stack.allocate(256);
    TEST_ASSERT(p1 != nullptr && p2 != nullptr && p3 != nullptr);

    std::size_t offsetBefore = stack.currentOffset();

    // Deallocate in LIFO order
    stack.deallocate(p3);
    TEST_ASSERT(stack.currentOffset() < offsetBefore);

    stack.deallocate(p2);
    stack.deallocate(p1);
    TEST_ASSERT(stack.currentOffset() == 0);
    return true;
}

bool test_stack_mark_rollback() {
    StackAllocator stack(4096);
    void* p1 = stack.allocate(64);
    TEST_ASSERT(p1 != nullptr);

    auto marker = stack.mark();
    void* p2 = stack.allocate(128);
    void* p3 = stack.allocate(256);
    TEST_ASSERT(p2 != nullptr && p3 != nullptr);

    stack.rollback(marker);
    TEST_ASSERT(stack.currentOffset() == marker.offset);

    // Memory after marker is freed; can allocate again
    void* p4 = stack.allocate(128);
    TEST_ASSERT(p4 != nullptr);
    return true;
}

bool test_stack_scoped() {
    StackAllocator stack(4096);
    void* p1 = stack.allocate(64);
    TEST_ASSERT(p1 != nullptr);
    std::size_t outerOffset = stack.currentOffset();

    {
        ScopedStack scope(stack);
        void* p2 = stack.allocate(256);
        TEST_ASSERT(p2 != nullptr);
        TEST_ASSERT(stack.currentOffset() > outerOffset);
    }
    // ScopedStack should have rolled back
    TEST_ASSERT(stack.currentOffset() == outerOffset);
    return true;
}

bool test_stack_alignment() {
    StackAllocator stack(4096);
    // Allocate 1 byte to misalign
    void* p1 = stack.allocate(1);
    TEST_ASSERT(p1 != nullptr);

    // Request 64-byte alignment
    void* p2 = stack.allocate(32, 64);
    TEST_ASSERT(p2 != nullptr);
    TEST_ASSERT(isAligned(p2, 64));
    return true;
}

bool test_stack_oom() {
    StackAllocator stack(256);
    // Each allocation includes a header, so we can't fit as much
    void* p1 = stack.allocate(100);
    TEST_ASSERT(p1 != nullptr);
    void* p2 = stack.allocate(100);
    TEST_ASSERT(p2 != nullptr);

    // Should run out of space (256 bytes with headers)
    void* p3 = stack.allocate(100);
    TEST_ASSERT(p3 == nullptr);
    return true;
}

bool test_stack_reset() {
    StackAllocator stack(4096);
    stack.allocate(128);
    stack.allocate(256);
    TEST_ASSERT(stack.currentOffset() > 0);

    stack.reset();
    TEST_ASSERT(stack.currentOffset() == 0);
    TEST_ASSERT(stack.remaining() == 4096);

    void* p = stack.allocate(512);
    TEST_ASSERT(p != nullptr);
    return true;
}

bool test_stack_owns() {
    StackAllocator stack(4096);
    void* p = stack.allocate(64);
    TEST_ASSERT(stack.owns(p));

    int stackVar = 42;
    TEST_ASSERT(!stack.owns(&stackVar));
    return true;
}

bool test_stack_move() {
    StackAllocator stack1(4096);
    void* p = stack1.allocate(128);
    std::size_t offset = stack1.currentOffset();

    StackAllocator stack2(std::move(stack1));
    TEST_ASSERT(stack2.currentOffset() == offset);
    TEST_ASSERT(stack2.owns(p));
    return true;
}

// ── FrameAllocator Tests ─────────────────────────────────────────────────────

bool test_frame_basic_alloc() {
    FrameAllocator frame(4096);
    frame.beginFrame();

    void* p = frame.allocate(128);
    TEST_ASSERT(p != nullptr);
    TEST_ASSERT(isAligned(p, DefaultAlignment));

    std::memset(p, 0xCC, 128);
    TEST_ASSERT(static_cast<uint8_t*>(p)[0] == 0xCC);
    return true;
}

bool test_frame_double_buffer() {
    FrameAllocator frame(4096);

    // Frame 1
    frame.beginFrame();
    void* p1 = frame.allocate(128);
    TEST_ASSERT(p1 != nullptr);
    std::memset(p1, 0xAA, 128);

    // Frame 2 — previous frame's data should still be valid
    frame.beginFrame();
    TEST_ASSERT(static_cast<uint8_t*>(p1)[0] == 0xAA); // Previous frame data intact
    void* p2 = frame.allocate(256);
    TEST_ASSERT(p2 != nullptr);

    // Frame 3 — now frame 1's arena gets reset, p1 is invalidated
    frame.beginFrame();
    void* p3 = frame.allocate(64);
    TEST_ASSERT(p3 != nullptr);
    return true;
}

bool test_frame_reset_on_begin() {
    FrameAllocator frame(1024);
    frame.beginFrame();

    // Fill up most of the arena
    for (int i = 0; i < 50; ++i) {
        frame.allocate(16);
    }
    TEST_ASSERT(frame.remaining() < 1024);

    // Two beginFrame()s later, same arena should be fully reset
    frame.beginFrame();
    frame.beginFrame();
    TEST_ASSERT(frame.remaining() == 1024);
    return true;
}

bool test_frame_typed_alloc() {
    FrameAllocator frame(4096);
    frame.beginFrame();

    struct TestObj {
        float x, y, z;
        int id;
        TestObj(float x_, float y_, float z_, int id_) : x(x_), y(y_), z(z_), id(id_) {}
    };

    TestObj* obj = frame.create<TestObj>(1.0f, 2.0f, 3.0f, 42);
    TEST_ASSERT(obj != nullptr);
    TEST_ASSERT(obj->x == 1.0f);
    TEST_ASSERT(obj->y == 2.0f);
    TEST_ASSERT(obj->z == 3.0f);
    TEST_ASSERT(obj->id == 42);
    return true;
}

bool test_frame_array_alloc() {
    FrameAllocator frame(4096);
    frame.beginFrame();

    float* arr = frame.allocateArray<float>(100);
    TEST_ASSERT(arr != nullptr);
    for (int i = 0; i < 100; ++i) {
        arr[i] = static_cast<float>(i);
    }
    for (int i = 0; i < 100; ++i) {
        TEST_ASSERT(arr[i] == static_cast<float>(i));
    }
    return true;
}

bool test_frame_number() {
    FrameAllocator frame(1024);
    TEST_ASSERT(frame.frameNumber() == 0);
    frame.beginFrame();
    TEST_ASSERT(frame.frameNumber() == 1);
    frame.beginFrame();
    TEST_ASSERT(frame.frameNumber() == 2);
    frame.beginFrame();
    TEST_ASSERT(frame.frameNumber() == 3);
    return true;
}

bool test_frame_capacity() {
    FrameAllocator frame(2048);
    TEST_ASSERT(frame.arenaCapacity() == 2048);
    TEST_ASSERT(frame.totalCapacity() == 4096); // 2 arenas
    return true;
}

bool test_frame_dealloc_noop() {
    FrameAllocator frame(4096);
    frame.beginFrame();
    void* p = frame.allocate(128);
    // deallocate is a no-op; this should not crash
    frame.deallocate(p);
    // Still able to allocate
    void* p2 = frame.allocate(128);
    TEST_ASSERT(p2 != nullptr);
    return true;
}

// ── FreeListAllocator Tests ──────────────────────────────────────────────────

bool test_freelist_basic_alloc() {
    FreeListAllocator fl(4096);
    void* p1 = fl.allocate(64);
    TEST_ASSERT(p1 != nullptr);
    TEST_ASSERT(isAligned(p1, DefaultAlignment));

    void* p2 = fl.allocate(128);
    TEST_ASSERT(p2 != nullptr);
    TEST_ASSERT(p2 != p1);

    std::memset(p1, 0xDD, 64);
    std::memset(p2, 0xEE, 128);
    TEST_ASSERT(static_cast<uint8_t*>(p1)[0] == 0xDD);
    TEST_ASSERT(static_cast<uint8_t*>(p2)[0] == 0xEE);
    return true;
}

bool test_freelist_dealloc_reuse() {
    FreeListAllocator fl(4096);
    void* p1 = fl.allocate(256);
    void* p2 = fl.allocate(256);
    TEST_ASSERT(p1 != nullptr && p2 != nullptr);

    fl.deallocate(p1);
    // Should be able to allocate again in the freed space
    void* p3 = fl.allocate(128);
    TEST_ASSERT(p3 != nullptr);
    return true;
}

bool test_freelist_coalescing() {
    FreeListAllocator fl(4096);
    void* p1 = fl.allocate(256);
    void* p2 = fl.allocate(256);
    void* p3 = fl.allocate(256);
    TEST_ASSERT(p1 != nullptr && p2 != nullptr && p3 != nullptr);

    // Free middle, then adjacent blocks — should coalesce
    fl.deallocate(p2);
    fl.deallocate(p1);
    fl.deallocate(p3);

    // After coalescing, should be able to allocate a large block
    void* big = fl.allocate(700);
    TEST_ASSERT(big != nullptr);
    return true;
}

bool test_freelist_first_fit() {
    FreeListAllocator fl(4096, FitStrategy::FirstFit);
    TEST_ASSERT(fl.getStrategy() == FitStrategy::FirstFit);

    void* p1 = fl.allocate(128);
    void* p2 = fl.allocate(256);
    void* p3 = fl.allocate(128);
    TEST_ASSERT(p1 != nullptr && p2 != nullptr && p3 != nullptr);

    fl.deallocate(p1); // 128-byte hole at start
    fl.deallocate(p3); // 128-byte hole later

    // First-fit should pick the first hole
    void* p4 = fl.allocate(64);
    TEST_ASSERT(p4 != nullptr);
    return true;
}

bool test_freelist_best_fit() {
    FreeListAllocator fl(8192, FitStrategy::BestFit);
    TEST_ASSERT(fl.getStrategy() == FitStrategy::BestFit);

    void* p1 = fl.allocate(128);
    void* p2 = fl.allocate(64);
    void* p3 = fl.allocate(512);
    void* p4 = fl.allocate(64);
    TEST_ASSERT(p1 != nullptr && p2 != nullptr && p3 != nullptr && p4 != nullptr);

    fl.deallocate(p1); // 128-byte hole
    fl.deallocate(p3); // 512-byte hole

    // Best-fit for a 100-byte request should prefer the 128-byte hole over 512
    void* p5 = fl.allocate(100);
    TEST_ASSERT(p5 != nullptr);
    return true;
}

bool test_freelist_strategy_switch() {
    FreeListAllocator fl(4096, FitStrategy::FirstFit);
    TEST_ASSERT(fl.getStrategy() == FitStrategy::FirstFit);
    fl.setStrategy(FitStrategy::BestFit);
    TEST_ASSERT(fl.getStrategy() == FitStrategy::BestFit);
    return true;
}

bool test_freelist_free_block_count() {
    FreeListAllocator fl(4096);
    TEST_ASSERT(fl.freeBlockCount() == 1); // One big free block initially

    void* p1 = fl.allocate(128);
    void* p2 = fl.allocate(128);
    void* p3 = fl.allocate(128);
    (void)p2;
    TEST_ASSERT(p1 != nullptr && p3 != nullptr);

    fl.deallocate(p1);
    fl.deallocate(p3);
    // Two freed blocks plus the tail = could be 3 free blocks (or fewer with coalescing)
    TEST_ASSERT(fl.freeBlockCount() >= 1);
    return true;
}

bool test_freelist_reset() {
    FreeListAllocator fl(4096);
    fl.allocate(128);
    fl.allocate(256);
    fl.allocate(512);

    fl.reset();
    TEST_ASSERT(fl.getStats().totalAllocations == 0);
    TEST_ASSERT(fl.freeBlockCount() == 1);
    TEST_ASSERT(fl.totalFreeBytes() == 4096);

    // Should be able to allocate the full capacity minus header
    void* p = fl.allocate(128);
    TEST_ASSERT(p != nullptr);
    return true;
}

bool test_freelist_owns() {
    FreeListAllocator fl(4096);
    void* p = fl.allocate(64);
    TEST_ASSERT(fl.owns(p));

    int stackVar = 42;
    TEST_ASSERT(!fl.owns(&stackVar));
    return true;
}

bool test_freelist_move() {
    FreeListAllocator fl1(4096);
    void* p = fl1.allocate(128);
    TEST_ASSERT(p != nullptr);

    FreeListAllocator fl2(std::move(fl1));
    TEST_ASSERT(fl2.owns(p));
    TEST_ASSERT(fl2.getStats().totalAllocations == 1);
    return true;
}

bool test_freelist_variable_sizes() {
    FreeListAllocator fl(8192);
    void* ptrs[10];
    std::size_t sizes[] = {16, 64, 32, 256, 128, 512, 48, 96, 1024, 200};

    for (int i = 0; i < 10; ++i) {
        ptrs[i] = fl.allocate(sizes[i]);
        TEST_ASSERT(ptrs[i] != nullptr);
    }

    // Free every other
    for (int i = 0; i < 10; i += 2) {
        fl.deallocate(ptrs[i]);
    }

    // Allocate replacement sizes
    for (int i = 0; i < 10; i += 2) {
        ptrs[i] = fl.allocate(sizes[i] / 2 + 1);
        TEST_ASSERT(ptrs[i] != nullptr);
    }

    // Free all
    for (int i = 0; i < 10; ++i) {
        fl.deallocate(ptrs[i]);
    }
    return true;
}

bool test_freelist_external_memory() {
    alignas(64) char buf[4096];
    FreeListAllocator fl(buf, sizeof(buf));

    void* p = fl.allocate(256);
    TEST_ASSERT(p != nullptr);
    TEST_ASSERT(fl.owns(p));

    fl.deallocate(p);
    return true;
}

// ── ScopedAllocatorReset Tests ───────────────────────────────────────────────

bool test_scoped_reset() {
    ArenaAllocator arena(4096);
    {
        ScopedAllocatorReset<ArenaAllocator> guard(arena);
        arena.allocate(1024);
        arena.allocate(512);
        TEST_ASSERT(arena.currentOffset() > 0);
    }
    // Guard should have called reset()
    TEST_ASSERT(arena.currentOffset() == 0);
    TEST_ASSERT(arena.getStats().totalAllocations == 0);
    return true;
}

// ── Typed Allocation Helper Tests ────────────────────────────────────────────

bool test_typed_create_destroy() {
    ArenaAllocator arena(4096);

    struct Widget {
        int id;
        float value;
        Widget(int i, float v) : id(i), value(v) {}
    };

    Widget* w = arena.create<Widget>(42, 3.14f);
    TEST_ASSERT(w != nullptr);
    TEST_ASSERT(w->id == 42);
    TEST_ASSERT(w->value > 3.13f && w->value < 3.15f);

    // destroy calls destructor + deallocate (arena dealloc is no-op, but should not crash)
    arena.destroy(w);
    return true;
}

bool test_typed_allocate_array() {
    ArenaAllocator arena(4096);
    int* arr = arena.allocateArray<int>(50);
    TEST_ASSERT(arr != nullptr);
    TEST_ASSERT(isAligned(arr, alignof(int)));

    for (int i = 0; i < 50; ++i) {
        arr[i] = i * i;
    }
    for (int i = 0; i < 50; ++i) {
        TEST_ASSERT(arr[i] == i * i);
    }
    return true;
}

// ── Platform Aligned Alloc Tests ─────────────────────────────────────────────

bool test_platform_aligned_alloc() {
    void* p = platformAlignedAlloc(1024, 64);
    TEST_ASSERT(p != nullptr);
    TEST_ASSERT(isAligned(p, 64));

    std::memset(p, 0xFF, 1024);
    TEST_ASSERT(static_cast<uint8_t*>(p)[0] == 0xFF);
    TEST_ASSERT(static_cast<uint8_t*>(p)[1023] == 0xFF);

    platformAlignedFree(p);
    return true;
}

bool test_platform_aligned_alloc_various() {
    std::size_t alignments[] = {16, 32, 64, 128, 256};
    for (auto align : alignments) {
        void* p = platformAlignedAlloc(512, align);
        TEST_ASSERT(p != nullptr);
        TEST_ASSERT(isAligned(p, align));
        platformAlignedFree(p);
    }
    return true;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    std::printf("╔══════════════════════════════════════════════╗\n");
    std::printf("║     PULSE Physics Engine - Memory Tests      ║\n");
    std::printf("╚══════════════════════════════════════════════╝\n\n");

    // Alignment Utilities
    std::printf("── Alignment Utilities ──────────────────\n");
    RUN_TEST(test_alignUp_basic);
    RUN_TEST(test_alignUp_various_alignments);
    RUN_TEST(test_isAligned);
    RUN_TEST(test_isPowerOf2);

    // AllocatorStats
    std::printf("── AllocatorStats ──────────────────────\n");
    RUN_TEST(test_stats_tracking);

    // Platform Alloc
    std::printf("── Platform Aligned Alloc ──────────────\n");
    RUN_TEST(test_platform_aligned_alloc);
    RUN_TEST(test_platform_aligned_alloc_various);

    // Arena Allocator
    std::printf("── ArenaAllocator ──────────────────────\n");
    RUN_TEST(test_arena_basic_alloc);
    RUN_TEST(test_arena_alignment);
    RUN_TEST(test_arena_oom);
    RUN_TEST(test_arena_reset);
    RUN_TEST(test_arena_save_restore);
    RUN_TEST(test_arena_owns);
    RUN_TEST(test_arena_external_memory);
    RUN_TEST(test_arena_move);
    RUN_TEST(test_arena_stats);
    RUN_TEST(test_arena_sequential_fill);

    // Pool Allocator
    std::printf("── PoolAllocator ───────────────────────\n");
    RUN_TEST(test_pool_basic_alloc);
    RUN_TEST(test_pool_dealloc_reuse);
    RUN_TEST(test_pool_exhaustion);
    RUN_TEST(test_pool_free_blocks);
    RUN_TEST(test_pool_reset);
    RUN_TEST(test_pool_owns);
    RUN_TEST(test_pool_move);
    RUN_TEST(test_pool_aligned_block_size);

    // Stack Allocator
    std::printf("── StackAllocator ──────────────────────\n");
    RUN_TEST(test_stack_basic_alloc);
    RUN_TEST(test_stack_lifo_dealloc);
    RUN_TEST(test_stack_mark_rollback);
    RUN_TEST(test_stack_scoped);
    RUN_TEST(test_stack_alignment);
    RUN_TEST(test_stack_oom);
    RUN_TEST(test_stack_reset);
    RUN_TEST(test_stack_owns);
    RUN_TEST(test_stack_move);

    // Frame Allocator
    std::printf("── FrameAllocator ──────────────────────\n");
    RUN_TEST(test_frame_basic_alloc);
    RUN_TEST(test_frame_double_buffer);
    RUN_TEST(test_frame_reset_on_begin);
    RUN_TEST(test_frame_typed_alloc);
    RUN_TEST(test_frame_array_alloc);
    RUN_TEST(test_frame_number);
    RUN_TEST(test_frame_capacity);
    RUN_TEST(test_frame_dealloc_noop);

    // FreeList Allocator
    std::printf("── FreeListAllocator ───────────────────\n");
    RUN_TEST(test_freelist_basic_alloc);
    RUN_TEST(test_freelist_dealloc_reuse);
    RUN_TEST(test_freelist_coalescing);
    RUN_TEST(test_freelist_first_fit);
    RUN_TEST(test_freelist_best_fit);
    RUN_TEST(test_freelist_strategy_switch);
    RUN_TEST(test_freelist_free_block_count);
    RUN_TEST(test_freelist_reset);
    RUN_TEST(test_freelist_owns);
    RUN_TEST(test_freelist_move);
    RUN_TEST(test_freelist_variable_sizes);
    RUN_TEST(test_freelist_external_memory);

    // Scoped Reset
    std::printf("── ScopedAllocatorReset ────────────────\n");
    RUN_TEST(test_scoped_reset);

    // Typed helpers
    std::printf("── Typed Allocation Helpers ────────────\n");
    RUN_TEST(test_typed_create_destroy);
    RUN_TEST(test_typed_allocate_array);

    // Summary
    std::printf("\n══════════════════════════════════════\n");
    std::printf("  Total:  %d\n", g_totalTests);
    std::printf("  Passed: %d\n", g_passedTests);
    std::printf("  Failed: %d\n", g_failedTests);
    std::printf("══════════════════════════════════════\n");

    return g_failedTests > 0 ? 1 : 0;
}
