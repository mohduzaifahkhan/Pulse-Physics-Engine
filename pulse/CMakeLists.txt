cmake_minimum_required(VERSION 3.28)
project(Pulse VERSION 0.1.0 LANGUAGES CXX)

# ── C++23 ──────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Compiler flags ─────────────────────────────────────
if(MSVC)
    add_compile_options(/W4 /WX /O2 /arch:AVX2 /GL /fp:fast /Oi /Ob2)
    add_link_options(/LTCG)
else()
    add_compile_options(
        -Wall -Wextra -Wpedantic -Werror
        -O3 -mavx2 -msse4.2 -flto -ffast-math
        -fno-exceptions -fno-rtti
        -march=native
    )
    add_link_options(-flto)
endif()

# ── Include directory ──────────────────────────────────
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ── Math module (header-only for now) ──────────────────
add_library(pulse_math INTERFACE)
target_include_directories(pulse_math INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ── Memory module (header-only for now) ────────────────
add_library(pulse_memory INTERFACE)
target_include_directories(pulse_memory INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(pulse_memory INTERFACE pulse_math)

# ── Utilities module ──────────────────────────────────
add_library(pulse_utilities INTERFACE)
target_include_directories(pulse_utilities INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(pulse_utilities INTERFACE pulse_math pulse_memory)

# ── Jobs module ────────────────────────────────────────
add_library(pulse_jobs INTERFACE)
target_include_directories(pulse_jobs INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(pulse_jobs INTERFACE pulse_math pulse_memory pulse_utilities)

# Platform thread library
find_package(Threads REQUIRED)
target_link_libraries(pulse_jobs INTERFACE Threads::Threads)

# ── Main engine library (will grow with each module) ───
add_library(pulse STATIC
    src/math/math_stub.cpp
    src/memory/memory_stub.cpp
    src/utilities/utilities_stub.cpp
    src/utilities/profiler.cpp
    src/jobs/job_system.cpp
)
target_link_libraries(pulse PUBLIC pulse_math pulse_memory pulse_utilities pulse_jobs)

# ── Tests ──────────────────────────────────────────────
option(PULSE_BUILD_TESTS "Build Pulse unit tests" ON)
if(PULSE_BUILD_TESTS)
    add_executable(pulse_math_tests tests/math/test_math.cpp)
    target_link_libraries(pulse_math_tests PRIVATE pulse_math)

    add_executable(pulse_memory_tests tests/memory/test_memory.cpp)
    target_link_libraries(pulse_memory_tests PRIVATE pulse_memory)

    add_executable(pulse_utilities_tests tests/utilities/test_utilities.cpp)
    target_link_libraries(pulse_utilities_tests PRIVATE pulse_utilities)

    add_executable(pulse_jobs_tests tests/jobs/test_jobs.cpp src/jobs/job_system.cpp)
    target_link_libraries(pulse_jobs_tests PRIVATE pulse_jobs)
endif()

# ── Benchmarks ─────────────────────────────────────────
option(PULSE_BUILD_BENCHMARKS "Build Pulse benchmarks" ON)
if(PULSE_BUILD_BENCHMARKS)
    add_executable(pulse_math_bench benchmarks/math_bench.cpp)
    target_link_libraries(pulse_math_bench PRIVATE pulse_math)

    add_executable(pulse_memory_bench benchmarks/memory_bench.cpp)
    target_link_libraries(pulse_memory_bench PRIVATE pulse_memory)

    add_executable(pulse_utilities_bench benchmarks/utilities_bench.cpp)
    target_link_libraries(pulse_utilities_bench PRIVATE pulse_utilities)

    add_executable(pulse_jobs_bench benchmarks/jobs_bench.cpp src/jobs/job_system.cpp)
    target_link_libraries(pulse_jobs_bench PRIVATE pulse_jobs)
endif()
