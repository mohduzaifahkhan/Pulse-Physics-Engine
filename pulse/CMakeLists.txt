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

# ── Main engine library (will grow with each module) ───
add_library(pulse STATIC
    src/math/math_stub.cpp
)
target_link_libraries(pulse PUBLIC pulse_math)

# ── Tests ──────────────────────────────────────────────
option(PULSE_BUILD_TESTS "Build Pulse unit tests" ON)
if(PULSE_BUILD_TESTS)
    add_executable(pulse_math_tests tests/math/test_math.cpp)
    target_link_libraries(pulse_math_tests PRIVATE pulse_math)
endif()

# ── Benchmarks ─────────────────────────────────────────
option(PULSE_BUILD_BENCHMARKS "Build Pulse benchmarks" ON)
if(PULSE_BUILD_BENCHMARKS)
    add_executable(pulse_math_bench benchmarks/math_bench.cpp)
    target_link_libraries(pulse_math_bench PRIVATE pulse_math)
endif()
