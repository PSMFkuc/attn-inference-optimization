cmake_minimum_required(VERSION 3.16)
project(Phase2_Operators LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================
# Phase 1 GEMM sources (reuse, in-tree build)
# ============================================================
set(PHASE1_DIR ${CMAKE_SOURCE_DIR}/../phase1_gemm)

add_library(gemm_phase1 STATIC
    ${PHASE1_DIR}/src/gemm_naive.cpp
    ${PHASE1_DIR}/src/gemm_tiled.cpp
)
target_include_directories(gemm_phase1 PUBLIC ${PHASE1_DIR}/src)

# ============================================================
# Operators library
# ============================================================
add_library(operators STATIC
    src/operators.cpp
)
target_include_directories(operators PUBLIC include)
target_link_libraries(operators PRIVATE gemm_phase1)

# ============================================================
# Benchmark
# ============================================================
add_executable(bench_operators
    profiling/bench_operators.cpp
)
target_link_libraries(bench_operators PRIVATE operators gemm_phase1)

# ============================================================
# GoogleTest (local third_party, same as Phase 1)
# ============================================================
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(${PHASE1_DIR}/third_party/googletest
                     ${CMAKE_BINARY_DIR}/googletest
                     EXCLUDE_FROM_ALL)

    # Main test suite (13 tests)
    add_executable(test_operators
        tests/test_operators.cpp
    )
    target_link_libraries(test_operators PRIVATE
        operators
        gemm_phase1
        gtest_main
    )

    # NumPy reference alignment tests
    add_executable(test_operators_numpy
        tests/test_operators_numpy_ref.cpp
    )
    target_link_libraries(test_operators_numpy PRIVATE
        operators
        gemm_phase1
        gtest_main
    )

    enable_testing()
    include(GoogleTest)
    gtest_discover_tests(test_operators)
    gtest_discover_tests(test_operators_numpy)
endif()
