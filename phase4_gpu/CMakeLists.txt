cmake_minimum_required(VERSION 3.16)
project(Phase4_CPU_Acceleration LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /arch:AVX2")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
endif()

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

set(PHASE1_DIR ${CMAKE_SOURCE_DIR}/../phase1_gemm)
set(PHASE2_DIR ${CMAKE_SOURCE_DIR}/../phase2_operators)
set(PHASE3_DIR ${CMAKE_SOURCE_DIR}/../phase3_graph)

add_library(gemm_phase1 STATIC
    ${PHASE1_DIR}/src/gemm_naive.cpp
    ${PHASE1_DIR}/src/gemm_tiled.cpp
    ${PHASE1_DIR}/src/gemm_parallel.cpp
)
target_include_directories(gemm_phase1 PUBLIC ${PHASE1_DIR}/src)
target_link_libraries(gemm_phase1 PUBLIC OpenMP::OpenMP_CXX Threads::Threads)

add_library(operators STATIC
    ${PHASE2_DIR}/src/operators.cpp
)
target_include_directories(operators PUBLIC ${PHASE2_DIR}/include)
target_link_libraries(operators PUBLIC gemm_phase1)

add_library(graph_core STATIC
    ${PHASE3_DIR}/src/graph.cpp
    ${PHASE3_DIR}/src/json_parser.cpp
    ${PHASE3_DIR}/src/memory_pool.cpp
)
target_include_directories(graph_core PUBLIC ${PHASE3_DIR}/include)

add_library(phase4_cpu_accel STATIC
    src/cpu_parallel_backend.cpp
    src/parallel_graph_executor.cpp
    src/quantization.cpp
)
target_include_directories(phase4_cpu_accel PUBLIC
    include
    ${PHASE1_DIR}/src
    ${PHASE2_DIR}/include
    ${PHASE3_DIR}/include
)
target_link_libraries(phase4_cpu_accel PUBLIC
    graph_core
    operators
    gemm_phase1
    OpenMP::OpenMP_CXX
    Threads::Threads
)

add_executable(test_phase4_cpu
    tests/test_phase4_cpu.cpp
)
target_link_libraries(test_phase4_cpu PRIVATE phase4_cpu_accel)

add_executable(bench_phase4_cpu
    profiling/bench_phase4_cpu.cpp
)
target_link_libraries(bench_phase4_cpu PRIVATE phase4_cpu_accel)

enable_testing()
add_test(NAME test_phase4_cpu COMMAND test_phase4_cpu)
