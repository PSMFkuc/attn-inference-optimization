cmake_minimum_required(VERSION 3.16)
project(Phase3_Graph LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================
# Paths to Phase 1 and Phase 2 (reuse libraries)
# ============================================================
set(PHASE1_DIR ${CMAKE_SOURCE_DIR}/../phase1_gemm)
set(PHASE2_DIR ${CMAKE_SOURCE_DIR}/../phase2_operators)

# Phase 1 GEMM library
add_library(gemm_phase1 STATIC
    ${PHASE1_DIR}/src/gemm_naive.cpp
    ${PHASE1_DIR}/src/gemm_tiled.cpp
)
target_include_directories(gemm_phase1 PUBLIC ${PHASE1_DIR}/src)

# Phase 2 operators library
add_library(operators STATIC
    ${PHASE2_DIR}/src/operators.cpp
)
target_include_directories(operators PUBLIC ${PHASE2_DIR}/include)
target_link_libraries(operators PRIVATE gemm_phase1)

# ============================================================
# Graph engine library
# ============================================================
add_library(graph_engine STATIC
    src/graph.cpp
    src/json_parser.cpp
    src/executor.cpp
    src/memory_pool.cpp
)
target_include_directories(graph_engine PUBLIC include)
target_link_libraries(graph_engine PRIVATE operators gemm_phase1)

# ============================================================
# Test 1: Graph structure + JSON parsing + Topological sort
# ============================================================
add_executable(test_graph
    tests/test_graph.cpp
)
target_link_libraries(test_graph PRIVATE graph_engine)

# ============================================================
# Test 2: End-to-end execution (JSON -> Execute)
# ============================================================
add_executable(test_executor
    tests/test_executor.cpp
)
target_link_libraries(test_executor PRIVATE graph_engine operators gemm_phase1)

# ============================================================
# Test 3: Memory pool
# ============================================================
add_executable(test_memory_pool
    tests/test_memory_pool.cpp
)
target_link_libraries(test_memory_pool PRIVATE graph_engine operators gemm_phase1)
