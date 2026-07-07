cmake_minimum_required(VERSION 3.14)
project(Phase2_Operators LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 优化级别（Phase 2 重正确性，用 -O2 就够了）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================
# 算子库（静态库，供测试和后续阶段链接）
# ============================================================
add_library(operators STATIC
    src/operators.cpp
)
target_include_directories(operators PUBLIC include)

# ============================================================
# 链接 Phase 1 的 GEMM（复用）
# ============================================================
# 假设 Phase 1 的 gemm 源文件在 ../phase1_gemm/src/
add_library(gemm_phase1 STATIC
    ${CMAKE_SOURCE_DIR}/../phase1_gemm/src/gemm_naive.cpp
    ${CMAKE_SOURCE_DIR}/../phase1_gemm/src/gemm_tiled.cpp
)
target_include_directories(gemm_phase1 PUBLIC
    ${CMAKE_SOURCE_DIR}/../phase1_gemm/src
)

# ============================================================
# 单元测试
# ============================================================
find_package(GTest REQUIRED)

add_executable(test_operators
    tests/test_operators.cpp
)
target_link_libraries(test_operators PRIVATE
    operators
    gemm_phase1
    GTest::GTest
    GTest::Main
)

# 注册到 CTest
enable_testing()
add_test(NAME operators_test COMMAND test_operators)
