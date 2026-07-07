cmake_minimum_required(VERSION 3.16)
project(Phase1_GEMM LANGUAGES CXX)

# 强制 C++17：现代 C++ 标准，带来结构化绑定、std::optional 等
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 默认 Release 模式。性能项目必须 Release，Debug 没优化测不出真实性能。
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# -O3: 激进优化（循环展开、向量化、内联）
# -march=native: 针对当前 CPU 生成最优指令（含 AVX2/AVX-512）
# 注意：跨机器分发时不要用 native，但本地开发用它最好
# Windows MSVC 不认 -O3/-march=native，用 /O2 + /arch:AVX2
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /arch:AVX2")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
endif()

# GoogleTest — 使用本地 third_party 源码（无需网络）
option(BUILD_TESTS "Build unit tests (requires GoogleTest)" ON)
if(BUILD_TESTS)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/googletest ${CMAKE_BINARY_DIR}/googletest EXCLUDE_FROM_ALL)
endif()

# 头文件搜索路径
include_directories(${CMAKE_SOURCE_DIR}/src)

# 线程库（std::thread、OpenMP 需要）
find_package(Threads REQUIRED)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP enabled")
else()
    message(WARNING "OpenMP not found, parallel GEMM will not compile")
endif()

# --- benchmark 可执行文件 ---
add_executable(bench_gemm
    profiling/bench_gemm.cpp
    src/gemm_naive.cpp
    src/gemm_tiled.cpp
    src/gemm_simd.cpp
    src/gemm_simd_unroll.cpp
    src/gemm_parallel.cpp
)
target_link_libraries(bench_gemm PRIVATE Threads::Threads OpenMP::OpenMP_CXX)

# --- 单元测试（需要 BUILD_TESTS=ON）---
if(BUILD_TESTS)
    add_executable(test_gemm
        tests/test_gemm.cpp
        src/gemm_naive.cpp
        src/gemm_tiled.cpp
        src/gemm_simd.cpp
        src/gemm_simd_unroll.cpp
        src/gemm_parallel.cpp
    )
    target_link_libraries(test_gemm PRIVATE gtest_main OpenMP::OpenMP_CXX)
    enable_testing()
    include(GoogleTest)
    gtest_discover_tests(test_gemm)
endif()
