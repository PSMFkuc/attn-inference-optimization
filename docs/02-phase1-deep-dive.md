# Phase 1 深度教学：CPU GEMM 优化与架构分析

> 阅读这份文档的方式：不要跳读。每一节我都设计了"思考一下"的停顿点，先自己想，再看答案。这是建立直觉的关键。

---

## 第 0 步：先搞清楚我们在算什么

矩阵乘法 C = A × B，其中 A 是 M×K，B 是 K×N，C 是 M×N。

```
        K                    N                    N
    ┌─────────┐          ┌─────────┐          ┌─────────┐
  M │   A     │    ×    K │   B     │    =    M │   C     │
    │         │            │         │            │         │
    └─────────┘            └─────────┘            └─────────┘
```

C 的每个元素 `C[i][j]` = A 的第 i 行 和 B 的第 j 列 的点积：

```
C[i][j] = Σ (A[i][k] × B[k][j])   for k = 0..K-1
```

就这一行公式，整个 Phase 1 围绕它转。

**思考一下**：计算一个 C[i][j] 需要访问 A 的多少个元素？访问 B 的多少个？这些访问在内存里是怎么排布的？

> 答案：访问 A 的第 i 行共 K 个元素（连续内存，友好），访问 B 的第 j 列共 K 个元素（**跨行访问，每跳一行就是跳 N 个 float = 4N 字节，cache 不友好**）。这就是 naive 实现慢的根本原因之一。

---

## 第 1 步：理解 CPU 内存层级（这是 Phase 1 的灵魂）

### 1.1 为什么不直接用内存？

CPU 算得极快（GHz 级，每秒几十亿次操作），但内存（DRAM）慢得令人发指——访问一次主存要 ~100 纳秒，这段时间 CPU 能执行几百条指令。如果每个数据都要从主存读，CPU 99% 的时间在等数据。

解决方案：**缓存（Cache）**——在 CPU 和主存之间放几层更小、更快、更贵的存储。

```
速度  容量       层级
快 ↑   小 ↑    ┌─────────────┐
            │  寄存器      │  ← CPU 内部，几十个，每个核独有
            │  (Registers) │    延迟 ~0.3ns，容量 ~几百字节
            ├─────────────┤
            │  L1 Cache    │  ← 每核独有，分指令/数据缓存
            │              │    延迟 ~1ns，容量 ~32KB
            ├─────────────┤
            │  L2 Cache    │  ← 每核或几核共享
            │              │    延迟 ~4ns，容量 ~256KB-1MB
            ├─────────────┤
            │  L3 Cache    │  ← 全核共享
            │  (LLC)       │    延迟 ~12ns，容量 ~几MB-几十MB
            ├─────────────┤
            │  主存 DRAM   │  ← 所有核共享
            │              │    延迟 ~100ns，容量 ~几GB-几十GB
慢 ↓   大 ↓    └─────────────┘
```

**关键事实**：数据在缓存里 = 快；数据不在缓存里（cache miss）= 要去下层取，慢 10-100 倍。

### 1.2 缓存是怎么工作的？为什么"访问模式"决定速度？

缓存以 **cache line**（通常 64 字节）为单位搬运。当你读 `A[i][k]` 这一个 float（4字节），实际上它周围的 16 个 float 会被一起搬到 L1。**如果你接下来正好用到这 16 个，就赚了**（cache hit）；如果跳到很远的地方，就白搬了（cache miss）。

这就是为什么：
- **按行遍历矩阵**：连续访问，每次搬 64 字节用满，cache 命中率高 → 快
- **按列遍历矩阵**：每次跳一整行，搬来的 64 字节大部分用不上 → cache 命中率低 → 慢

**思考一下**：回顾第 0 步，naive GEMM 里访问 B 是按列访问的。这意味着什么？

> 答案：B 的访问会大量 cache miss，这是性能杀手。所有后续优化（转置、分块）本质上都在解决"怎么让 B 的访问也变成连续的"。

### 1.3 寄存器：最快但最稀缺的资源

寄存器是 CPU 内部的存储，速度最快（0 延迟），但数量极少（x86-64 只有 16 个通用寄存器，加上 16-32 个 YMM/ZMM 向量寄存器）。

编译器会尽量把热点变量放寄存器里。但寄存器不够时，变量会被"溢出（spill）"到内存，每次读写都慢。**优化的高级技巧之一就是"减少同时存活的变量数，让寄存器够用"。**

---

## 第 2 步：环境搭建（CMake 项目模板）

我们用 C++ + CMake。这一步给你完整模板，每一行 CMake 都解释。

### 2.1 目录结构

```
phase1_gemm/
├── CMakeLists.txt          ← 构建配置
├── src/
│   ├── gemm_naive.cpp      ← 第一个实现（naive 三重循环）
│   ├── gemm_tiled.cpp      ← 分块优化版本
│   ├── gemm_simd.cpp       ← SIMD 向量化版本（进阶）
│   └── timer.h             ← 计时工具
├── tests/
│   └── test_gemm.cpp       ← 正确性测试（对齐 NumPy）
├── profiling/
│   └── bench_gemm.cpp      ← 性能基准测试
└── README.md
```

### 2.2 CMakeLists.txt 逐行解读

```cmake
cmake_minimum_required(VERSION 3.16)
project(Phase1_GEMM LANGUAGES CXX)

# 强制 C++17。C++17 带来结构化绑定、std::optional，写现代 C++ 必须。
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 默认 Release（-O3 优化）。性能项目必须 Release，Debug 没优化测不出真实性能。
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# "-march=native" 让编译器针对当前 CPU 生成最优指令（含 AVX）。
# 注意：跨机器分发时不要用 native，本地开发用它最好。
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# FetchContent 拉取 GoogleTest（比 git submodule 干净）。
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(bench_gemm
    profiling/bench_gemm.cpp
    src/gemm_naive.cpp
)
find_package(Threads REQUIRED)
target_link_libraries(bench_gemm PRIVATE Threads::Threads)

add_executable(test_gemm tests/test_gemm.cpp src/gemm_naive.cpp)
target_link_libraries(test_gemm PRIVATE gtest_main)
include(GoogleTest)
gtest_discover_tests(test_gemm)
```

### 2.3 为什么这么配置

- **为什么 `-O3` 而不是 `-O2`**：`-O3` 会激进做循环展开、向量化。我们要测"自己手写优化"的效果，但也要让编译器做它能做的。
- **为什么 `-march=native`**：让编译器用上你 CPU 的 AVX2/AVX-512 指令集。不加它，编译器只用最基础的 SSE，SIMD 优化发挥不出来。
- **为什么分离 bench 和 test**：测试要快（小矩阵），benchmark 要慢（大矩阵、多次跑）。混在一起互相干扰。

---

## 第 3 步：Naive 实现——每一行代码讲清楚

这是你的起点。**不要跳过 naive 直接优化**，它是你的 baseline，没有 baseline 无法衡量优化效果。

### 3.1 timer.h —— 计时工具

```cpp
#ifndef TIMER_H
#define TIMER_H

#include <chrono>

// RAII 计时器：构造时记录开始，析构时打印耗时。
// 为什么用 chrono 不用 clock()？chrono 高精度、单调时钟，不受系统时间调整影响。
class Timer {
public:
    explicit Timer(const char* label) : label_(label) {
        start_ = std::chrono::high_resolution_clock::now();
    }
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        printf("[%s] %.2f ms\n", label_, us / 1000.0);
    }
private:
    const char* label_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

#endif
```

### 3.2 gemm_naive.h —— 接口

```cpp
#ifndef GEMM_NAIVE_H
#define GEMM_NAIVE_H

#include <vector>

// C = A × B
// A: M×K (行主序，A[i][k] 在 A[i*K + k])
// B: K×N
// C: M×N
// 为什么用一维 vector 而不是 vector<vector>？
//   vector<vector> 每行单独分配，内存不连续，cache 友好性差，分配慢。
//   一维数组 + 索引计算是高性能代码的标准做法。
void gemm_naive(int M, int N, int K,
                const std::vector<float>& A,
                const std::vector<float>& B,
                std::vector<float>& C);

#endif
```

### 3.3 gemm_naive.cpp —— 逐行解读

```cpp
#include "gemm_naive.h"

void gemm_naive(int M, int N, int K,
                const std::vector<float>& A,
                const std::vector<float>& B,
                std::vector<float>& C) {
    // 初始化 C 为 0。后面用 += 累加，不清零结果就错。
    // resize 在已是大小时不重新分配，比 clear()+resize 高效。
    C.resize(M * N, 0.0f);

    // 三重循环，顺序 ijk。固定 C 一个元素，遍历 k 累加。
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // acc 放寄存器里（编译器会做），减少对 C[i*N+j] 的反复读写。
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                // A[i*K+k]：k 递增时连续访问 → cache 友好 ✅
                // B[k*N+j]：k 递增时跨行，每次跳 4N 字节 → cache 不友好 ❌
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;  // 一次写回，避免循环内反复写
        }
    }
}
```

**这一行代码是整个 Phase 1 的核心矛盾**：
```cpp
acc += A[i * K + k] * B[k * N + j];
```
- `A[i*K+k]`：k 递增时连续访问 → cache 友好 ✅
- `B[k*N+j]`：k 递增时跨行访问，每次跳 4N 字节 → cache 不友好 ❌

**所有后续优化，本质都是在解决 B 的访问不连续问题。**

### 3.4 为什么循环顺序重要？

GEMM 有 6 种循环顺序：ijk、ikj、jik、jki、kij、kji。**数学结果相同，性能差 3-10 倍。**

| 顺序 | A 访问 | B 访问 | C 访问 | 评价 |
|---|---|---|---|---|
| ijk | 行连续 ✅ | 列跳跃 ❌ | 单点写 ✅ | 最直观，B 不友好 |
| ikj | 行连续 ✅ | 行连续 ✅ | 反复写 ⚠️ | A、B 都友好，C 靠寄存器缓解 |
| jki | 列跳跃 ❌ | 行连续 ✅ | 列跳跃 ❌ | 最差 |

**经验法则**：让两个矩阵的访问都连续，宁愿让第三个稍差。读源是大头，输出可放寄存器/分块缓解。

`ikj` 通常比 `ijk` 快 2-3 倍，**不写新算法、只调顺序**就拿到。

```cpp
void gemm_ikj(int M, int N, int K, ...) {
    C.assign(M * N, 0.0f);
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a_ik = A[i * K + k];  // 提到内层循环外，减少寻址
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];  // A 不变，B 和 C 都连续
            }
        }
    }
}
```

---

## 第 4 步：先测量——建立 Baseline

**没有 baseline 之前，任何优化都是盲目的。**

### 4.1 bench_gemm.cpp

```cpp
#include "gemm_naive.h"
#include "timer.h"
#include <cstdio>
#include <random>

// 计算 GFLOPS。GEMM 浮点运算量 = 2*M*N*K（每个元素 K 次乘加 = 2K 次浮点运算）。
double compute_gflops(int M, int N, int K, double ms) {
    double flops = 2.0 * M * N * K;
    return flops / (ms / 1000.0) / 1e9;
}

int main() {
    std::mt19937 rng(42);  // 固定种子，可复现
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    int sizes[] = {64, 128, 256, 512, 1024};

    for (int sz : sizes) {
        int M = sz, N = sz, K = sz;
        std::vector<float> A(M * K), B(K * N), C(M * N);
        for (auto& x : A) x = dist(rng);
        for (auto& x : B) x = dist(rng);

        // 预热：第一次有冷启动（page fault、指令 cache miss），不能算。
        gemm_naive(M, N, K, A, B, C);

        double best_ms = 1e9;
        for (int trial = 0; trial < 5; ++trial) {
            auto start = std::chrono::high_resolution_clock::now();
            gemm_naive(M, N, K, A, B, C);
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() / 1000.0;
            best_ms = std::min(best_ms, ms);  // 取最小值，最干净
        }
        printf("size=%d  naive=%.2f ms  %.2f GFLOPS\n",
               sz, best_ms, compute_gflops(sz,sz,sz,best_ms));
    }
    return 0;
}
```

**关键工程细节**：
- **预热（warmup）**：第一次数据不在任何 cache，时间偏长，必须跳过。
- **多次测量取最小值**：不取平均！系统有干扰（其他进程、中断），最小值最接近"干净性能"。工业 benchmark 标准做法。
- **测多个尺寸**：小矩阵在 cache 内，大矩阵 cache miss，对比能看出 cache 影响。

### 4.2 你会观察到的现象（预测）

| 尺寸 | 时间 | GFLOPS | 现象 |
|---|---|---|---|
| 64 | 几 ms | 低 | 还行，在 L2 里 |
| 256 | 明显变慢 | 更低 | A(256KB) 开始溢出 L2 |
| 512 | 大幅下降 | 很低 | B 访问大量 cache miss |
| 1024 | 非常慢 | 极低 | 三个矩阵都远超 L2 |

**这个曲线就是 baseline。每做一次优化，重跑，对比提升。**

---

## 第 5 步：第一个优化——循环顺序（不写新代码）

把 ijk 改成 ikj，重测，2-3 倍提升。

**这一步教会你最重要的一课：有时候最好的优化是"重新组织数据访问模式"，连算法都不用改。**

记录：`naive_ijk: 1024×1024 = 1200ms, 1.8 GFLOPS` → `naive_ikj: 1024×1024 = 450ms, 4.8 GFLOPS`。

---

## 第 6 步：第二个优化——分块（Tiling / Blocking）

### 6.1 为什么分块？

回到核心矛盾：B 的列访问 cache 不友好。

但如果把矩阵切成小块呢？把 1024×1024 切成 64×64 的小块，每次只算一个块。小块能完全放进 L1/L2，访问 B 的小块时即使按列，也只在小块范围内跳，不会 miss 到主存。

**分块的核心思想：让工作集（working set）小于 cache 大小。**

### 6.2 分块实现的思考过程

分块 GEMM 循环结构变 6 层（3 层块 + 3 层块内）：

```
for ii in 0..M step BM:        // 块外：A 的行块
  for jj in 0..N step BN:      // 块外：B 的列块
    for kk in 0..K step BK:    // 块外：K 维块
      for i in ii..ii+BM:      // 块内：算小块
        for j in jj..jj+BN:
          for k in kk..kk+BK:
            C[i,j] += A[i,k] * B[k,j]
```

**关键参数：BM, BN, BK 怎么选？**

原则：让一小块的工作集塞进目标 cache 层。

- 三个小块 A(BM×BK) + B(BK×BN) + C(BM×BN)，都是 float。
- 目标塞进 L1（32KB）：3 × BM×BK×4 < 32000 → BM×BK < 2600。
- 常见选择：BM=BN=BK=32 → 3×32×32×4 = 12KB，留余量，安全。

这是经验值，**最终靠实验测**——不同 CPU cache 大小不同，最优 block size 也不同。这就是为什么需要 profiling。

### 6.3 分块代码

```cpp
#include "gemm_tiled.h"
#include <algorithm>

#define BM 32
#define BN 32
#define BK 32

void gemm_tiled(int M, int N, int K,
                const std::vector<float>& A,
                const std::vector<float>& B,
                std::vector<float>& C) {
    C.assign(M * N, 0.0f);

    // 块外三层
    for (int ii = 0; ii < M; ii += BM) {
        for (int jj = 0; jj < N; jj += BN) {
            for (int kk = 0; kk < K; kk += BK) {

                // 块内三层。用 ikj 顺序——块内也要最优顺序！
                // 块外顺序决定 cache 友好性，块内顺序决定寄存器利用。
                int i_end = std::min(ii + BM, M);  // 边界处理
                int k_end = std::min(kk + BK, K);
                int j_end = std::min(jj + BN, N);

                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        float a_ik = A[i * K + k];  // 提到最内层外
                        for (int j = jj; j < j_end; ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}
```

**为什么分块会快？**

当 ii,jj,kk 固定时，内层三重循环访问：
- A 的 [ii..ii+BM] × [kk..kk+BK] 小块 = 32×32×4 = 4KB → 在 L1 ✅
- B 的 [kk..kk+BK] × [jj..jj+BN] 小块 = 4KB → 在 L1 ✅
- C 的 [ii..ii+BM] × [jj..jj+BN] 小块 = 4KB → 在 L1 ✅

12KB 总共，远小于 L1 的 32KB。内层 cache miss 几乎为零。

**实测效果**：1024×1024 上，分块通常比 ikj naive 再快 2-4 倍。

### 6.4 思考题

**为什么不是 block size 越大越好？**

> 块越大，单块计算越多，cache 利用率越高——看起来越大越好。但块太大会溢出 cache，反而 miss。最优值在"塞满但不超过 cache"的临界点。这就是为什么要 profiling——理论算出范围，实测确定最优值。

---

## 第 7 步：第三个优化——SIMD 向量化（进阶，可选）

### 7.1 什么是 SIMD？

SIMD = Single Instruction Multiple Data。一条指令同时处理多个数据。

```
普通（标量）：    4 条指令算 4 个加法
  add a0, b0 → c0
  add a1, b1 → c1
  ...

SIMD（向量）：    1 条指令算 8 个加法（AVX2，256位 = 8个float）
  vaddps ymm_a, ymm_b → ymm_c   （同时算 8 个 float）
```

AVX2 寄存器 256 位 = 8 个 float，理论 8 倍加速。AVX-512 是 16 个 float。

### 7.2 两种用 SIMD 的方式

**方式 1：自动向量化（auto-vectorization）**
你写普通循环，加 `-O3 -march=native`，编译器帮你转。90% 简单循环能自动向量化。**先用这个。**

判断是否向量化：GCC 用 `-fopt-info-vec` 看报告"loop vectorized"。

**方式 2：手写 intrinsics**
用 `<immintrin.h>` 显式控制 SIMD 寄存器。性能更可控，但代码难读。

```cpp
#include <immintrin.h>  // AVX2 intrinsics

void gemm_simd_inner(const float* A, const float* B, float* C,
                     int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            // _mm256_set1_ps：标量广播到 8 个 lane（a_ik 复制 8 份）
            __m256 a_ik = _mm256_set1_ps(A[i * K + k]);
            for (int j = 0; j < N; j += 8) {  // 每次 8 个 j
                // _mm256_loadu_ps：加载 8 个连续 float（u=不要求对齐）
                __m256 b = _mm256_loadu_ps(&B[k * N + j]);
                __m256 c = _mm256_loadu_ps(&C[i * N + j]);
                // _mm256_fmadd_ps：融合乘加 a*b+c（FMA 指令，比分开快）
                c = _mm256_fmadd_ps(a_ik, b, c);
                _mm256_storeu_ps(&C[i * N + j], c);
            }
        }
    }
}
```

**为什么 `j += 8`？** AVX2 一次 8 个 float，步进 8 正好填满 256 位寄存器。N 必须是 8 的倍数，否则要处理"尾巴"（remainder loop）。

### 7.3 SIMD 的坑

1. **对齐**：`_mm256_load_ps` 要求 32 字节对齐，不对齐 segfault。用 `_mm256_loadu_ps`（unaligned）更安全但稍慢。生产代码精心对齐内存以用 aligned 版本。
2. **数据依赖**：连续 FMA 形成依赖链（前一条 c 是后一条输入），流水线停顿。解决：**循环展开 + 多累加器**（acc0, acc1, acc2... 交替用，打破依赖链）。
3. CPU SIMD "宽度有限"，GPU 是"成千上万线程"，思路不同。

---

## 第 8 步：性能分析（Profiling）——Proposal 硬性要求

Proposal 要求："cache miss rate analysis, performance profiling data"。

### 8.1 Linux 上（推荐 WSL2）

```bash
# perf stat：统计硬件事件
perf stat -e cache-misses,cache-references,L1-dcache-load-misses,LLC-load-misses ./bench_gemm

# 输出类似：
# 1,234,567  cache-misses
# 9,876,543  cache-references
#    12.5%   miss rate
```

**Cachegrind（精确模拟）**：
```bash
valgrind --tool=cachegrind ./bench_gemm
# 输出每层 cache 命中/miss 数，非常详细
```

### 8.2 Windows 上（替代方案）

Windows 没有 perf。两个选择：
1. **用 WSL2**（强烈推荐，工具链完整）。
2. **退而求其次**：用 `__rdtsc()`（读 CPU 时间戳计数器）做精确计时，cache miss 用理论分析 + 性能对比间接推断（"分块后快了，说明 cache miss 减少了"）。

```cpp
#include <intrin.h>
uint64_t tsc = __rdtsc();  // 读时间戳，CPU 周期级精度
// ... do work ...
uint64_t elapsed = __rdtsc() - tsc;
```

### 8.3 性能日志模板

```markdown
## 性能日志：1024×1024 GEMM

| 版本 | 时间(ms) | GFLOPS | L1 miss率 | LLC miss率 | 相比baseline |
|---|---|---|---|---|---|
| naive ijk | 1200 | 1.8 | 35% | 12% | 1.0x |
| naive ikj | 450 | 4.8 | 18% | 6% | 2.7x |
| tiled 32x32 | 180 | 12.0 | 3% | 1% | 6.7x |
| tiled + SIMD | 95 | 22.6 | 3% | 1% | 12.6x |
| OpenBLAS (参考) | 25 | 85.9 | - | - | 48x |
```

**最后一行很重要**：始终和业界最优（OpenBLAS/MKL）对比。目标是"理解差距为什么存在"，不是超越它们（它们用汇编级优化 + 多线程 + Strassen 算法）。

---

## 第 9 步：正确性测试

每个优化版本必须验证正确。对齐 NumPy/PyTorch。

### 9.1 C++ 测试

```cpp
#include <gtest/gtest.h>
#include "gemm_naive.h"

TEST(GemmTest, Correctness) {
    int M = 64, N = 64, K = 64;
    std::vector<float> A(M*K), B(K*N), C(M*N), C_ref(M*N);

    // 用固定数据填充（不要全 0 或全 1，测不出边界 bug）
    for (int i = 0; i < M*K; ++i) A[i] = (float)(i % 7) * 0.1f;
    for (int i = 0; i < K*N; ++i) B[i] = (float)(i % 5) * 0.1f;

    gemm_naive(M, N, K, A, B, C);

    // 参考实现（最朴素 ijk，ground truth）
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0;
            for (int k = 0; k < K; ++k)
                acc += A[i*K+k] * B[k*N+j];
            C_ref[i*N+j] = acc;
        }

    // 对比，容差 1e-4（float 精度有限，不能用 ==）
    for (int i = 0; i < M*N; ++i) {
        EXPECT_NEAR(C[i], C_ref[i], 1e-4f)
            << "mismatch at index " << i;
    }
}
```

**为什么容差用 1e-4 而不是 Proposal 说的 1e-5？**

> Proposal 的 1e-5 针对的是 float32 算子。GEMM 累积误差大，1e-4 更现实。double 可以到 1e-7。**容差要匹配数据类型和运算复杂度。**

### 9.2 和 PyTorch 对齐（更权威）

用 Python 生成参考答案，C++ 读进来对比：

```python
# scripts/gen_reference.py
import numpy as np
np.random.seed(42)
A = np.random.randn(64, 64).astype(np.float32)
B = np.random.randn(64, 64).astype(np.float32)
C = A @ B  # NumPy 参考答案
A.tofile("ref_A.bin")  # 二进制写，C++ 直接读
B.tofile("ref_B.bin")
C.tofile("ref_C.bin")
```

C++ 读二进制对比：
```cpp
std::ifstream fa("ref_A.bin", std::ios::binary);
fa.read(reinterpret_cast<char*>(A.data()), M*K*sizeof(float));
// ... 同理读 B、C_ref ...
// 跑你的 gemm，对比 C 和 C_ref，容差 1e-3（numpy 用更优算法，差异正常）
```

---

## 第 10 步：Phase 1 的"思考总结"

做完 Phase 1，你应该能回答这些问题（面试常问）：

1. **naive GEMM 为什么慢？** 答：B 的列访问 cache 不友好。
2. **ikj 比 ijk 快在哪？** 答：让 A、B 都按行访问，C 靠寄存器缓解。
3. **分块为什么快？** 答：让工作集小于 cache，减少 cache miss。
4. **block size 怎么选？** 答：让小块工作集塞进目标 cache 层，留余量。
5. **你的实现和 OpenBLAS 差多少倍？为什么？** 答：差 ~4-8 倍。OpenBLAS 用了：手写汇编 kernel、多线程、寄存器分块（micro-kernel）、Strassen 算法（大矩阵）。
6. **Amdahl's Law 怎么影响你的优化策略？** 答：先 profile 找热点，优化占比大的部分。如果 cache miss 占 80% 时间，先解决 cache miss 而不是 SIMD。

**如果这 6 个问题你都能流畅回答，Phase 1 就过关了。**

---

## 下一步

完成 Phase 1 后：
1. 把性能日志整理进 `docs/performance-log.md`
2. 打开 `docs/03-phase2-operators.md`，我们进入 Attention 算子库

**Phase 1 的方法论（naive → 测量 → 优化 → 再测量）会贯穿后面所有阶段，一定要吃透。**
