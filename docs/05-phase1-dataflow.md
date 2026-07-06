# Phase 1 数据流全景：从一行公式到一份性能日志

> 这份文档换个视角。前面 `02-phase1-deep-dive.md` 是"每行代码做什么"，这份是"**文件之间怎么协作、数据怎么流动**"。
>
> 读完后你会理解：编译器如何把 5 个文件织成一个可执行程序、一个矩阵从生成到结果校验经历了哪些环节、每个文件在整个流水线里扮演什么角色。

---

## 全景图：Phase 1 的两条流水线

Phase 1 实际上有**两条独立的执行流水线**，共享同一组算子源码，但目的完全不同：

```
┌─────────────────────────────────────────────────────────────────┐
│                      编译期（CMake 编织）                        │
│   5 个 .h/.cpp 文件 ──编译──►  两个可执行文件                    │
└─────────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
   ┌─────────────────────┐    ┌─────────────────────────┐
   │ 流水线 A: 正确性验证  │    │ 流水线 B: 性能基准       │
   │   test_gemm          │    │   bench_gemm            │
   │   (小矩阵, 跑一次)    │    │   (多尺寸, 多次取最小)   │
   └─────────────────────┘    └─────────────────────────┘
              │                           │
              ▼                           ▼
        PASS / FAIL              性能日志(GFLOPS曲线)
```

**关键认知**：这两条流水线**不能合并**。
- 测试要快（小矩阵、一次跑完），目的是"验证算法对不对"
- Benchmark 要慢（大矩阵、多次跑），目的是"测出真实性能"
混在一起会互相干扰——小矩阵的数据进了 cache，影响大矩阵测量的干净性。

---

## 数据的一生：从一个矩阵的视角讲

让我们跟踪**一个 1024×1024 的矩阵 A**，从它在 benchmark 里被生成，到最后变成性能日志里的一行数字，经历了什么。

### 第 1 站：诞生（bench_gemm.cpp 第 35-37 行）

```cpp
std::mt19937 rng(42);                                      // 固定随机种子
std::uniform_real_distribution<float> dist(-1.0f, 1.0f);   // [-1,1] 均匀分布
// ...
std::vector<float> A(M * K), B(K * N), C(M * N);           // 一维数组，连续内存
for (auto& x : A) x = dist(rng);                           // 填随机数
```

**这里发生的事**：
- `std::vector<float>` 在堆上分配 `M*K*4` 字节 = 1024×1024×4 = 4MB 连续内存
- 数据填入 `[-1, 1]` 范围。为什么是这个范围？避免极端值导致 float 累加误差过大
- **A、B、C 都是一维的**，没有二维 vector。这是高性能代码的第一原则：**内存必须连续**

此时 A 在主存（DRAM）里，是一坨 4MB 的 float 数组。CPU 还没碰过它。

### 第 2 站：预热（bench_gemm.cpp 第 56 行）

```cpp
// 预热：第一次跑有冷启动（page fault、指令缓存未命中），不能算
func(M, N, K, A, B, C);
```

**这里发生的事**：调一次 GEMM，但结果扔掉。

为什么不直接测？因为第一次调用会触发：
1. **缺页中断（page fault）**：操作系统按需把虚拟内存页映射到物理内存，第一次访问每个页都要陷入内核
2. **指令缓存未命中**：GEMM 函数的机器指令第一次加载到 CPU 的指令 cache
3. **分支预测器冷启动**：CPU 还没学到这个函数的分支模式

这些都是"启动开销"，不是 GEMM 本身的性能。**预热让这些开销发生在测量之外。**

预热后，A 的部分数据可能已经在 L3 cache 里了——这没关系，因为真实场景下也是这样的（你不会每次都从冷启动开始）。

### 第 3 站：进入 GEMM 函数（src/gemm_naive.cpp）

数据被传给 `gemm_naive_ikj`：

```cpp
void gemm_naive_ikj(int M, int N, int K,
                    const std::vector<float>& A,    // ← A 在这里被引用传入
                    const std::vector<float>& B,
                    std::vector<float>& C) {
    C.assign(M * N, 0.0f);                          // 第 4 站：清零 C
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a_ik = A[i * K + k];              // 第 5 站：读 A 的一个元素
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j]; // 第 6 站：乘加，写 C
            }
        }
    }
}
```

**关键细节**：
- `const std::vector<float>& A` —— **引用传递**，没有拷贝。A 的 4MB 数据不会被复制，只是传了一个指针（vector 内部就是指针+长度+容量）
- 如果写成 `std::vector<float> A`（不带 &），会深拷贝 4MB，每次调用浪费几毫秒——这在 benchmark 里会严重污染数据

### 第 4 站：C 被清零（同一函数第 1 行）

```cpp
C.assign(M * N, 0.0f);
```

**这里发生的事**：
- `assign` 把 C 重置为 `M*N` 大小，全部填 0
- 为什么必须清零？因为后面是 `+=` 累加，如果 C 有上次的残留值，结果就错了
- `assign` 比 `resize` + `memset` 更高效：它合并了"调整大小"和"填值"两个操作
- 如果 C 之前已经是 M*N 大小，`assign` 不会重新分配内存，只覆盖数据

### 第 5 站：CPU 第一次读 A 的元素（cache miss 发生处）

```cpp
float a_ik = A[i * K + k];
```

**这一行是性能宇宙的中心**。让我们放大看发生了什么：

```
代码：A[i * K + k]
         │
         │  ① 地址计算：i*K + k，结果是数组下标
         │     乘以 4 字节得到内存地址
         ▼
    内存地址 0x7ff8_1234_5678
         │
         │  ② CPU 查 L1 cache：在吗？
         │     ── 不在（第一次访问这个地址）
         │     ③ 查 L2 cache：在吗？── 不在
         │     ④ 查 L3 cache：在吗？── 不在
         │     ⑤ 去 DRAM 取
         ▼
    DRAM 返回一个 cache line（64 字节 = 16 个 float）
         │
         │  ⑥ 这 16 个 float 被装入 L1
         │     同时 a_ik 这一个值返回给寄存器
         ▼
    寄存器 xmm0 = a_ik 的值
```

**关键点**：虽然代码只读一个 float，但内存系统**搬运了一整条 64 字节的 cache line**。这 16 个 float 里，**只有 1 个被立即使用，其余 15 个"免费"待在 L1 里**。

如果接下来的代码访问到这 15 个里的任何一个 → cache hit，极快。
如果跳到很远的地方 → 这 15 个白搬了，下次还要 miss。

**这就是为什么"访问模式"比"算法"更重要。**

### 第 6 站：内层循环——B 和 C 的访问（同一函数最内层）

```cpp
for (int j = 0; j < N; ++j) {
    C[i * N + j] += a_ik * B[k * N + j];
}
```

j 从 0 递增，**B 和 C 都按行连续访问**：
- `B[k*N + j]`：j++ 时地址 +4，连续 → 第 1 次访问搬来的 16 个 float，接下来 15 次都是 hit ✅
- `C[i*N + j]`：同上，连续 ✅

这就是 ikj 比 ijk 快的根本原因：**内层循环里所有访问都连续**。

但注意 `C[i*N + j] += ...` 是**读-改-写**：先读 C 的当前值，加完再写回。每次都要碰 C。好在 C 也在 L1 里，问题不大。

### 第 7 站：函数返回，数据流回 bench_gemm

```cpp
// bench_gemm.cpp
auto start = now();
gemm_naive_ikj(M, N, K, A, B, C);   // ← 数据在这里被处理
auto end = now();
double ms = elapsed_ms(start, end);  // 第 8 站：计时
```

函数返回后，C 里装着正确的乘积结果。A、B 没变（const 保证）。**数据所有权一直在 bench_gemm 手里**，GEMM 函数只是"借用"。

### 第 8 站：计时（src/timer.h）

```cpp
inline std::chrono::time_point<std::chrono::high_resolution_clock> now() {
    return std::chrono::high_resolution_clock::now();
}
inline double elapsed_ms(start, end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}
```

**这里发生的事**：
- `now()` 调用操作系统的单调时钟，返回一个时间点
- `elapsed_ms` 算两个时间点的差，转成毫秒
- 为什么用 `high_resolution_clock` 而不是 `clock()`？因为 `clock()` 测的是 CPU 时间（多线程会累加），我们要的是墙钟时间（真实经过的时间）

### 第 9 站：多次取最小值（bench_gemm.cpp 第 48-53 行）

```cpp
double best = 1e9;
for (int t = 0; t < trials; ++t) {
    auto start = now();
    func(M, N, K, A, B, C);
    auto end = now();
    double ms = elapsed_ms(start, end);
    best = std::min(best, ms);   // ← 取最小值，不是平均！
}
return best;
```

**为什么取最小值，不取平均？**

假设你跑 5 次：`[95, 102, 98, 150, 97]` ms。那个 150 是什么？是操作系统在这期间做别的事了（其他进程抢占、中断处理、GC 等）。这些**不是 GEMM 的性能**，是噪声。

最小值 95 ms 是"最少被干扰的那一次"，最接近 GEMM 的真实性能。**这是工业 benchmark 的标准做法。**

### 第 10 站：算 GFLOPS 并打印（bench_gemm.cpp 第 28-31 行）

```cpp
static double compute_gflops(int M, int N, int K, double ms) {
    double flops = 2.0 * M * N * K;           // 浮点运算总量
    return flops / (ms / 1000.0) / 1e9;       // 转 GFLOPS
}
```

**为什么是 2*M*N*K？**
- 每个 `C[i][j]` 要做 K 次乘法 + K 次加法 = 2K 次浮点运算
- 共 M×N 个 `C[i][j]`
- 总运算量 = 2×M×N×K

GFLOPS（Giga FLOPS = 十亿次浮点运算/秒）是跨尺寸可比的标准化指标。1024×1024 上：
- naive ijk: ~1.7 GFLOPS
- naive ikj: ~6.9 GFLOPS
- 分块: ~22.5 GFLOPS
- OpenBLAS: ~85 GFLOPS

同样 1024×1024，不同实现的 GFLOPS 差 50 倍——这就是优化的价值。

### 第 11 站：终端输出 → 性能日志

```
size     | ijk(ms) GFLOPS    | ikj(ms) GFLOPS    | tiled(ms) GFLOPS
64       | 0.12    2.18      | 0.05    5.24      | 0.04    6.55
1024     | 1280    1.68      | 312     6.89      | 95.4    22.53
```

这张表就是你 Phase 1 的**真实交付物**。Proposal 要求的"performance log documenting each optimization step"就是这个。

---

## 文件分工：5 个文件各自的角色

现在回头看，每个文件在流水线里扮演什么角色：

### `src/timer.h` —— 计时基础设施

```cpp
class Timer { /* RAII 析构打印 */ };     // 作用域计时（适合粗粒度）
inline double elapsed_ms(start, end);    // 手动计时（适合多次取最小）
inline now();                            // 时间点获取
```

**角色**：被 benchmark 和测试调用，本身不参与计算。是"工具人"。
**为什么不做成 .cpp？** 全是 inline 函数，放头文件避免链接开销，也是 C++ 性能代码惯例。

### `src/gemm_naive.h` —— 接口契约

```cpp
void gemm_naive_ijk(int M, int N, int K, const std::vector<float>& A, ...);
void gemm_naive_ikj(int M, int N, int K, const std::vector<float>& A, ...);
```

**角色**：声明"我提供哪些函数，签名长什么样"。**只有声明，没有实现。**
- 被调用方（bench_gemm.cpp、test_gemm.cpp）`#include` 它来知道函数签名
- 实现在 gemm_naive.cpp，编译期链接时绑定

这是 C++ 的"接口与实现分离"原则。好处：换实现不用改调用方代码。

### `src/gemm_naive.cpp` —— 算法实现

**角色**：流水线的"主车间"。矩阵 A/B/C 在这里被实际处理。

```cpp
void gemm_naive_ikj(...) {
    C.assign(M * N, 0.0f);        // ← 数据被改写
    for (...) for (...) for (...) {
        C[i * N + j] += a_ik * B[k * N + j];   // ← 核心计算
    }
}
```

**它接收的是引用**，所以它修改的就是调用方（bench/test）手里的那个 vector。没有拷贝、没有返回值——数据在原地被处理。

### `src/gemm_tiled.h / .cpp` —— 优化版实现

和 gemm_naive 同样的接口契约，但算法不同（6 层循环分块）。**调用方代码完全不用改**——这就是接口分离的好处。

bench_gemm.cpp 里只是多调一个函数指针：
```cpp
double ms_tiled = bench_tiled(M, N, K, A, B, C, 32, 32, 32);
```

### `tests/test_gemm.cpp` —— 正确性裁判

```cpp
TEST(GemmTest, IKJ_MatchesReference) {
    // 1. 准备输入数据
    std::vector<float> A(M*K), B(K*N), C_test(M*N), C_ref(M*N);
    // 2. 跑被测函数
    gemm_naive_ikj(M, N, K, A, B, C_test);
    // 3. 跑参考实现（独立的 ijk）
    reference_gemm(M, N, K, A, B, C_ref);
    // 4. 逐元素对比，容差 1e-4
    for (int i = 0; i < M*N; ++i) {
        EXPECT_NEAR(C_test[i], C_ref[i], 1e-4f);
    }
}
```

**角色**：流水线 A 的核心。它**不关心性能，只关心正确性**。
**为什么单独写一个 reference_gemm？** 因为不能让被测函数自己和自己比——如果 gemm_naive_ikj 有 bug，那它和它自己比永远"一致"。必须用**独立实现**的参考来对比。

### `profiling/bench_gemm.cpp` —— 性能裁判

```cpp
int main() {
    // 1. 准备多组尺寸的数据
    // 2. 对每个尺寸，预热 + 多次取最小值
    // 3. 算 GFLOPS，打印表格
}
```

**角色**：流水线 B 的核心。它**不关心正确性（假设已经测过），只关心性能**。
**为什么用函数指针？** `bench_once(..., void (*func)(...), ...)` 让同一个测量框架能测 ijk、ikj、tiled 三个版本，代码不重复。

---

## 编译期：CMake 怎么把 5 个文件织成 2 个可执行文件

数据流是运行时的事，但**运行之前**，CMake 要先把源文件织成可执行文件。

```cmake
# 织 test_gemm（流水线 A）
add_executable(test_gemm
    tests/test_gemm.cpp      ← 裁判
    src/gemm_naive.cpp       ← 被测实现1
    src/gemm_tiled.cpp       ← 被测实现2
)
target_link_libraries(test_gemm PRIVATE gtest_main)   ← GoogleTest 框架

# 织 bench_gemm（流水线 B）
add_executable(bench_gemm
    profiling/bench_gemm.cpp ← 裁判
    src/gemm_naive.cpp       ← 被测实现1
    src/gemm_tiled.cpp       ← 被测实现2
)
target_link_libraries(bench_gemm PRIVATE Threads::Threads)
```

**注意**：`gemm_naive.cpp` 和 `gemm_tiled.cpp` 被**两个可执行文件各自编译一次**。这是 CMake 的默认行为（不共享 object 文件）。对小项目无所谓，大项目会用 OBJECT library 优化。

`timer.h` 不在源文件列表里——因为它是 header-only（全是 inline），被 `#include` 进其他 .cpp 文件一起编译。

```
编译期数据流：

test_gemm.cpp ─┐
gemm_naive.cpp ─┼─► 编译 ─► 链接 ─► test_gemm.exe
gemm_tiled.cpp ─┤              （+ GoogleTest 库）
gtest_main.lib ─┘

bench_gemm.cpp ─┐
gemm_naive.cpp ─┼─► 编译 ─► 链接 ─► bench_gemm.exe
gemm_tiled.cpp ─┤              （+ 线程库）
timer.h (inline)┘
```

---

## 两条流水线的运行期对比

| 维度 | 流水线 A（test_gemm） | 流水线 B（bench_gemm） |
|---|---|---|
| 目的 | 验证算法对不对 | 测出真实性能 |
| 矩阵尺寸 | 64×64, 70×50（故意非整数倍，测边界） | 64→1024（从小到大，看 cache 影响） |
| 运行次数 | 每个测试 1 次 | 每个尺寸 1 次预热 + 5 次测量 |
| 关心什么 | `EXPECT_NEAR` 通过 | GFLOPS 数字 |
| 不关心什么 | 速度 | 结果对不对（假设已测过） |
| 输出 | PASS/FAIL | 性能表格 |
| 数据范围 | 有规律的模式（i%7*0.1） | 随机数 [-1,1] |

**为什么测试用有规律的数据？** 因为如果用全 0 或全 1，某些 bug 会被掩盖（比如符号错误、零值短路）。用 `i%7*0.1` 这种模式，覆盖正负、不同量级，更容易暴露问题。

**为什么 benchmark 用随机数？** 因为有规律的数据可能有"友好"的 cache 行为，测出来的性能不真实。随机数让 cache 行为接近真实场景。

---

## 完整数据流：一张图说清

```
                  ┌─────────────────────────────────┐
                  │  CMakeLists.txt (编译期编织)     │
                  │  5 个源文件 → 2 个可执行文件      │
                  └────────────┬────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
   ╔═══════════════════╗              ╔════════════════════╗
   ║ 流水线 A: 测试     ║              ║ 流水线 B: 基准     ║
   ║ test_gemm.exe     ║              ║ bench_gemm.exe     ║
   ╠═══════════════════╣              ╠════════════════════╣
   ║                   ║              ║                    ║
   ║ 1. 构造小矩阵     ║              ║ 1. 构造多尺寸矩阵  ║
   ║    A,B (64x64)    ║              ║    A,B (64~1024)   ║
   ║                   ║              ║                    ║
   ║ 2. 调被测函数     ║              ║ 2. 预热(扔结果)    ║
   ║    gemm_naive_ikj ║              ║    gemm_naive_ikj  ║
   ║    → C_test       ║              ║                    ║
   ║                   ║              ║ 3. 多次取最小值    ║
   ║ 3. 调参考实现     ║              ║    gemm_xxx × 5    ║
   ║    reference_gemm ║              ║    → best_ms       ║
   ║    → C_ref        ║              ║                    ║
   ║                   ║              ║ 4. 算 GFLOPS       ║
   ║ 4. 逐元素对比     ║              ║    2*M*N*K/time    ║
   ║    |C_t - C_r|    ║              ║                    ║
   ║    < 1e-4 ?       ║              ║ 5. 打印性能表格    ║
   ║                   ║              ║                    ║
   ║ 5. PASS / FAIL    ║              ║ → 复制到性能日志   ║
   ╚═══════════════════╝              ╚════════════════════╝
              │                                 │
              ▼                                 ▼
        算法可信                          性能数据可信
              │                                 │
              └────────────┬────────────────────┘
                           ▼
                  才能进入 Phase 2
              （用可信的算子构建 Attention）
```

**核心结论**：Phase 1 产出的不是"代码"，是**"经过验证的、性能可测的、理解透彻的 GEMM 实现"**。代码只是载体，理解和数据才是交付物。

---

## 一句话总结整个 Phase 1 的数据流

> 一组随机矩阵从 `bench_gemm` 的 `main()` 诞生，以引用传给 GEMM 函数被原地改写，函数内部 CPU 通过 cache 层级搬运数据完成 2×M×N×K 次浮点运算，结果回流到 `main()` 被计时和换算成 GFLOPS，最终落成性能日志的一行——这行数字背后，是 cache line 的搬运、分支预测、SIMD 向量化等硬件机制的总和体现。
