# 战略总览：从行业从业者视角看这个推理框架项目

> 这份文档不讲代码，讲的是"怎么想"。代码随时能查，但工程师的思维方式一旦建立，受用终身。

---

## 一、这个项目在工业界到底是什么位置

先把它放到真实世界的技术版图里看。你日常用的 PyTorch、TensorRT、vLLM、llama.cpp，背后都有这样一套"推理引擎"。它们的演进路线几乎和你这个 Proposal 的 6 个 Phase 一一对应：

| 真实工业项目 | 对应你 Proposal 的阶段 | 它解决的问题 |
|---|---|---|
| **llama.cpp** | Phase 1 + Phase 2 + Phase 4(Alt) | 在没有 GPU 的设备上跑大模型，靠 CPU 优化 + INT8 量化 |
| **TensorRT** (NVIDIA) | Phase 3 + Phase 4 + Phase 5 | 把模型编译成 GPU 上最优的算子组合，做 kernel fusion |
| **FlashAttention** (Tri Dao) | Phase 5 | 解决标准 Attention 的 HBM 带宽瓶颈，是 ChatGPT 能跑快的核心之一 |
| **ONNX Runtime** | Phase 3 | 通用计算图执行引擎，跨硬件后端 |
| **TVM / Triton** | 贯穿 | 编译器视角的算子生成与自动调优 |

所以你的项目不是"玩具"——你在重走工业级推理框架的演化之路，只是规模缩小了。这个认知很重要，它决定了你该怎么对待每一行代码。

---

## 二、从业者怎么看待这个 Proposal 的"含金量"

说实话，这类项目在面试和实习评估里含金量极高，但**坑也极深**。区分"水过"和"真学到东西"的关键在于三个字：**会测量**。

### 水过的典型表现
- Phase 1 直接调用 NumPy 的 `@` 或 `np.matmul`，然后说"我实现了 GEMM"
- Phase 2 全部用 PyTorch 算子包一层函数
- Phase 4 写个 CUDA kernel 但跑得比 PyTorch 还慢，不分析为什么
- 最后报告里只有"功能实现了"，没有性能数据

### 真正学到东西的表现
- Phase 1 能画出 cache 命中率的曲线，能解释"为什么分块后快了 3 倍"而不是只说"快了 3 倍"
- Phase 2 的 Softmax 能讲清楚为什么要减最大值（数值稳定性，防止 exp 溢出）
- Phase 4 的 CUDA kernel 能说出 occupancy、shared memory bank conflict 是什么
- Phase 5 的 Flash Attention 能推导出为什么 HBM 访问从 O(N²) 降到 O(N²·d/M)，M 是 SRAM 大小

**记住：这个项目的交付物不是"能跑的代码"，而是"性能日志 + 你的理解"。** Proposal 里每个 Phase 的 Deliverables 都明确写了"profiling data""comparative analysis"——这不是装饰，是核心。

---

## 三、六个 Phase 的真实优先级和依赖关系

Proposal 把 6 个 Phase 列成线性，但实际执行时依赖关系是这样的（我按"学习价值/难度比"重新排序）：

```
Phase 1 (CPU GEMM)  ────────►  方法论的根基，必须做透
   │                              （naive→测量→优化→再测量）
   ▼
Phase 2 (算子库)  ──────────►   Attention 的积木，必须正确
   │                              （对齐 PyTorch 到 1e-5）
   ▼
Phase 3 (计算图引擎)  ──────►   把积木拼成引擎，是"框架"的核心
   │
   ├──► Phase 4 主线 (CUDA)      需要 NVIDIA GPU + Nsight
   │        │
   │        ▼
   │    Phase 5 (Flash Attention)  CUDA 的高阶应用，工业级热点
   │
   └──► Phase 4 Alt (CPU 并行+量化)  没有 GPU 时的替代路径
            │
            ▼
        Phase 5 Alt (CPU Flash Attention)  llama.cpp 的核心思想

Phase 6 (验证+Benchmark)  ────►  贯穿所有阶段，不是最后才做
```

### 我的建议执行顺序

1. **Phase 1 做到 80% 透彻**（naive + 分块 + 一次 SIMD 尝试 + profiling），不求全，求理解。这是你建立"性能直觉"的窗口。
2. **Phase 2 快速做**（重点是正确性，不是性能），它服务于 Phase 3。
3. **Phase 3 是分水岭**——做出一个能加载 JSON 跑通 Attention block 的引擎，这个项目就有了"框架"的骨架，简历上能写。
4. **Phase 4 选主线还是 Alt，取决于你有没有 NVIDIA GPU**。有就走 CUDA 主线（工业价值更高），没有就走 CPU Alt（llama.cpp 路线，同样硬核）。
5. **Phase 5 是亮点**，无论主线还是 Alt，Flash Attention 都是面试高频考点，做出来就是加分项。
6. **Phase 6 贯穿始终**，每完成一个阶段就 benchmark 一次，别堆到最后。

---

## 四、技术栈选型：每一项为什么这么选

### 编程语言
| 部分 | 语言 | 为什么 |
|---|---|---|
| Phase 1-3, 4-Alt, 5-Alt | **C++ (配合一点 C)** | 这是系统级性能项目的标配。Python 解释器开销会完全淹没你优化的效果，你测不出 cache 的影响。C++ 让你直接控制内存布局。 |
| Phase 4 主线, 5 主线 | **CUDA C++** | GPU 编程的事实标准。 |
| 测试基准 / 编排脚本 | **Python (NumPy/PyTorch)** | 用来生成参考答案、跑 benchmark、画图。Python 是"裁判"，不是"选手"。 |

**关键认知**：很多人用 Python 实现优化算法，然后纳闷"为什么我优化了反而更慢"——因为 Python 函数调用、对象创建的开销远大于你省下的计算。性能敏感的核心逻辑必须用 C++/CUDA。

### 构建系统：CMake
工业 C++ 项目都用 CMake。它会处理：
- 跨平台编译（Windows/Linux/Mac）
- 编译选项管理（`-O3 -mavx2` 这种优化 flag）
- 多目标（库、可执行文件、测试）
- 第三方依赖（FetchContent 找 GoogleTest 等）

你现在不需要精通 CMake，但要用它。我会在 Phase 1 给你完整模板。

### 性能分析工具
| 工具 | 平台 | 测什么 |
|---|---|---|
| `perf stat` | Linux | cache miss、指令数、分支预测（最常用） |
| `Valgrind + Cachegrind` | Linux | 精确的 cache 命中率模拟 |
| Intel VTune | Win/Linux | 商业级，cache、内存、热点全覆盖 |
| `perf` 的 Windows 替代：用编译器内置计时 + 自己计数 | Windows | 见 Phase 1 说明 |
| NVIDIA Nsight Compute | CUDA | 单 kernel 的 SM 占用率、带宽利用率 |
| NVIDIA Nsight Systems | CUDA | 整体时间线、kernel 启动开销、stream 并行 |

> Windows 上做 Phase 1 的 cache 分析比较麻烦（没有 perf/Cachegrind）。两个方案：(1) 用 WSL2 跑 Linux 工具链；(2) 退而求其次用 `__rdtsc()` 计时 + 理论分析。我会在 Phase 1 文档里讲两种做法。

### 测试框架：GoogleTest
工业项目用 GoogleTest 做 C++ 单元测试。它处理：
- 断言（`EXPECT_NEAR` 对比浮点数，正好满足 1e-5 容差）
- 参数化测试（一次测多个矩阵尺寸）
- 测试发现与报告

---

## 五、贯穿全项目的三条工程原则

这三条比任何具体技术都重要，请刻在脑子里：

### 原则 1：先正确，再快速
永远先用最 naive、最直白的方式实现，对齐参考实现（NumPy/PyTorch）到容差以内。**一个跑得飞快但结果错误的 kernel 毫无价值。** 每次优化前后都要跑一次正确性测试。

### 原则 2：先测量，再优化
"我觉得这里慢"——错。用 profiler 告诉你哪里慢。性能优化的第一定律是 **Amdahl's Law**：优化占总时间 5% 的部分，哪怕提速 100 倍，整体只快 5%。先找到热点。

实操：每次优化前记录 baseline，优化后记录新数据，写进性能日志。Proposal 要求的"percentage of performance improvement at each step"就是这么来的。

### 原则 3：一次只改一个变量
不要同时加分块 + SIMD + 循环展开，然后说"快了 4 倍"——你不知道哪个起作用了。一次只改一处，测量，记录。这叫 **controlled experiment**，是工程师的基本素养。

---

## 六、时间投入的现实预期

如果你是全职投入（每天 4-6 小时），一个合理的节奏是：

| Phase | 时间 | 说明 |
|---|---|---|
| Phase 1 | 1-1.5 周 | 含学 CMake、搭环境、学 perf |
| Phase 2 | 3-5 天 | 算子不复杂，重点是正确性测试 |
| Phase 3 | 1 周 | 计算图引擎是新的概念，要想清楚 |
| Phase 4 | 1.5-2 周 | CUDA 学习曲线陡 |
| Phase 5 | 1-1.5 周 | Flash Attention 理解难，实现可简化 |
| Phase 6 | 贯穿 | 边做边记录 |

**如果时间紧张**，最小可行版本（MVP）是：Phase 1(naive+分块) + Phase 2 + Phase 3 + Phase 4 或 4-Alt 选一 + Phase 5 简化版。这个组合已经能讲出一个完整故事。

---

## 七、推荐的前置知识补给

如果某些概念你还没接触过，按需补：

| 概念 | 推荐资源 | 优先级 |
|---|---|---|
| C++ 基础（指针、引用、内存管理） | *Effective C++* 或 cppreference | 必须 |
| CPU 缓存原理 | *Computer Systems: A Programmer's Perspective* 第 6 章 | 必须 |
| GEMM 优化方法论 | [How to Optimize GEMM (MIT)](https://ocw.mit.edu/) / 论文 *Anatomy of High-Performance Matrix Multiplication* | 必须 |
| CUDA 编程 | NVIDIA *Programming Guide* + 《Professional CUDA C Programming》 | Phase 4 时 |
| Flash Attention 原理 | Tri Dao 原论文 + [他的博客讲解](https://arxiv.org/abs/2205.14135) | Phase 5 时 |
| Attention 机制 | *Attention is All You Need* + The Illustrated Transformer (Jay Alammar) | Phase 2 前 |

---

## 下一步

看完这份总览后，打开 `docs/02-phase1-deep-dive.md`，我们从 Phase 1 开始动手。那里我会：
- 解释 CPU 内存层级（配图）
- 讲清楚 GEMM 为什么慢
- 一步步实现 naive 版本，每一行代码都讲作用
- 带你做分块优化，告诉你"为什么这样改快了"
- 教你用工具测量

**带着问题来**：在开始前，自己先想想——"两个矩阵相乘，CPU 在干嘛？它的瓶颈可能是什么？" 带着这个问题读下一份文档，收获会大十倍。
