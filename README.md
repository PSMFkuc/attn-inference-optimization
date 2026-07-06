# AttnInferenceFramework
# 高性能深度学习推理框架（Attention 专用）—— 实习项目

从零搭建一个轻量级深度学习推理框架，专注 Attention 架构。覆盖 CPU/GPU 架构、内存层级、计算图编排、CUDA 优化的完整链路。

## 项目结构

```
AttnInferenceFramework/
├── docs/                          ← 教学文档（先读这里！）
│   ├── 01-strategic-overview.md         行业视角战略总览
│   ├── 02-phase1-deep-dive.md           Phase 1 深度教学（CPU GEMM）
│   ├── 03-phase2-operators-preview.md   Phase 2 预览（Attention 算子）
│   └── 04-roadmap-phase3-6.md           Phase 3-6 路线图
├── phase1_gemm/                   ← Phase 1: CPU GEMM 优化
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── timer.h                计时工具（RAII）
│   │   ├── gemm_naive.h/.cpp      朴素实现（ijk + ikj）
│   │   └── gemm_tiled.h/.cpp      分块优化
│   ├── tests/
│   │   └── test_gemm.cpp          正确性测试（对齐参考实现）
│   └── profiling/
│       └── bench_gemm.cpp         性能基准测试
├── phase2_operators/              ← Phase 2: Attention 算子库（待实现）
├── phase3_graph/                  ← Phase 3: 计算图引擎（待实现）
├── phase4_gpu/                    ← Phase 4: GPU/CUDA 或 CPU 并行（待实现）
├── phase5_flash/                  ← Phase 5: Flash Attention（待实现）
├── phase6_benchmark/              ← Phase 6: 端到端验证
├── common/                        ← 跨阶段共享代码
└── scripts/                       ← Python 辅助脚本
    └── gen_reference.py           生成 PyTorch 参考答案
```

## 阅读顺序（重要）

**第一次读，按这个顺序：**

1. `docs/01-strategic-overview.md` —— 先建立全局认知，理解这个项目在工业界的位置
2. `docs/02-phase1-deep-dive.md` —— Phase 1 深度教学，含每一行代码的讲解
3. 动手跑 Phase 1 代码（见下方"快速开始"）
4. 完成后读 `docs/03-phase2-operators-preview.md` 进入下一阶段

**不要跳过教学文档直接看代码**——代码里的注释是"怎么写"，文档里是"为什么这么写"。两者配合才能学到位。

## 快速开始（Phase 1）

### 依赖
- CMake >= 3.16
- C++17 编译器（GCC 9+/Clang 10+/MSVC 2019+）
- （可选）WSL2 用于 Linux 性能分析工具

### 构建与运行

```bash
cd phase1_gemm
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .

# 跑正确性测试
ctest

# 跑性能基准
./bench_gemm
```

### 预期输出（bench_gemm）
```
size     | ijk(ms) GFLOPS         | ikj(ms) GFLOPS         | tiled(ms) GFLOPS
--------------------------------------------------------------------------------
64       | 0.12    2.18           | 0.05    5.24           | 0.04    6.55
128      | 1.05    1.99           | 0.38    5.49           | 0.22    9.48
256      | 12.3    2.17           | 3.85    6.94           | 1.52    17.56
512      | 145     1.85           | 38.2    7.02           | 11.8    22.72
1024     | 1280    1.68           | 312     6.89           | 95.4    22.53
```
（具体数字取决于你的 CPU，但趋势应该一致：ikj 比 ijk 快 2-3 倍，分块再快 2-4 倍）

## 学习方法（必读）

这个项目的目的不是"跑通代码"，而是"理解硬件"。三条原则贯穿全程：

1. **先正确，再快速** —— naive 版本先对齐参考实现，再优化
2. **先测量，再优化** —— 没有 baseline 不优化，每次优化都对比
3. **一次只改一个变量** —— 不要同时加多个优化，搞不清谁起作用

每完成一个优化，记录到 `docs/performance-log.md`，这是 Proposal 的硬性交付物。

## 当前进度

- [x] Phase 1 教学文档 + 起步代码（naive ijk/ikj + 分块）
- [ ] Phase 1 SIMD 版本（进阶，可选）
- [ ] Phase 1 性能日志模板
- [ ] Phase 2 算子库
- [ ] Phase 3 计算图引擎
- [ ] Phase 4 GPU/CUDA 或 CPU 并行
- [ ] Phase 5 Flash Attention
- [ ] Phase 6 端到端 benchmark

## 求助方式

动手时遇到任何问题：
- 编译错误：贴完整报错信息
- 性能异常：贴 benchmark 输出 + 你的 CPU 型号
- 概念不懂：直接问，"我不理解为什么分块能减少 cache miss"

这是迭代式学习，不是一次性任务。一步步来。
