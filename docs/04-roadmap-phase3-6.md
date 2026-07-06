# Phase 3-6 路线图：后续阶段的战略指南

> 这份文档不展开教学（到时候每个阶段会有独立的深度文档），而是让你**现在就看清全局**，知道每一步在通往哪里。

---

## Phase 3：计算图引擎——"框架"的灵魂

### 核心问题
Phase 2 你有了一堆算子函数。但真实推理框架不是"手动调函数"，而是：
1. 模型定义在配置文件里（JSON/Protobuf）
2. 引擎读配置 → 构建计算图 → 按拓扑序执行

### 你要实现的四个组件

**1. 图表示（Graph Representation）**
```
用 JSON 描述网络。比如 Attention block：
{
  "nodes": [
    {"id": "q_proj", "op": "MatMul", "inputs": ["x"], "weights": "Wq"},
    {"id": "k_proj", "op": "MatMul", "inputs": ["x"], "weights": "Wk"},
    {"id": "v_proj", "op": "MatMul", "inputs": ["x"], "weights": "Wv"},
    {"id": "attn", "op": "Attention", "inputs": ["q_proj","k_proj","v_proj"]},
    {"id": "ln", "op": "LayerNorm", "inputs": ["attn"]}
  ]
}
```

**2. 图解析器（Graph Parser）**
读 JSON → 构建内存中的 DAG（有向无环图）。数据结构：
```cpp
struct Node {
    std::string id;
    std::string op_type;       // "MatMul", "Softmax" 等
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    // 算子特定参数（如 LayerNorm 的 eps）
};
struct Graph {
    std::vector<Node> nodes;
    std::map<std::string, Tensor> tensors;  // 张量名 → 数据
};
```

**3. 拓扑排序（Topological Sort）**
确定执行顺序。用 Kahn's 算法（基于入度）：
```
1. 找所有入度为 0 的节点入队
2. 取队首节点执行，把它指向的后继节点入度 -1
3. 入度变 0 的后继入队
4. 重复直到队空
```
**为什么必须拓扑序？** 因为图里有依赖（B 需要 A 的输出），乱序执行会读到未计算的数据。

**4. 内存池（Memory Pooling）——Proposal 的加分项**
naive 做法：每个中间张量单独 `new`/`delete`。问题：分配开销大 + 内存碎片。
优化：预分配一个大 buffer，中间张量"借"buffer 里的区域，用完归还。
```
[buffer: ┌──A──┬──B──┬──C──┬──空闲──┐]
算子执行时 A/B/C 复用同一块内存的不同区间
```
进阶：**生命周期分析**——两个张量如果生命周期不重叠，可以复用同一块内存。这是 TensorRT、ONNX Runtime 的核心技术。

### Phase 3 完成标志
- 能读 JSON 配置 → 构建 DAG → 拓扑排序 → 执行一个完整 Attention block
- 内存池工作，对比 naive 分配的内存占用

---

## Phase 4：GPU 加速 vs CPU 并行——选你的路

### 主线：CUDA（需要 NVIDIA GPU）

**学习曲线陡，但工业价值最高。** 你要做：

1. **CUDA 基础**：理解 grid/block/thread 层次、shared memory、warp
2. **重写算子**：GEMM、Softmax、LayerNorm、GELU 的 CUDA 版
3. **H2D/D2H 传输**：CPU↔GPU 数据搬运，理解 PCIe 带宽瓶颈
4. **CUDA Stream**：异步执行，重叠计算和传输
5. **Profiling**：Nsight Compute/Systems 找瓶颈

**CUDA GEMM 的思维模型**（和 CPU 完全不同）：
```
CPU: 几个核，每个核干重活，靠 SIMD 提速
GPU: 几千个核，每个核干轻活，靠海量并行提速
```
CPU 分块是为了塞进 cache；GPU 分块是为了塞进 shared memory + 让每个 block 独立计算。

**CUDA GEMM 的经典分块策略（tiling）**：
```
Grid: (M/BM) × (N/BN) 个 block
每个 block:
  - shared memory 里加载 A 的 BM×BK 小块 + B 的 BK×BN 小块
  - BM×BN 个 thread，每个算 C 的一个元素
  - 累加 K 维度上的多个小块
```
这就是 "shared memory tiling"，CPU 分块思想的 GPU 版本。

### 替代线：CPU 并行 + INT8 量化（无 GPU 时）

**这条路是 llama.cpp 的核心，同样硬核。**

1. **多线程**：OpenMP 或线程池，并行化算子
   ```cpp
   #pragma omp parallel for
   for (int i = 0; i < M; ++i) { ... }  // 自动并行化
   ```
   测 1/2/4/8 核的 scaling，分析 Amdahl's Law。

2. **INT8 量化**：
   - 把 FP32 权重转 INT8（4 倍内存节省 + 2-4 倍速度提升）
   - 数学：`fp32_value = scale * (int8_value - zero_point)`
   - 实现 INT8 GEMM：`int8_a × int8_b → int32 累加 → 反量化回 fp32`
   - 关键：用 `_mm256_maddubs_epi16`（AVX2 INT8 乘加指令）

3. **精度损失分析**：量化前后输出对比，报告 PSNR/最大误差

---

## Phase 5：Flash Attention——项目最大亮点

### 为什么 Flash Attention 重要？
标准 Attention 要把 N×N 的 attention 矩阵写回 HBM（显存），再读出来做 softmax，再读出来乘 V。N=2048 时这个矩阵 16MB，HBM 带宽是瓶颈。

Flash Attention 的核心：**用 SRAM（片上缓存，比 HBM 快 10 倍）分块计算，不把 N×N 矩阵写回 HBM**。

### 核心思想（一句话）
> "分块计算 Softmax + Attention，中间结果只存在 SRAM 寄存器里，最终结果直接写 HBM。"

### 实现难点
1. **分块 Softmax 的"在线算法"**：标准 softmax 要两遍（一遍 sum 一遍归一化），分块要一遍搞定。需要维护"当前块的 max 和 sum"两个 running 统计量。
2. **重缩放**：每加一个新块，之前的结果要按新的 max 重新缩放。

### GPU vs CPU 版本
- **GPU Flash Attention**（主线）：CUDA kernel，SRAM tiling
- **CPU Flash Attention**（替代线）：分块让小块塞进 L1/L2，思路完全一样，只是 SRAM→L1

**这个概念面试极高频，做出来简历直接加分。**

---

## Phase 6：端到端验证与 Benchmarking

### 不是"最后做"，是"贯穿始终"
每完成一个 Phase，都做：
1. **正确性验证**：对齐 PyTorch，多尺寸/多 batch
2. **性能测量**：延迟、吞吐、内存占用
3. **记录对比**：和前一版本比，和 PyTorch/工业实现比

### 最终交付
Proposal 要求"final internship presentation"。包含：
- 架构图（计算图引擎设计、数据流）
- 性能曲线（各优化阶段的 GFLOPS、cache miss 率）
- Profiling 截图（perf/Nsight）
- 精度报告（FP32 vs INT8 的误差）
- 结论：学到了什么、和工业实现的差距在哪

---

## 整体时间线建议（回顾）

| 阶段 | 时间（全职） | 关键交付 |
|---|---|---|
| Phase 1 | 1-1.5 周 | 性能日志 + cache miss 分析 |
| Phase 2 | 3-5 天 | 5 个算子 + 单元测试 |
| Phase 3 | 1 周 | 能跑 JSON 的计算图引擎 |
| Phase 4 | 1.5-2 周 | CUDA kernel 或 CPU 多线程+量化 |
| Phase 5 | 1-1.5 周 | Flash Attention kernel |
| Phase 6 | 贯穿 | 最终报告 + 演示 |

---

## 现在你要做的

1. **回头读 Phase 1 文档**（`02-phase1-deep-dive.md`），从头动手
2. 搭环境：装 CMake、编译器（Windows 用 MSVC 或 MinGW，推荐 WSL2）
3. 跑通 naive ijk，记录 baseline
4. 做 ikj 优化，对比
5. 做分块优化，对比
6. （可选）SIMD
7. 写性能日志

**每一步遇到问题，把错误信息贴给我，我帮你分析。这不是一次性任务，是迭代学习。**
