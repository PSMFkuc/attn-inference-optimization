# Phase 4 周报 - Week 4: CPU Inference Acceleration 后端、图级并行与 INT8 权重量化

> 日期：2026-07-23 | 项目：AttnInferenceFramework / Phase 4 CPU Acceleration  
> 路线选择：当前环境没有 CUDA-capable GPU，因此 Phase 4 从 CUDA 主线调整为 CPU inference acceleration 主线，对应 llama.cpp 类 CPU 推理优化路线。

---

## 一、本周工作摘要

本周完成 Phase 4 的 CPU 推理加速版本。核心目标是：在不推翻 Phase 3 计算图引擎的前提下，为图执行增加可替换后端、CPU 并行算子、DAG level 并行调度和 INT8 weight-only 量化路径。

主要完成内容：

- 新增 `Backend` 抽象接口，把“图怎么调度”和“算子怎么执行”解耦。
- 新增 `CpuParallelBackend`，复用 Phase 1 的 OpenMP `gemm_parallel_tiled`，并实现并行 Softmax、row-wise LayerNorm、GELU approx、标准 Attention。
- 新增 `ParallelGraphExecutor`，支持按拓扑序串行执行，也支持按 DAG level 并行执行。
- 新增 `QuantizedTensor` 和 INT8 weight-only GEMM，支持 per-tensor 和 per-output-channel symmetric quantization。
- 新增 Phase 4 单元测试，覆盖 DAG 分层、串行/并行一致性、row-wise LayerNorm、INT8 量化误差。
- 新增 Phase 4 benchmark，量化比较 naive GEMM、OpenMP parallel GEMM、INT8 weight GEMM 和图级调度。
- 新增 Phase 4 CPU 加速说明文档和 Phase3+Phase4 Word 教学文档。

正确性验证：

```text
test_phase4_cpu:
Results: 27 passed, 0 failed
```

性能摘要：

```text
512 x 512 GEMM:
naive      75.505 ms
parallel   19.652 ms
speedup     3.84x

INT8 weight-only GEMM:
512 x 512 max_abs_error = 0.03274
```

---

## 二、Phase 4 的定位与技术路线

### 2.1 为什么不做 CUDA

原始路线图里 Phase 4 有两条路：

| 路线 | 依赖 | 代表系统 | 当前是否适合 |
|---|---|---|---|
| CUDA GPU acceleration | NVIDIA GPU + CUDA Toolkit + Nsight | TensorRT / CUDA backend | 当前环境不适合 |
| CPU parallel + INT8 quantization | 多核 CPU + OpenMP / SIMD | llama.cpp / CPU inference runtime | 当前主线 |

当前机器没有 CUDA-capable GPU，因此继续写 CUDA kernel 会导致两个问题：

1. 无法真实运行和 profile，性能数据没有可信度。
2. 代码只能停留在“看起来像 CUDA”，不能证明推理系统真的变快。

所以本周采用 CPU inference acceleration 路线。这个路线不是降级，而是另一条真实工业路线：面向本地 CPU 推理、边缘设备推理、无独显部署场景，重点是多线程、缓存、量化、内存访问和调度开销。

### 2.2 Phase 4 与 Phase 3 的关系

Phase 3 已经完成了计算图：

```text
JSON -> ComputeGraph -> topological_sort -> Executor -> Phase1/2 operators
```

Phase 4 不重写这些结构，而是替换执行后端：

```text
Phase 3:
ComputeGraph + sorted_nodes
        |
        v
Executor hard-coded if/else
        |
        v
gemm_naive_ikj / Phase2 operators

Phase 4:
ComputeGraph + sorted_nodes
        |
        v
ParallelGraphExecutor
        |
        v
Backend interface
        |
        v
CpuParallelBackend / Quantization kernels
```

这体现了推理框架中的关键分层：Graph 负责表示模型，Scheduler 负责决定执行顺序，Backend 负责实现具体算子，Kernel 负责最终计算。

---

## 三、整体架构

### 3.1 Phase 4 数据流

```text
ComputeGraph
  |
  | topological_sort()
  v
sorted_nodes
  |
  | ParallelGraphExecutor
  v
execution levels
  |
  | level 0: q_proj, k_proj, v_proj
  | level 1: attn_out
  | level 2: ln_out
  v
CpuParallelBackend
  |
  | MatMul    -> gemm_parallel_tiled
  | Softmax   -> row-wise OpenMP softmax
  | LayerNorm -> row-wise OpenMP layernorm
  | GELU      -> OpenMP gelu_approx
  | Attention -> parallel GEMM + softmax + parallel GEMM
  v
output tensors written back to ComputeGraph
```

### 3.2 Attention Block 的 level 分层

```text
输入:
  x, Wq, Wk, Wv

level 0:
  q_proj = MatMul(x, Wq)
  k_proj = MatMul(x, Wk)
  v_proj = MatMul(x, Wv)

level 1:
  attn_out = Attention(q_proj, k_proj, v_proj)

level 2:
  ln_out = LayerNorm(attn_out)
```

`q_proj / k_proj / v_proj` 都只依赖外部输入 `x` 和权重，因此它们没有彼此依赖，理论上可以并行。这是 Phase 3 拓扑排序向 Phase 4 图级并行自然延伸的地方。

---

## 四、文件结构与职责

```text
phase4_gpu/
├── CMakeLists.txt
├── include/
│   ├── backend.h                    # Backend 抽象接口 + CpuParallelBackend 声明
│   ├── parallel_graph_executor.h    # Phase4 图执行器：serial / level-parallel
│   └── quantization.h               # INT8 量化结构和函数声明
├── src/
│   ├── cpu_parallel_backend.cpp     # OpenMP CPU 后端实现
│   ├── parallel_graph_executor.cpp  # DAG level 分层与 std::async 调度
│   └── quantization.cpp             # symmetric quantization + INT8 weight GEMM
├── tests/
│   └── test_phase4_cpu.cpp          # 27 项正确性测试
└── profiling/
    └── bench_phase4_cpu.cpp         # GEMM 和图调度 benchmark
```

配套文档：

```text
docs/
├── 09-phase4-cpu-inference-acceleration.md
├── phase3_phase4_teaching_guide.docx
└── weekly-report-phase4-week4.md
```

---

## 五、核心实现详解

### 5.1 Backend 抽象

新增文件：`phase4_gpu/include/backend.h`

核心接口：

```cpp
class Backend {
public:
    virtual ~Backend() = default;
    virtual const char* name() const = 0;
    virtual bool execute(const Node& node,
                         const std::vector<Tensor>& inputs,
                         Tensor& output,
                         std::string& error) const = 0;
};
```

设计目的：

- `Executor` 不再必须知道每个 op 如何实现。
- 后续可以增加 `CpuSimdBackend`、`CudaBackend`、`QuantizedBackend`。
- 测试和 benchmark 可以在同一张图上切换不同后端。

这一步是 Phase 4 最重要的架构动作。没有这个抽象，后续所有优化都会堆在 `executor.cpp` 的 if/else 里，代码会越来越难维护。

### 5.2 CpuParallelBackend

新增文件：`phase4_gpu/src/cpu_parallel_backend.cpp`

支持的算子：

| op_type | Phase 3 实现 | Phase 4 实现 | 变化 |
|---|---|---|---|
| MatMul | `gemm_naive_ikj` | `gemm_parallel_tiled` | 单线程 GEMM -> OpenMP 分块 GEMM |
| Softmax | `softmax_2d` 串行逐行 | OpenMP row-wise softmax | 行级并行 |
| LayerNorm | flattened tensor norm | row-wise LayerNorm | 更符合 Transformer 推理语义 |
| GELU | `gelu_vector_exact` | OpenMP `gelu_approx` | exact -> inference-friendly approx |
| Attention | Phase2 标准 attention | parallel GEMM + row-wise softmax | 组合算子加速 |

#### 5.2.1 MatMul

Phase 4 MatMul 的核心：

```cpp
gemm_parallel_tiled(M, N, K_A, A.data, B.data, output.data);
```

它复用 Phase 1 已经实现的多线程分块 GEMM。这里没有改变图结构，只替换底层 kernel，这是推理框架正确的扩展方式。

维度规则：

```text
A: [M, K]
B: [K, N]
C: [M, N]
```

如果 `K_A != K_B`，直接返回错误，避免 silent wrong result。

#### 5.2.2 Softmax

Softmax 使用 row-wise parallel：

```text
for each row r in parallel:
    softmax(row r)
```

这样做是安全的，因为每一行读写的内存区间不同：

```text
row r = data[r * cols : (r + 1) * cols]
```

没有写冲突，也没有归约冲突。

#### 5.2.3 LayerNorm

Phase 4 把 LayerNorm 改成逐行归一化：

```text
rows = data.size() / shape.back()
cols = shape.back()

for each row:
    mean = sum(row) / cols
    var  = sum((x - mean)^2) / cols
    out  = gamma * (x - mean) / sqrt(var + eps) + beta
```

这比 Phase 3 的 flattened LayerNorm 更接近真实 Transformer：

```text
输入 Tensor: [seq_len, d_model]
LayerNorm 应该对每个 token 的 d_model 维做归一化
```

#### 5.2.4 GELU

Phase 4 使用 `gelu_approx`：

```text
GELU exact:  使用 erf，数学更精确
GELU approx: 使用 tanh 近似，推理更快
```

这是推理场景常见取舍：训练/验证时优先 exact，推理加速时可接受小误差换速度。

#### 5.2.5 Attention

Phase 4 的 Attention 仍然是标准 Attention，不是 Flash Attention：

```text
K_T     = transpose(K)
scores  = Q x K_T
scores *= 1 / sqrt(d_k)
weights = softmax(scores)
output  = weights x V
```

其中两个 GEMM 都走 `gemm_parallel_tiled`，K 转置和 softmax 也使用 OpenMP 并行。

这个实现的意义是：先把标准 Attention 在 CPU 上加速并可测，Phase 5 再进一步做 Flash Attention，减少 `scores` 这个 `seq_len x seq_len` 中间矩阵的内存读写。

### 5.3 ParallelGraphExecutor

新增文件：

```text
phase4_gpu/include/parallel_graph_executor.h
phase4_gpu/src/parallel_graph_executor.cpp
```

提供两种模式：

| 函数 | 含义 | 用途 |
|---|---|---|
| `execute_serial` | 按拓扑序线性执行 Phase4 backend | 与 Phase3 对照，建立 baseline |
| `execute_level_parallel` | 同一 DAG level 内并行执行 | 验证图级并行 |
| `build_execution_levels` | 把拓扑序转换成 level 列表 | 找出可并行节点 |

level 计算规则：

```text
如果 node 的输入由图内其他 node 产生：
    node.level = max(input_producer.level + 1)
否则：
    node.level = 0
```

对于 Attention Block：

```text
level 0: q_proj, k_proj, v_proj
level 1: attn_out
level 2: ln_out
```

#### 5.3.1 为什么当前 level-parallel 反而慢

当前 benchmark：

| shape | serial(ms) | level(ms) | level 相对 serial |
|---|---:|---:|---:|
| 64x64 | 1.365 | 10.301 | 0.13x |
| 128x64 | 3.064 | 13.069 | 0.23x |

这不是正确性失败，而是一个重要性能发现：

1. `ComputeGraph` 的 Tensor store 是 `unordered_map`，不是线程安全容器。
2. 为避免 worker 线程直接读写 graph，当前实现会先复制本 level 的输入 Tensor。
3. 每个 MatMul 内部已经使用 OpenMP，再用 `std::async` 跑多个 MatMul，容易形成线程过度订阅。
4. 小图和小矩阵下，线程启动和复制成本大于并行收益。

结论：图级并行已经实现并验证正确，但后续需要调度策略避免盲目并行。比如：大节点使用 operator-level OpenMP，小而独立的节点使用 graph-level parallel，对同一时刻的线程数做统一控制，并引入 TensorView 减少输入快照拷贝。

### 5.4 INT8 weight-only quantization

新增文件：

```text
phase4_gpu/include/quantization.h
phase4_gpu/src/quantization.cpp
```

实现内容：

| 函数/结构 | 作用 |
|---|---|
| `QuantizedTensor` | 保存 INT8 values、shape、scales、per_channel |
| `quantize_symmetric_per_tensor` | 整个 Tensor 共用一个 scale |
| `quantize_weight_per_output_channel` | 每个输出通道/列一个 scale |
| `gemm_fp32_int8_weight` | FP32 activation x INT8 weight -> FP32 output |
| `max_abs_error` | 量化误差评估 |

量化公式：

```text
scale = max(abs(W)) / 127
W_int8 = round(W_fp32 / scale)
W_fp32_approx = W_int8 * scale
```

per-output-channel 版本：

```text
W: [K, N]
每一列 j 单独计算 scale[j]
```

GEMM 计算：

```text
for i in M:
  for j in N:
    acc = 0
    for k in K:
      acc += A_fp32[i,k] * W_int8[k,j]
    C_fp32[i,j] = acc * scale[j]
```

当前实现重点是建立量化链路和误差评估，还没有使用 AVX2/VNNI 的 INT8 dot-product 指令。因此 INT8 路径在部分尺寸下未必比 OpenMP FP32 GEMM 更快，但它已经能证明权重量化流程正确、误差可测可控，后续可以替换内层 kernel 而不改变接口。

---

## 六、性能指标与量化分析

### 6.1 测试环境与命令

当前 Windows 环境中 `cmake` 不在 PATH，因此本次使用项目自带 MinGW 工具链直接编译运行：

```powershell
.\mingw64\bin\g++.exe -std=c++17 -O2 -fopenmp ...
.\phase4_gpu\build_manual\test_phase4_cpu.exe
.\phase4_gpu\build_manual\bench_phase4_cpu.exe
```

编译参数：

```text
语言标准: C++17
优化级别: -O2
并行运行时: OpenMP (-fopenmp)
工具链: 项目内 mingw64 g++
```

### 6.2 正确性测试

```text
============================================================
  Phase 4 CPU Inference Acceleration Tests
============================================================

Test 1: DAG Levelization
Test 2: Serial vs Level-Parallel Execution
Test 3: Row-wise LayerNorm Semantics
Test 4: INT8 Weight GEMM

Results: 27 passed, 0 failed
```

测试覆盖：

| 测试项 | 验证内容 | 指标 |
|---|---|---|
| DAG Levelization | Attention graph 被分为 3 个执行 level | level 数 = 3 |
| q/k/v 并行性 | `q_proj/k_proj/v_proj` 同属 level 0 | level 0 节点数 = 3 |
| Serial vs Level Parallel | 两种调度输出一致 | max_diff < 1e-5 |
| Row-wise LayerNorm | 每行均值约 0、方差约 1 | mean < 1e-4, var error < 1e-3 |
| INT8 GEMM | 量化输出接近 FP32 GEMM | max_abs_error = 0.005825 |

### 6.3 GEMM benchmark

实测结果：

| size | naive(ms) | parallel(ms) | INT8 weight(ms) | INT8 max_abs_error |
|---:|---:|---:|---:|---:|
| 128 | 2.111 | 2.414 | 1.063 | 0.01516 |
| 256 | 8.866 | 4.973 | 5.665 | 0.02296 |
| 512 | 75.505 | 19.652 | 31.259 | 0.03274 |

加速比：

| size | parallel vs naive | INT8 vs naive | INT8 vs parallel |
|---:|---:|---:|---:|
| 128 | 0.87x | 1.99x | 2.27x |
| 256 | 1.78x | 1.56x | 0.88x |
| 512 | 3.84x | 2.42x | 0.63x |

分析：

- 128 尺寸下，OpenMP parallel GEMM 比 naive 慢，说明小矩阵线程调度开销超过计算收益。
- 256 尺寸下，parallel GEMM 开始产生收益，达到 1.78x。
- 512 尺寸下，parallel GEMM 收益明显，达到 3.84x，说明矩阵足够大后，多线程和分块能摊薄调度成本。
- INT8 weight-only 在 128 尺寸下最快，说明减少权重带宽和较轻计算在小尺寸下有优势。
- INT8 在 512 尺寸下慢于 parallel FP32，主要原因是当前 INT8 GEMM 是标量乘加，没有使用 AVX2/VNNI 指令；而 FP32 parallel GEMM 已有分块和 OpenMP 优化。
- INT8 误差随 size 增大略有上升，但最大误差仍在 0.033 以内，作为 weight-only quantization MVP 是可接受的。

### 6.4 Graph schedule benchmark

实测结果：

| shape | serial(ms) | level-parallel(ms) | levels |
|---|---:|---:|---:|
| 64x64 | 1.365 | 10.301 | 3 |
| 128x64 | 3.064 | 13.069 | 3 |

加速比：

| shape | level-parallel vs serial | 结论 |
|---|---:|---|
| 64x64 | 0.13x | 明显更慢 |
| 128x64 | 0.23x | 仍然更慢 |

原因分析：

1. 当前图很小，只有 5 个节点，能并行的只有第一层 3 个 MatMul。
2. 每个 MatMul 内部已经 OpenMP 并行，再在图层面 `std::async` 并行，线程数可能过多。
3. `gather_inputs` 为线程安全复制 Tensor，复制成本在小 shape 下不可忽略。
4. level 之间仍然有硬依赖，Attention 和 LayerNorm 无法与上游投影并行。

工程结论：

```text
图级并行已经正确实现，但当前不应默认视为性能优化。
后续需要 runtime policy：
  - 小图: serial schedule + operator OpenMP
  - 大图: selective graph-level parallel
  - 多独立小算子: graph-level parallel
  - 大 GEMM: operator-level parallel
```

---

## 七、设计决策

| 决策 | 原因 | 影响 |
|---|---|---|
| 继续使用 `phase4_gpu/` 目录名 | 保持原路线图目录结构，不大规模移动文件 | 目录名略历史化，但内容文档已明确为 CPU acceleration |
| 新增 Backend 抽象 | 解耦图调度和算子实现 | 为未来 CUDA/SIMD/Quantized backend 留接口 |
| Phase4 不直接修改 Phase3 Executor | 降低回归风险 | Phase3 原测试和行为保持稳定 |
| MatMul 复用 Phase1 `gemm_parallel_tiled` | 已有实现经过 Phase1 benchmark | 快速建立 CPU 加速收益 |
| Softmax 按行并行 | Attention 中每行 softmax 独立 | 无写冲突，容易验证 |
| LayerNorm 改成 row-wise | Transformer 推理按 hidden dim 归一化 | 语义比 Phase3 flattened norm 更准确 |
| GELU 使用 approx | 推理更重速度 | 与 exact 存在微小数值差异，需要测试约束 |
| level-parallel 先复制输入 | 保护 `ComputeGraph` 的非线程安全 tensor store | 正确性高，但有额外性能开销 |
| INT8 先做 weight-only | 实现成本低，适合推理权重场景 | 尚未覆盖 activation quantization |
| INT8 先做标量 kernel | 建立正确性和误差评估链路 | 后续需要 AVX2/VNNI 才能充分提速 |

---

## 八、开发中遇到的问题与修复

### 8.1 无 CUDA GPU，Phase4 主线调整

问题：原路线图中的 CUDA 主线需要 NVIDIA GPU，但当前环境无法运行 CUDA kernel。

处理：改走 CPU inference acceleration 路线，目标切换为 OpenMP operator parallelism、DAG level scheduling、INT8 weight-only quantization、CPU benchmark and correctness report。

结果：Phase4 仍然满足硬件加速阶段的核心目标，并且能产生真实可运行的性能数据。

### 8.2 `cmake` 不在 PATH

问题：当前环境执行 `cmake` 时提示命令不存在。

处理：保留标准 CMakeLists.txt，同时使用项目内 MinGW g++ 直接编译测试和 benchmark。

结果：Phase4 源码已通过直接编译验证：

```text
test_phase4_cpu: 27/27 PASS
bench_phase4_cpu: 正常输出性能表
```

### 8.3 C++ `std::move` 实参求值顺序问题

问题：最初写法类似：

```cpp
graph.set_tensor(output.name, std::move(output));
```

C++ 对函数实参求值顺序不保证。如果 `std::move(output)` 先发生，`output.name` 可能已经被移动，导致输出 Tensor 被写入错误名字，后续节点找不到 `q_proj`。

修复：

```cpp
const std::string output_name = output.name;
graph.set_tensor(output_name, std::move(output));
```

结果：串行和 level-parallel 执行都能正确产生 `ln_out`，测试通过。

### 8.4 level-parallel 性能低于 serial

问题：图级并行路径功能正确，但 benchmark 显示慢于 serial。

原因：输入 Tensor 复制成本、`std::async` 启动成本、图很小可并行节点少，以及 OpenMP 算子内部并行和图级并行叠加导致过度订阅。

处理：在周报和文档中明确标注该结论，不把它包装成“加速成功”。后续会设计 runtime policy 根据图规模和算子规模选择调度策略。

### 8.5 INT8 GEMM 不总是比 FP32 parallel 快

问题：512 尺寸下 INT8 weight-only GEMM 为 31.259ms，慢于 FP32 parallel 的 19.652ms。

原因：当前 INT8 内层循环是标量实现，没有使用 AVX2/VNNI `_mm256_maddubs_epi16` / dot-product 指令；FP32 parallel GEMM 已有较好的 OpenMP + tiling 基础。

处理：把当前 INT8 定位为正确性和量化误差链路 MVP，后续再做 SIMD INT8 kernel。

---

## 九、与 Phase 1/2/3 的关系

```text
Phase 1:
  gemm_naive_ikj
  gemm_tiled
  gemm_parallel_tiled
          |
          v
Phase 2:
  softmax
  layernorm
  gelu_exact / gelu_approx
  scaled_dot_product_attention
          |
          v
Phase 3:
  Tensor / Node / ComputeGraph
  JsonParser
  topological_sort
  Executor
  MemoryPool
          |
          v
Phase 4:
  Backend interface
  CpuParallelBackend
  ParallelGraphExecutor
  INT8 QuantizedTensor
  bench_phase4_cpu
```

Phase4 对前面阶段的复用关系：

| 来源 | 被 Phase4 如何使用 |
|---|---|
| Phase1 `gemm_parallel_tiled` | Phase4 MatMul 和 Attention 中两个 GEMM 的核心计算 |
| Phase2 `softmax` | Phase4 row-wise Softmax 和 Attention softmax |
| Phase2 `layernorm` | Phase4 row-wise LayerNorm 的单行计算 |
| Phase2 `gelu_approx` | Phase4 推理版 GELU |
| Phase3 `Tensor/Node/ComputeGraph` | Phase4 直接复用图表示和 Tensor store |
| Phase3 `topological_sort` | Phase4 的 serial schedule 和 levelization 输入 |

---

## 十、完整测试与 Benchmark 运行命令

### 10.1 CMake 标准路径

适用于 `cmake` 已加入 PATH 的环境：

```powershell
cd phase4_gpu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --output-on-failure
.\build\test_phase4_cpu.exe
.\build\bench_phase4_cpu.exe
```

### 10.2 当前环境直接编译路径

测试：

```powershell
.\mingw64\bin\g++.exe -std=c++17 -O2 -fopenmp `
  -Iphase4_gpu\include `
  -Iphase1_gemm\src `
  -Iphase2_operators\include `
  -Iphase3_graph\include `
  phase4_gpu\tests\test_phase4_cpu.cpp `
  phase4_gpu\src\cpu_parallel_backend.cpp `
  phase4_gpu\src\parallel_graph_executor.cpp `
  phase4_gpu\src\quantization.cpp `
  phase1_gemm\src\gemm_naive.cpp `
  phase1_gemm\src\gemm_tiled.cpp `
  phase1_gemm\src\gemm_parallel.cpp `
  phase2_operators\src\operators.cpp `
  phase3_graph\src\graph.cpp `
  phase3_graph\src\json_parser.cpp `
  phase3_graph\src\memory_pool.cpp `
  -o phase4_gpu\build_manual\test_phase4_cpu.exe

$env:Path = "C:\Users\16947\WorkBuddy\2026-06-29-14-11-22\AttnInferenceFramework\mingw64\bin;$env:Path"
.\phase4_gpu\build_manual\test_phase4_cpu.exe
```

Benchmark：

```powershell
.\mingw64\bin\g++.exe -std=c++17 -O2 -fopenmp `
  -Iphase4_gpu\include `
  -Iphase1_gemm\src `
  -Iphase2_operators\include `
  -Iphase3_graph\include `
  phase4_gpu\profiling\bench_phase4_cpu.cpp `
  phase4_gpu\src\cpu_parallel_backend.cpp `
  phase4_gpu\src\parallel_graph_executor.cpp `
  phase4_gpu\src\quantization.cpp `
  phase1_gemm\src\gemm_naive.cpp `
  phase1_gemm\src\gemm_tiled.cpp `
  phase1_gemm\src\gemm_parallel.cpp `
  phase2_operators\src\operators.cpp `
  phase3_graph\src\graph.cpp `
  phase3_graph\src\json_parser.cpp `
  phase3_graph\src\memory_pool.cpp `
  -o phase4_gpu\build_manual\bench_phase4_cpu.exe

.\phase4_gpu\build_manual\bench_phase4_cpu.exe
```

---

## 十一、本周量化交付清单

| 类别 | 交付物 | 数量/指标 |
|---|---|---:|
| 新增 Phase4 源码模块 | backend / executor / quantization / tests / benchmark | 9 个核心文件 |
| 正确性测试 | `test_phase4_cpu` | 27/27 PASS |
| DAG 分层 | Attention Block | 3 levels |
| 可并行节点 | q_proj / k_proj / v_proj | level 0 共 3 个节点 |
| GEMM 加速 | 512 x 512 parallel vs naive | 3.84x |
| INT8 误差 | 512 x 512 weight-only GEMM | max_abs_error 0.03274 |
| INT8 小尺寸收益 | 128 x 128 INT8 vs naive | 1.99x |
| 图级并行验证 | serial vs level-parallel | 输出一致，max_diff < 1e-5 |
| 文档 | Phase4 说明 + Phase3/4 Word 教学文档 + 本周周报 | 3 份 |

---

## 十二、问题总结

| # | 问题 | 类型 | 严重程度 | 当前状态 |
|---:|---|---|---|---|
| 1 | 无 CUDA GPU | 环境约束 | 高 | 已调整为 CPU acceleration 主线 |
| 2 | `cmake` 不在 PATH | 工具链问题 | 中 | CMake 文件保留，当前用 MinGW 直接验证 |
| 3 | `std::move` 后 output name 丢失风险 | C++ 语义 bug | 高 | 已保存 key 后再 move |
| 4 | level-parallel 慢于 serial | 性能问题 | 中 | 已定位为复制 + 线程过度订阅 |
| 5 | INT8 512 下慢于 FP32 parallel | 性能问题 | 中 | 已定位为缺少 SIMD INT8 kernel |
| 6 | Phase4 目录名仍叫 `phase4_gpu` | 命名历史遗留 | 低 | 文档中明确说明实际为 CPU acceleration |

---

## 十三、下周计划

| 优先级 | 任务 | 目标 |
|---|---|---|
| P1 | Runtime scheduling policy | 根据算子规模选择 serial / graph-level parallel / operator-level OpenMP |
| P1 | 减少 level-parallel Tensor copy | 引入 TensorView 或只读引用，降低图级并行开销 |
| P2 | INT8 GEMM SIMD 优化 | 使用 AVX2/VNNI 思路替换标量内层循环 |
| P2 | QuantizedBackend | 把 INT8 MatMul 接入 Backend，而不是只在独立 benchmark 中使用 |
| P2 | 完整 Transformer Block | 加入 FFN: MatMul -> GELU -> MatMul，以及 residual connection |
| P3 | CPU Flash Attention 预研 | 用 block-wise softmax 减少 `seq_len x seq_len` 中间矩阵写回 |
| P3 | 更完整 benchmark | 增加不同 seq_len、d_model、线程数、batch size 的性能曲线 |

---

## 十四、本周结论

Phase 4 已经从“计划中的 CUDA 加速阶段”调整为“可运行、可测试、可量化的 CPU 推理加速阶段”。本周最重要的成果不是单个 benchmark 数字，而是完成了后端抽象和 CPU 加速闭环：

```text
Phase3 graph representation
  -> Phase4 backend abstraction
  -> OpenMP CPU operators
  -> DAG level scheduling
  -> INT8 quantization path
  -> correctness tests
  -> benchmark report
```

当前性能结论也比较清楚：

- 大矩阵下 OpenMP parallel GEMM 有明确收益，512 尺寸达到 3.84x。
- INT8 weight-only 路径误差可控，但需要 SIMD kernel 才能稳定超过 FP32 parallel。
- 图级并行已经正确，但当前小图下性能不占优，需要调度策略和 Tensor copy 优化。

这为 Phase5 的 CPU Flash Attention 或后续更工业化的 CPU inference runtime 打好了结构基础。
