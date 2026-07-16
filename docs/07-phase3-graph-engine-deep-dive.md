# Phase 3 深度教学：计算图引擎与内存池

> 这份文档面向刚接触推理框架的新人。不讲教科书定义，用"一个矩阵如何在框架里自动流转"的视角串起全部概念。
>
> **核心问题**：Phase 2 你有 5 个能用的算子函数，Phase 3 教你**把它们自动编排成一个能跑起来的推理引擎**。

---

## 一、先理解"问题"——Phase 2 留给你的麻烦

Phase 2 结束后，你手里有这些函数：

```cpp
gemm_naive_ikj(M, N, K, A, B, C);     // 矩阵乘法
softmax(x, N);                          // Softmax 归一化
layernorm(x, N, eps, gamma, beta, out); // LayerNorm
gelu(x);                                // GELU 激活
scaled_dot_product_attention(Q, K, V, ...); // 组合 Attention
```

如果要跑一个完整的 Transformer 推理，你现在只能**手动写死调用链**：

```cpp
// 手动跑一次推理——噩梦级别
void run_manual_inference() {
    // 1. QKV 投影
    std::vector<float> Q, K, V, QK, attn_out, ln1_out, ffn_out, ln2_out;
    gemm_naive_ikj(seq_len, d_k, d_model, X, Wq, Q);   // Q = X × Wq
    gemm_naive_ikj(seq_len, d_k, d_model, X, Wk, K);   // K = X × Wk
    gemm_naive_ikj(seq_len, d_v, d_model, X, Wv, V);   // V = X × Wv

    // 2. Attention
    scaled_dot_product_attention(Q, K, V, seq_len, d_k, d_v, attn_out);

    // 3. 残差连接 + LayerNorm
    // ... 手动加 X + attn_out，手动调 layernorm ...

    // 4. FFN (两个 MatMul + GELU)
    // ... 手动 W1, W2 矩阵乘，手动 GELU ...

    // 5. 再一个残差 + LayerNorm
    // ...

    // 问题：换一个模型结构就要重写整个函数
    // 问题：中间张量数量爆炸，内存管理混乱
    // 问题：GPU 版本要完全重写调度逻辑
}
```

**这就是计算图引擎要解决的问题**——把"算子函数"从"手写调用"升级为"自动化编排"。

---

## 二、计算图引擎——给你的算子安一个"大脑"

### 2.1 一句话说清它在干嘛

> 计算图引擎 = 把模型结构描述成一个**有向无环图（DAG）**，然后**自动按正确顺序调用对应的算子函数，并管理中间数据的存储**。

听起来很复杂？拆成三步你就明白了：

### 2.2 第一步：把模型写成 JSON（图表示）

你的 Attention Block 长这样：

```
输入 X
  ├──→ Wq ──→ Q ──┐
  ├──→ Wk ──→ K ──├──→ Attention ──→ 残差+LayerNorm ──→ FFN ──→ 残差+LayerNorm ──→ 输出
  └──→ Wv ──→ V ──┘      ↑                                 ↑
                      MatMul GEMM                   MatMul ×2 + GELU
```

用 JSON 描述它——这就是"模型配置"：

```json
{
  "inputs": ["x"],
  "outputs": ["output"],
  "nodes": [
    {
      "id": "q_proj",
      "op": "MatMul",
      "inputs": ["x"],
      "weights": "Wq",
      "params": {"transpose_b": false}
    },
    {
      "id": "k_proj",
      "op": "MatMul",
      "inputs": ["x"],
      "weights": "Wk"
    },
    {
      "id": "v_proj",
      "op": "MatMul",
      "inputs": ["x"],
      "weights": "Wv"
    },
    {
      "id": "attn",
      "op": "ScaledDotProductAttention",
      "inputs": ["q_proj", "k_proj", "v_proj"],
      "params": {"d_k": 64, "d_v": 64}
    },
    {
      "id": "add1",
      "op": "Add",
      "inputs": ["x", "attn"]
    },
    {
      "id": "ln1",
      "op": "LayerNorm",
      "inputs": ["add1"],
      "params": {"eps": 1e-5}
    },
    {
      "id": "ffn1",
      "op": "MatMul",
      "inputs": ["ln1"],
      "weights": "W1"
    },
    {
      "id": "gelu1",
      "op": "GELU",
      "inputs": ["ffn1"]
    },
    {
      "id": "ffn2",
      "op": "MatMul",
      "inputs": ["gelu1"],
      "weights": "W2"
    },
    {
      "id": "add2",
      "op": "Add",
      "inputs": ["ln1", "ffn2"]
    },
    {
      "id": "ln2",
      "op": "LayerNorm",
      "inputs": ["add2"],
      "params": {"eps": 1e-5}
    }
  ]
}
```

**关键点**：每个节点只有三个信息——"我叫什么"（id）、"我是什么算子"（op）、"我吃谁的数据"（inputs）。**没有写"我该什么时候执行"**——这个由引擎自动推出来。

### 2.3 第二步：从 JSON 构建 DAG（图解析）

引擎读 JSON 后，在内存里构建这张图。数据结构大概是：

```cpp
struct Tensor {
    std::string name;           // 张量名，如 "q_proj"
    std::vector<float> data;   // 实际数据
    int rows, cols;            // 形状
};

struct Node {
    std::string id;            // "q_proj"
    std::string op_type;       // "MatMul"
    std::vector<std::string> input_names;   // ["x"]
    std::vector<Node*> predecessors;        // 指向输入节点（运行时填充）
    std::vector<Node*> successors;          // 指向输出节点
    std::map<std::string, float> params;    // {"eps": 1e-5}
    int in_degree;             // 入度，拓扑排序用
};

struct Graph {
    std::map<std::string, Node*> nodes;     // id → Node
    std::map<std::string, Tensor> tensors;  // 张量名 → 数据
    std::vector<std::string> input_ids;     // 外部输入
    std::vector<std::string> output_ids;    // 外部输出
};
```

图解析器做的事就是把 JSON 的字符串依赖关系，转成 C++ 指针：

```
"inputs": ["x"]       →   node->predecessors.push_back(&nodes["x"])
"weights": "Wq"       →   node->params["weight_name"] = "Wq"
```

### 2.4 第三步：拓扑排序——自动计算执行顺序

**为什么需要它？** 你不能随便执行。必须先算 `attn` 再算 `add1`（`add1` 吃 `attn` 的输出）。拓扑排序自动求出一个**满足所有依赖关系的执行顺序**。

我们来手动走一遍 Kahn 算法：

```
初始入度（每个节点被几个其他节点依赖）：
  x:       0   ← 外部输入，已有数据
  q_proj:  1   (依赖 x)
  k_proj:  1   (依赖 x)
  v_proj:  1   (依赖 x)
  attn:    3   (依赖 q_proj, k_proj, v_proj)  ← 入度最高，最后才能执行
  add1:    2   (依赖 x, attn)
  ln1:     1   (依赖 add1)
  ffn1:    1   (依赖 ln1)
  gelu1:   1   (依赖 ffn1)
  ffn2:    1   (依赖 gelu1)
  add2:    2   (依赖 ln1, ffn2)
  ln2:     1   (依赖 add2)
```

Kahn 算法执行过程：

```
第 1 轮：入度为 0 的节点 = [x]（已有，跳过）
  执行完 x？不，x 是输入，已经存在

第 2 轮：从 x 出发，它的后继入度 -1
  q_proj: 1 → 0, k_proj: 1 → 0, v_proj: 1 → 0
  入度变 0 的：q_proj, k_proj, v_proj
  → 这三个可以**并行**执行！（它们之间没有依赖）

第 3 轮：执行 q_proj, k_proj, v_proj
  从它们出发，attn 的入度：3 → 0
  → attn 可以执行

第 4 轮：执行 attn
  add1: 2 → 1 (x 已经处理过，只剩 attn 的依赖)
  → 等等，x 在第 2 轮就处理了，所以 add1 入度：1 → 0
  → add1 可以执行

第 5 轮：执行 add1
  ln1: 1 → 0

第 6 轮：执行 ln1
  ffn1: 1 → 0

第 7 轮：执行 ffn1
  gelu1: 1 → 0

第 8 轮：执行 gelu1
  ffn2: 1 → 0

第 9 轮：执行 ffn2
  add2: 2 → 1 (ln1 在第 6 轮就处理了)
  → add2: 1 → 0

第 10 轮：执行 add2
  ln2: 1 → 0

第 11 轮：执行 ln2 → 输出
```

**最终拓扑序**：
```
[q_proj, k_proj, v_proj]  →  attn  →  add1  →  ln1  →  ffn1  →  gelu1  →  ffn2  →  add2  →  ln2
```

**三个你可以并行执行的分组**：`{q_proj, k_proj, v_proj}` 和 `{ffn1, ffn2}` 各自独立。这在 Phase 4 的 CUDA Stream 并行和 Phase 5 的 Flash Attention 调度中都极其重要。

### 2.5 第四步：调度执行（Operator Dispatch）

有了拓扑序，执行就简单了——遍历排好序的节点列表，每个节点根据 `op_type` 调对应的算子：

```cpp
void execute_node(Node* node, Graph& graph) {
    // 1. 从 Graph 里取出输入张量
    std::vector<Tensor*> inputs;
    for (auto* pred : node->predecessors) {
        inputs.push_back(&graph.tensors[pred->id]);
    }

    // 2. 根据算子类型分发
    if (node->op_type == "MatMul") {
        gemm_naive_ikj(...);   // ← 调你的 Phase 1 函数
    }
    else if (node->op_type == "Softmax") {
        softmax(...);          // ← 调你的 Phase 2 函数
    }
    else if (node->op_type == "LayerNorm") {
        layernorm(...);        // ← 调你的 Phase 2 函数
    }
    else if (node->op_type == "GELU") {
        gelu_exact(...);       // ← 调你的 Phase 2 函数
    }
    else if (node->op_type == "ScaledDotProductAttention") {
        scaled_dot_product_attention(...);  // ← 调你的 Phase 2 函数
    }

    // 3. 输出存入 Graph
    graph.tensors[node->id] = output_tensor;
}
```

**换一个模型结构会怎样？** 你不需要改任何 C++ 代码——只改 JSON 配置文件。图引擎会重新解析、重新拓扑排序、重新调度。这就是"框架"的核心价值。

---

## 三、内存池——不让内存分配拖慢你的推理

### 3.1 朴素做法的问题

上面每次 `graph.tensors[node->id] = output_tensor` 都会在堆上分配一块新内存。

一个 Attention Block 会产生多少中间张量？

```
q_proj, k_proj, v_proj      3 个 [seq_len × d_k] 张量
attn                         1 个 [seq_len × d_v]
add1                         1 个 [seq_len × d_model]
ln1                          1 个 [seq_len × d_model]
ffn1                         1 个 [seq_len × d_ff]
gelu1                        1 个 [seq_len × d_ff]
ffn2                         1 个 [seq_len × d_model]
add2                         1 个 [seq_len × d_model]
ln2                          1 个 [seq_len × d_model]
共计：11 个张量，其中保留的有 3 个（Q,K,V），其余 8 个只用于中间计算
```

每次 `new float[N]` 都要：
1. 调用 `malloc`，进入内核态
2. 内核找一块连续的虚存
3. 可能触发 `brk/mmap` 系统调用

11 次分配 × 每次 ~10μs = ~110μs 开销。对你的 1ms 推理来说，**10% 时间浪费在内存分配上**。更糟的是，多次分配会导致**内存碎片**——大矩阵可能找不到连续的可用块。

### 3.2 内存池的思想：一次分配，反复使用

```
普通做法：
  算子 1 跑完 → 输出 malloc 一块 → 算子 2 用它
  算子 1 的输出用完 → 没人用了 → 但 malloc 出去的内存不会自动回收
  → 你忘记 delete → 内存泄漏
  → 你手动 delete → 代码到处都是 new/delete → 难维护

内存池做法：
  推理开始前：预分配一块大 buffer（比如 10MB）
  算子 1 跑完 → 从 buffer 里"借"一段连续区域 → 算子 2 用它
  算子 2 跑完 → 归还这段区域 → 算子 3 复用同一段
```

### 3.3 内存池的核心数据结构

```cpp
class MemoryPool {
private:
    float* buffer_;               // 一大块预分配的内存
    size_t capacity_;             // 总字节数
    size_t used_;                 // 当前已分配字节数
    std::vector<size_t> freed_;   // 归还的空隙（简化版用栈式分配）

public:
    MemoryPool(size_t total_bytes) {
        buffer_ = new float[total_bytes / sizeof(float)];
        capacity_ = total_bytes;
        used_ = 0;
    }

    // 分配：返回一个偏移量（指针）
    float* allocate(size_t num_floats) {
        if (used_ + num_floats * sizeof(float) > capacity_) {
            throw std::runtime_error("MemoryPool exhausted");
        }
        float* ptr = buffer_ + (used_ / sizeof(float));
        used_ += num_floats * sizeof(float);
        return ptr;
    }

    // 重置：把所有分配标记为未使用（栈式回收）
    void reset() {
        used_ = 0;
    }

    ~MemoryPool() {
        delete[] buffer_;
    }
};
```

**这是最简单的栈式内存池**：只增长不回收中间碎片。推理结束后直接 `reset()` 全部清空。对于**顺序执行的推理框架**，这已经够用了。

### 3.4 进阶：生命周期分析——TensorRT 的核心技术

栈式内存池的问题是：假设你预分配了 10MB，11 个中间张量可能只需要 4MB 的总峰值——但栈式不会复用，最终可能用满 10MB。

**生命周期分析**解决了这个问题：

```
追踪每个张量的"存活期"：

张量        存活范围
──────────────────────────────────
q_proj      创建于 MatMul → 最后被 attn 使用
k_proj      创建于 MatMul → 最后被 attn 使用
v_proj      创建于 MatMul → 最后被 attn 使用
attn        创建于 attn  → 最后被 add1 使用
add1        创建于 add1   → 最后被 ln1 使用
ln1         创建于 ln1    → 最后被 add2 使用  ← 活得最久
ffn1        创建于 ffn1   → 最后被 gelu1 使用
gelu1       创建于 gelu1  → 最后被 ffn2 使用
ffn2        创建于 ffn2   → 最后被 add2 使用
add2        创建于 add2   → 最后被 ln2 使用
ln2         创建于 ln2    → 最后输出
```

**关键洞察**：`ffn1` 和 `gelu1` 的生命周期**不重叠**——`ffn1` 产生后立刻被 `gelu1` 消耗，之后 `ffn1` 再也不会被任何人使用。

于是内存可以这样复用：

```
时间轴 →

位置 0: [q_proj██████████████████████████████████████████] → attn 用完即释放
        [add1██████████████████████████] → ln1 用完即释放
        [ffn1██] → gelu1 用完即释放
        [ffn2██████████████████████████] → add2 用完即释放
        [ln2████████████████████████████████████████████████] → 最终输出

位置 1: [k_proj██████████████████████████████████████████████] → 和 q_proj 同时存活
        [ln1██████████████████████████████████████████████████████████████] → 最长的中间张量

位置 2: [v_proj██████████████████████████████████████████████] → 和 q/k_proj 同时存活
        [gelu1██████████] → 短命
        [add2██████████████████████████] → 复用 ffn1 的空间

峰值内存比栈式分配少 30-40%。
```

**这在工业界叫什么？** TensorRT 叫它 "memory planner"、ONNX Runtime 叫它 "arena allocator"。本质都一样——**生命周期不重叠的张量复用同一块内存**。

---

## 四、全局视角：Phase 3 如何串起前后所有阶段

### 4.1 上游：Phase 1 和 Phase 2 是 Phase 3 的"算子库"

```
Phase 1 ──→ gemm_naive_ikj ──→ Phase 3 的 MatMul 节点
Phase 2 ──→ softmax ────────→ Phase 3 的 Softmax 节点
Phase 2 ──→ layernorm ──────→ Phase 3 的 LayerNorm 节点
Phase 2 ──→ gelu_exact ─────→ Phase 3 的 GELU 节点
Phase 2 ──→ attention ──────→ Phase 3 的 Attention 组合节点
```

没有 Phase 1/2 的正确实现，Phase 3 就只是个空壳——知道该调谁但没有能调的函数。**Phase 3 不关心算子怎么实现，只关心算子有什么接口**。这就是为什么 Phase 2 强调"正确性"——一个错的 softmax 在上游看不出，到 Phase 3 整个模型输出全错。

### 4.2 下游：Phase 4/5 是 Phase 3 的"硬件加速版"

```
Phase 3 的计算图引擎                      Phase 4 的 CUDA 版本
────────────────────────                  ─────────────────────
 Node { op_type = "MatMul" }     →        调 cuda_gemm_kernel 代替 gemm_naive_ikj
 Node { op_type = "Softmax" }    →        调 cuda_softmax_kernel 代替 softmax
 Graph 结构                       →        完全不变！JSON 配置复用
 拓扑排序                         →        完全不变！
 内存池                           →        升级为 GPU 显存池（cudaMalloc 预分配）
```

**这意味着什么？** Phase 3 做对了，Phase 4 只需要**升级算子实现**——图的结构、调度逻辑、内存管理全部复用。这是工业框架的标准架构：**计算图是骨架，算子实现是血肉**。PyTorch 把算子写成 C++/CUDA kernel 替换 python 实现，也是这个道理。

```
Phase 5 Flash Attention 的接入：

Phase 3 图引擎的 Attention 组合节点：
  scores = Q × K^T / √d_k
  weights = softmax(scores)
  output = weights × V

Phase 5 替换为 Flash Attention kernel：
  flash_attention(Q, K, V, output)  → 一步完成，不产生中间 N×N 矩阵
  → 把三个阶段融合为一个 CUDA kernel
  → 图节点从 3 个变成 1 个（图融合 Graph Fusion）
  → 内存池峰值从 O(N²) 降到 O(N)
```

**这就是计算图框架的真正威力**——你可以把整个 attention block 融合成一个 kernel，只需在图里换一个节点类型。

### 4.3 一张总图

```
┌─────────────────────────────────────────────────────────────┐
│                       JSON 模型配置                          │
│  { "nodes": [ ... ], "inputs":[...], "outputs":[...] }     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Phase 3: 计算图引擎                       │
│                                                             │
│  JSON ──解析──→ DAG ──拓扑排序──→ 执行队列                   │
│                          │                                  │
│                          ▼                                  │
│  内存池: [█████████████████████████░░░░░░░░]                │
│          alloc("q_proj") → ptr                               │
│          alloc("attn") → ptr                                 │
│          free("q_proj") → 空间被 gelu1 复用                 │
└─────────────────────────────────────────────────────────────┘
           │                                    │
           │ 算子调用                             │ CUDA 替换
           ▼                                    ▼
┌─────────────────────┐              ┌─────────────────────┐
│   Phase 1 + 2        │              │   Phase 4 + 5        │
│                     │              │                     │
│  gemm_naive_ikj    │              │  cuda_gemm_kernel   │
│  softmax           │              │  cuda_softmax_kernel │
│  layernorm         │              │  cuda_layernorm      │
│  gelu_exact        │              │  flash_attention     │
│  attention          │              │                     │
│                     │              │  显存池: cudaMalloc │
│  堆内存: malloc/new │              │  显存: HBM          │
└─────────────────────┘              └─────────────────────┘
```

---

## 五、Phase 3 的完成标准与下一步

### 完成标志
- JSON 配置文件定义一个 Attention Block
- 计算图引擎能解析 JSON → 构建 DAG → 拓扑排序 → 按序调用 Phase 2 的算子
- 内存池工作正常，端到端跑通一个完整 Attention Block
- 对比：有内存池 vs 每次 new/delete 的内存峰值差异

### 核心交付物
```
phase3_graph/
├── include/
│   ├── tensor.h         # 张量数据结构
│   ├── graph.h          # Node / Graph 定义
│   ├── json_parser.h    # JSON → Graph 解析
│   ├── memory_pool.h    # 内存池实现
│   └── executor.h       # 算子调度 & 执行器
├── src/
│   ├── graph.cpp
│   ├── json_parser.cpp
│   ├── memory_pool.cpp
│   └── executor.cpp
├── tests/
│   ├── test_graph.cpp           # 图构建测试
│   ├── test_topological_sort.cpp # 拓扑排序测试
│   ├── test_executor.cpp        # 端到端执行测试
│   └── test_memory_pool.cpp     # 内存池对齐测试
├── models/
│   └── attention_block.json     # 模型配置
└── CMakeLists.txt
```

---

## 六、给你的一条核心建议

Phase 3 是**整个项目最重要的架构阶段**。代码量不大（图引擎核心可能 500-800 行），但**设计决策会影响 Phase 4/5 的难易程度**。

两条铁律：

1. **算子和图解耦**：图的执行器不要直接知道 softmax 怎么实现。用一个注册表（map<string, function>）把算子名映射到函数指针。这样 Phase 4 换 CUDA 版本时，只需改注册表，不改图引擎。

2. **内存池先做栈式再升级**：栈式内存池 50 行代码就能跑，先确保图和调度逻辑正确。生命周期分析留到后面优化——**先正确再快速**，Phase 1 的方法论在这也用得上。

---

下一步：你想让我直接搭 Phase 3 的骨架代码，还是你先读这份文档自己动手？
