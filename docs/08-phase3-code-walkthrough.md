# Phase 3 代码走查与架构详解

> 用途：向技术领导汇报 Phase 3 整体架构、运行流程、关键实现细节。
> 读者定位：刚接手项目，需要理解"代码怎么跑、数据怎么流、每个文件负责什么"的新人。

---

## 一、文件清单与职责矩阵

```
phase3_graph/
├── CMakeLists.txt                  → 构建脚本，织 Phase 1/2/3 为一个整体
├── configs/
│   └── attention_block.json        → 模型定义（5 个节点，描述一个 Attention Block）
├── include/
│   ├── graph.h                     → 数据结构：Tensor / Node / ComputeGraph
│   ├── json_parser.h               → 接口：JsonParser（JSON → ComputeGraph）
│   ├── executor.h                  → 接口：Executor（两种执行模式）
│   └── memory_pool.h               → 接口：MemoryPool（预分配 + 生命周期复用）
├── src/
│   ├── graph.cpp                   → ComputeGraph 实现（拓扑排序 + 张量存储）
│   ├── json_parser.cpp             → 手写 JSON Tokenizer + 递归下降解析器
│   ├── executor.cpp                → 算子调度（5 类算子 × 2 种模式 = 10 个 dispatch 函数）
│   └── memory_pool.cpp             → 内存池核心（alloc/dealloc/compact/峰值追踪）
└── tests/
    ├── test_graph.cpp              → 4 个测试：JSON 解析 / 拓扑序 / 手动建图 / 环检测
    └── test_executor.cpp           → 3 个测试：全流水线 / 简单图 / 中间张量验证
```

**总共 10 个源文件，~950 行 C++**（不含空行注释）。编译后产出 `libgraph_engine.a` 静态库 + 3 个测试可执行文件。

---

## 二、五阶段执行流程总览

从命令行敲下 `test_executor.exe` 到输出 `PASS`，数据经历了 5 个阶段：

```
阶段 1: JSON 解析          阶段 2: 拓扑排序         阶段 3: 生命周期分析
configs/*.json             ComputeGraph              Executor (pool 模式)
    │                          │                          │
    ▼                          ▼                          ▼
Tokenize → parse_node×5   入度计算 → Kahn BFS       遍历排序节点 → 记录每
→ Node{id,op,inputs}      → 顺序数组                 个张量的 first/last_use
→ graph.add_node()        → 检测环
    │                          │                          │
    │                          │                          │
    ▼                          ▼                          ▼
graph 里有 5 个 Node       sorted = [q_proj,            lifetimes =
  + 键值对 node_index      k_proj, v_proj,            { "x": [input],
                             attn_out, ln_out]          "q_proj": [0,0],
                                                        "attn_out": [3,4], ...}
       阶段 4: 算子调度                   阶段 5: 内存回收
       ┌─────────────────┐               ┌──────────────────┐
    sorted[0] → "MatMul" → gemm_naive_ikj   idx=0 执行后:
    sorted[1] → "MatMul" → gemm_naive_ikj     哪些张量的 last_use==0?
    sorted[2] → "MatMul" → gemm_naive_ikj     → 无，q_proj 还要被 attn 用
    sorted[3] → "Attention" → attention()    idx=3 执行后:
    sorted[4] → "LayerNorm" → layernorm()        q/k/v_proj 的 last_use==3
       │                                        → pool.deallocate("q_proj")
       │                                        → 内存池标记槽位为 free
       └─────────────────┘               └──────────────────┘
```

下面按文件拆开讲。

---

## 三、`graph.h` + `graph.cpp`——数据结构与图算法

### 3.1 Tensor 结构体（graph.h:17-39）

```cpp
struct Tensor {
    std::string name;               // "q_proj"
    std::vector<int> shape;         // {128, 64}
    std::vector<float> data;        // 1024 个 float
};
```

**为什么 `data` 是 `vector<float>` 而不是裸指针？**
因为 Phase 3 的普通模式（`execute`）需要每个中间张量拥有自己的存储——不同张量的生命周期不同，需要 RAII。等进入 pool 模式后 `data` 实际指向内存池里的地址（通过 `assign` 拷贝），但对外接口保持一致。

**`total_size()` 方法**：shape 元素乘积。`[128, 64]` → `128 × 64 = 8192`。

**`Tensor::zeros()` 静态工厂**：一次性完成"命名 + 分配 + 填零"。Phase 1/2 里你手写了 `C.assign(M*N, 0.0f)` 的模式，这里封装为接口。

### 3.2 Node 结构体（graph.h:45-53）

```cpp
struct Node {
    std::string id;                              // 唯一标识：q_proj
    std::string op_type;                         // 算子类型：MatMul
    std::vector<std::string> inputs;             // ["x", "Wq"]
    std::vector<std::string> outputs;            // ["q_proj"]（自动生成）
    std::unordered_map<std::string, float> params; // {"eps": 1e-5}
};
```

**关键设计：`inputs` 和 `outputs` 存的都是字符串名，不是指针。**
这是为了和 JSON 解耦——JSON 里只有字符串，解析时不需要知道其他节点的内存地址。运行时 `executor` 通过 `graph.get_tensor(name)` 查找实际数据。

**`outputs` 自动生成逻辑**（json_parser.cpp:110-113）：
```cpp
if (node.outputs.empty()) {
    node.outputs.push_back(node.id);  // 默认输出名 = 节点 ID
}
```
这样 JSON 里不需要写 `"outputs"`，减少配置冗余。

### 3.3 ComputeGraph 类（graph.h:59-85）

```cpp
class ComputeGraph {
    std::vector<Node> nodes_;                       // 所有节点（插入顺序）
    std::unordered_map<std::string, size_t> node_index_; // id → nodes_[i] 索引
    std::unordered_map<std::string, Tensor> tensors_;    // 张量名 → 数据
};
```

**两个内部索引**：
- `nodes_`：vector，O(1) 按索引访问，O(n) 按 ID 查找
- `node_index_`：map，补足按 ID 查找的 O(1)，典型"空间换时间"
- `tensors_`：map，存所有权重和中间数据

**`set_tensor` 使用 `std::move`**（graph.cpp:30）：
```cpp
void ComputeGraph::set_tensor(const std::string& name, Tensor t) {
    tensors_[name] = std::move(t);  // 移动语义，不拷贝 float 数组
}
```
如果你的输入张量有 8192 个 float，不用 move 就会深拷贝 ~32KB，每个节点都拷一遍就浪费了。

### 3.4 拓扑排序实现（graph.cpp:49-107）——整份代码的技术核心

**算法**：Kahn 算法（基于入度的 BFS），时间复杂度 O(V + E)。

**第 1-2 步（L51-68）：建辅助索引 + 算入度**

```cpp
// 构建 name → Node* 映射，用于判断"某个 input 是不是由其他节点产生的"
std::unordered_map<std::string, const Node*> name_to_node;
for (const auto& n : nodes_) {
    name_to_node[n.id] = &n;
}

// 算入度：遍历每个节点，检查它的每个 input
// 如果 input 是某个节点的 ID → 这个节点依赖那个节点 → 入度+1
std::unordered_map<std::string, int> in_degree;
for (const auto& n : nodes_) {
    in_degree[n.id] = 0;
}
for (const auto& n : nodes_) {
    for (const auto& input : n.inputs) {
        if (name_to_node.count(input)) {  // input 是节点产出 → 形成依赖
            in_degree[n.id]++;
        }
    }
}
```

**以 attention_block.json 为例走一遍**：

```
节点          inputs                     依赖哪个节点？              入度
─────────────────────────────────────────────────────────────────────────
q_proj        ["x", "Wq"]                x/Wq 不在 nodes 里          0
k_proj        ["x", "Wk"]                x/Wk 不在 nodes 里          0
v_proj        ["x", "Wv"]                x/Wv 不在 nodes 里          0
attn_out      ["q_proj","k_proj","v_proj"] q/k/v 都在 nodes 里!       3
ln_out        ["attn_out"]               attn_out 在 nodes 里!        1
```

**第 3 步（L71-77）：入度为 0 的节点入队**
```
init: queue = [q_proj, k_proj, v_proj]   ← 三个投影可以并行！
```

**第 4 步（L79-97）：BFS，每取出一个节点，把它所有后继的入度 -1**
```
取出 q_proj(入度0) → 加入 sorted → 遍历所有节点找"谁依赖 q_proj"
  → attn_out 的 inputs 里有 q_proj → attn_out 入度 3→2

取出 k_proj → attn_out 入度 2→1
取出 v_proj → attn_out 入度 1→0 → attn_out 入队

取出 attn_out → 加入 sorted → ln_out 的 inputs 里有 attn_out
  → ln_out 入度 1→0 → ln_out 入队

取出 ln_out → 加入 sorted → 队列空，结束
```

**最终 sorted = [q_proj, k_proj, v_proj, attn_out, ln_out]**

**第 5 步（L100-104）：环检测**
```cpp
if (sorted.size() != nodes_.size()) {
    fprintf(stderr, "[ERROR] Graph has a cycle!\n");
    return {};  // 空 vector 表示失败
}
```
如果 sorted 数量少于节点数，说明有节点入度永远不为 0 → 存在环。这防止了 A→B→A 这种配置导致死循环。

---

## 四、`json_parser.h` + `json_parser.cpp`——手写 JSON 解析器

### 4.1 为什么手写？

注释里写了原因（json_parser.cpp:11-13）：
> Phase 3 is about understanding graph execution, not JSON parsing.
> A hand-rolled parser keeps dependencies minimal and is ~100 lines.

### 4.2 Tokenizer 类（json_parser.cpp:23-72）

一个逐字符扫描的状态机，支持 4 种 token：

| Token 类型 | 起始字符 | `Tokenizer` 方法 | 返回值 |
|---|---|---|---|
| 字符串 | `"` | `read_string()` | `std::string` |
| 数字 | `-` 或 数字 | `read_number()` | `float` |
| 结构符号 | `{`, `}`, `[`, `]`, `:` | `expect(c)` | void（不匹配抛异常） |
| 分隔符 | 空格, `\t`, `\n`, `\r`, `,` | 被 `skip_whitespace()` 吃掉 | — |

**关键方法 `skip_whitespace()`**（L64-68）：
```cpp
void skip_whitespace() {
    while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' ||
           s_[pos_] == '\n' || s_[pos_] == '\r' || s_[pos_] == ','))
        pos_++;
}
```
**注意**：逗号被当作空白一起跳过！这就是为什么 JSON 里可以写 `"key": value,` 而 parser 看到的始终是干净的 token。简化了递归下降的逻辑。

### 4.3 主解析逻辑 `JsonParser::parse()`（json_parser.cpp:118-151）

```cpp
bool JsonParser::parse(const std::string& json, ComputeGraph& graph) {
    Tokenizer tok(json);
    tok.expect('{');                    // 读 {

    while (tok.peek() != '}') {
        std::string key = tok.read_string();   // 读 "nodes"
        tok.expect(':');                       // 读 :

        if (key == "nodes") {
            tok.expect('[');                   // 读 [
            while (tok.peek() != ']') {
                parse_node(tok, graph);         // 读一个 node 对象
            }
            tok.expect(']');                   // 读 ]
        } else {
            // 跳过不认识的 key（向前兼容）
        }
    }
    tok.expect('}');
}
```

**向前兼容设计**：如果 JSON 加了一个 `"version": "2.0"` 字段，parser 不会崩溃——`else` 分支会跳到 `}` 继续。

### 4.4 节点解析 `parse_node()`（json_parser.cpp:79-115）

```cpp
static void parse_node(Tokenizer& tok, ComputeGraph& graph) {
    Node node;
    tok.expect('{');

    while (tok.peek() != '}') {
        std::string key = tok.read_string();  // "id" / "op" / "inputs" / "params"
        tok.expect(':');

        if (key == "id")      node.id = tok.read_string();
        if (key == "op")      node.op_type = tok.read_string();
        if (key == "inputs")  { /* [...] → vector<string> */ }
        if (key == "params")  { /* {...} → map<string,float> */ }
    }
    tok.expect('}');

    // auto-generate output
    if (node.outputs.empty()) node.outputs.push_back(node.id);
    graph.add_node(node);
}
```

**params 解析细节**（L97-105）：只支持 float 值。这符合当前需求（`eps: 1e-5`），但如果将来需要字符串参数（如 `"activation": "gelu"`），需要扩展这里。

---

## 五、`memory_pool.h` + `memory_pool.cpp`——内存池实现

### 5.1 数据结构（memory_pool.h:45-83）

```cpp
class MemoryPool {
    struct Slot {
        std::string name;     // "q_proj"
        float* ptr;           // 指向 pool_ 内部某个位置
        size_t size;          // 元素个数
        bool in_use;          // 是否还在被使用
    };

    float* pool_;             // 整块 buffer（aligned_alloc 分配）
    size_t total_size_;       // 总容量（float 个数）
    size_t peak_used_;        // 历史峰值（stats 用）
    size_t alloc_count_;      // 累计分配次数
    std::vector<Slot> slots_; // 所有槽位（包括 in_use 和 free 的）
};
```

**关键设计**：`Slot` 不维护链表，用简单的 `vector` + 线性扫描。因为 Phase 3 的图只有 5 个节点，中间张量 < 10 个，O(n) 扫描完全可以接受。

### 5.2 内存分配（memory_pool.cpp:15-28）

```cpp
MemoryPool::MemoryPool(size_t total_floats) {
    // 32 字节对齐——为 Phase 4 的 AVX2 _mm256_load_ps 做准备
#ifdef _WIN32
    pool_ = static_cast<float*>(_aligned_malloc(total_floats * sizeof(float), 32));
#else
    pool_ = static_cast<float*>(std::aligned_alloc(32, total_floats * sizeof(float)));
#endif
    std::memset(pool_, 0, total_floats * sizeof(float));  // 清零
}
```

**为什么 32 字节对齐？** SIMD `_mm256_load_ps` 要求 32 字节对齐。Phase 4/5 的 CUDA 版本也会要求对齐——现在做好，后续不用改数据结构。

### 5.3 allocate() 的两种策略（memory_pool.cpp:42-106）

**策略 1：复用已释放的槽位（best-fit）**（L52-59）

```cpp
for (auto& slot : slots_) {
    if (!slot.in_use && slot.size >= count) {
        slot.name = name;
        slot.in_use = true;
        return slot.ptr;   // 原地复用，零拷贝
    }
}
```

**为什么用 `>=` 而不是 `==`？** 一个 4096 float 的旧槽可以被 2048 float 的新张量复用。内存池不关心张量语义，只关心空间大小。

**策略 2：在 pool 尾部追加**（L62-98）

```cpp
size_t offset = 0;
for (const auto& slot : slots_) {
    offset = std::max(offset, (size_t)(slot.ptr - pool_) + slot.size);
}
// 如果不够 → 尝试 compaction（碎片整理）
if (offset + count > total_size_) {
    // 把所有 in_use 的槽位移到前面，消除空隙
    compact_offset = 0;
    for (auto& slot : slots_) {
        if (slot.in_use) {
            std::memmove(pool_ + compact_offset, slot.ptr, slot.size * sizeof(float));
            slot.ptr = pool_ + compact_offset;
            compact_offset += slot.size;
        }
    }
}
```

**compaction（碎片整理）**：当内存碎片太多、无法连续分配时，把所有 in_use 的槽 `memmove` 到 pool 前端。这和 Java GC 的 compact 阶段原理一样——只是这里手动触发。

### 5.4 deallocate()——不归还 OS（memory_pool.cpp:112-119）

```cpp
void MemoryPool::deallocate(const std::string& name) {
    for (auto& slot : slots_) {
        if (slot.in_use && slot.name == name) {
            slot.in_use = false;  // 只改 flag，不回收内存
            return;
        }
    }
}
```

**关键设计**：`in_use = false` 后，`pointer` 和 `size` 不变——下次 allocate 时策略 1 会复用这个槽位。**内存永远在 pool 里，不归还 OS**。对推理框架来说这是对的——同一模型反复推理，内存复用比重复 malloc 高效得多。

---

## 六、`executor.h` + `executor.cpp`——算子调度核心

### 6.1 两种执行模式

| 模式 | 方法 | 内存策略 | 用途 |
|---|---|---|---|
| 普通模式 | `execute()` | 每个输出张量用 `std::vector<float>` 独立分配 | 先验证正确性 |
| Pool 模式 | `execute_with_pool()` | 所有输出从 MemoryPool 分配，生命周期分析后自动释放 | 优化内存 |

### 6.2 生命周期分析函数 `analyze_lifetimes()`（executor.cpp:43-78）

```cpp
struct TensorLifetime {
    int first_use;   // 在 sorted_nodes 数组中第一次被使用的索引
    int last_use;    // 最后一次被使用的索引
    bool is_input;   // 外部输入（x, Wq）——永远不释放
};

static auto analyze_lifetimes(sorted_nodes, graph) {
    for (int idx = 0; idx < sorted_nodes.size(); idx++) {
        const Node* node = sorted_nodes[idx];

        // 每个 node 的 inputs：更新这些张量的 last_use
        for (auto& input : node->inputs) {
            if (graph 里没有这个张量) continue;  // 权重/输入可能未加载
            lifetimes[input].last_use = idx;       // 更新到最后一次读取
        }

        // 每个 node 的 outputs：初始化为 first_use = idx, last_use = idx
        for (auto& output : node->outputs) {
            lifetimes[output] = {idx, idx, false};
        }
    }
}
```

**以 attention_block.json 走一遍**：

```
idx=0 执行 q_proj:
  inputs: "x" → lifetimes["x"] = {0, 0}  （首次记录）
          "Wq" → Wq 不在 graph tensors 里，跳过
  outputs: "q_proj" → lifetimes["q_proj"] = {0, 0}

idx=3 执行 attn_out:
  inputs: "q_proj" → lifetimes["q_proj"].last_use = 0 → 3  ← 关键！
          "k_proj" → k_proj.last_use = 1 → 3
          "v_proj" → v_proj.last_use = 2 → 3
  outputs: "attn_out" → lifetimes["attn_out"] = {3, 3}

idx=4 执行 ln_out:
  inputs: "attn_out" → attn_out.last_use = 3 → 4
  outputs: "ln_out" → lifetimes["ln_out"] = {4, 4}
```

**最终 lifetimes 表**：

```
张量        first_use  last_use  何时释放？
───────────────────────────────────────────
x            0          3         idx=3 时释放（不再需要）
q_proj       0          3         idx=3 时释放（attn_out 用完了）
k_proj       1          3         idx=3 时释放
v_proj       2          3         idx=3 时释放
attn_out     3          4         idx=4 时释放（ln_out 用完了）
ln_out       4          4         输出，最后释放
```

### 6.3 普通模式算子调度（executor.cpp:85-160）

以 `execute_matmul` 为例：

```cpp
static bool execute_matmul(const Node& node, ComputeGraph& graph) {
    Tensor* A = require_tensor(graph, node.inputs[0], node.id);  // "x"
    Tensor* B = require_tensor(graph, node.inputs[1], node.id);  // "Wq"

    int M = A->shape[0];      // 128
    int K_A = A->shape[1];    // 64
    int K_B = B->shape[0];    // 64
    int N = B->shape[1];      // 64

    if (K_A != K_B) return false;  // 维度不匹配 → 报错

    Tensor C = Tensor::zeros(node.outputs[0], {M, N});  // 128×64 全零矩阵
    gemm_naive_ikj(M, N, K_A, A->data, B->data, C.data);  // ← 调 Phase 1
    graph.set_tensor(node.outputs[0], std::move(C));     // 存入 graph
    return true;
}
```

**一条完整的数据追踪（q_proj 节点）**：
1. `require_tensor("x")` → 找到 `shape=[128,64], data=[...]` 的张量
2. `require_tensor("Wq")` → 找到 `shape=[64,64], data=weights[...]`
3. `Tensor::zeros("q_proj", {128,64})` → `data.resize(8192, 0.0f)`
4. `gemm_naive_ikj(128, 64, 64, x.data, Wq.data, C.data)` → Phase 1 的 ikj GEMM
5. `C.data` 现在包含 `X × Wq` 的结果
6. `graph.set_tensor("q_proj", std::move(C))` → `tensors_["q_proj"]` 接管所有权

**其余 4 个算子（softmax/layernorm/gelu/attention）的模式完全一样**：取输入 → 分配输出 → 调 Phase 2 函数 → 存入 graph。

### 6.4 Pool 模式——与普通模式的关键区别（executor.cpp:166-290）

以 `execute_matmul_pool` 为例：

```cpp
static bool execute_matmul_pool(const Node& node, ComputeGraph& graph, MemoryPool& pool) {
    // ... 维度计算同上 ...

    // 关键区别 1：输出数据从 pool 分配，不是 vector<float>
    float* C_data = pool.allocate(node.outputs[0], M * N);

    // 关键区别 2：先计算到 temp vector，再 memcpy 到 pool
    // 原因：gemm_naive_ikj 的接口要求 std::vector<float>&（Phase 1 接口）
    Tensor C;
    C.data.assign(C_data, C_data + M * N);  // 包装为 vector<float>
    gemm_naive_ikj(M, N, K_A, A->data, B->data, C.data);
    std::memcpy(C_data, C.data.data(), M * N * sizeof(float));  // 拷回 pool

    graph.set_tensor(node.outputs[0], std::move(C));
    return true;
}
```

**为什么 pool 版本需要"从 pool 分配→用 vector 计算→拷回 pool"这个间接步骤？**
因为 Phase 1 的 `gemm_naive_ikj` 接口是 `const std::vector<float>&`——它需要一个连续内存的 vector。`MemoryPool` 返回的是裸 `float*`。未来优化方向：修改 Phase 1 接口接受 `float*`，消除这次额外拷贝。

**softmax_pool 的 same-buffer 安全检查（executor.cpp:204-211）**：
```cpp
bool same_buffer = (out_data == in->data.data());
if (same_buffer) {
    // Softmax does 3 passes: find max, exp+sum, normalize.
    // Pass 1 only reads, passes 2/3 read then overwrite.
    // Since we've already memcpyd input to output, it's safe.
}
```
当内存池把输入和输出分配到同一个地址（复用）时，必须确保算子支持 in-place 操作——或者手动先 memcpy 一份。这里通过注释向代码阅读者解释了为什么 softmax 的 3-pass 结构天然支持 in-place。

### 6.5 `execute_with_pool` 主循环（executor.cpp:329-376）

```cpp
bool Executor::execute_with_pool(sorted_nodes, graph, pool) {
    auto lifetimes = analyze_lifetimes(sorted_nodes, graph);  // 阶段 3

    for (int idx = 0; idx < sorted_nodes.size(); idx++) {
        const Node* node = sorted_nodes[idx];

        // 分发到对应的 pool 算子
        if (node->op_type == "MatMul")       ok = execute_matmul_pool(...);
        else if (node->op_type == "Softmax")  ok = execute_softmax_pool(...);
        // ... 其他算子 ...

        // 释放 last_use == idx 的张量
        for (auto& [name, lt] : lifetimes) {
            if (!lt.is_input && lt.last_use == idx) {
                pool.deallocate(name);  // 标记槽位为 free
            }
        }
    }
}
```

**这条释放逻辑是整个内存池的灵魂**：执行完 attn_out（idx=3）后，自动扫描 lifetimes 表发现 q_proj/k_proj/v_proj 的 `last_use==3` → 调用 `pool.deallocate` → 三个槽位变成 free → 下一个节点（ln_out）分配时直接复用这些空间。

---

## 七、`CMakeLists.txt`——三个阶段的编织

```
libgemm_phase1.a    ← phase1_gemm/src/gemm_naive.cpp + gemm_tiled.cpp
liboperators.a      ← phase2_operators/src/operators.cpp（链接 libgemm_phase1）
libgraph_engine.a   ← graph.cpp + json_parser.cpp + executor.cpp + memory_pool.cpp
                       （链接 liboperators + libgemm_phase1）

test_graph          ← 链接 libgraph_engine
test_executor       ← 链接 libgraph_engine + liboperators + libgemm_phase1
test_memory_pool    ← 同上
```

**依赖关系很干净**：Phase 3 不碰 Phase 1/2 的源码，只链接其 `.a` 文件。这意味着：
- Phase 1 换一种 GEMM 实现 → 重编 libgemm_phase1.a → Phase 3 自动用新版本
- Phase 4 CUDA 版本 → 换一个 `libgemm_cuda.a` → Phase 3 图引擎**代码一行不改**

---

## 八、测试文件——验证点与边界覆盖

### 8.1 test_graph.cpp（4 个测试）

| 测试 | 覆盖 | 验证点 |
|---|---|---|
| test_json_parsing | 5 节点 JSON → ComputeGraph | 节点数、类型、inputs 数、params |
| test_topological_sort | 拓扑排序输出 | 3 个投影在 attn_out 前，attn_out 在 ln_out 前 |
| test_manual_graph | 不用 JSON，手动建图 | A→B→C 的线性排序 |
| test_cycle_detection | A→B→A 环 | `topological_sort()` 返回空 vector |

### 8.2 test_executor.cpp（3 个测试）

| 测试 | 数据 | 验证点 |
|---|---|---|
| test_full_pipeline | 128×64 随机输入 | JSON→排序→执行→输出 shape + LayerNorm 统计量 |
| test_simple_graph | 32×32，3 节点手动图 | MatMul→GELU→LayerNorm 端到端 + 统计量检验 |
| test_intermediate_tensors | 16×32 全流水线 | 5 个中间张量全部存在 + shape 正确 |

**LayerNorm 输出验证（关键正确性检查）**：
```cpp
float mean = 0, var = 0;
for (auto v : out->data) mean += v;
mean /= out->data.size();
for (auto v : out->data) var += (v - mean) * (v - mean);
var /= out->data.size();
CHECK(std::abs(mean) < 1e-4f, "LayerNorm output mean ~0");
CHECK(std::abs(var - 1.0f) < 0.05f, "LayerNorm output var ~1");  // 容差 0.05
```
方差容差 0.05 比均值的 1e-4 宽松——因为 LayerNorm 的 `var + eps` 和浮点累积误差会轻微影响方差精度，这是正常现象。

---

## 九、与前后阶段的全面连接

### 9.1 Phase 3 从 Phase 1/2 继承了什么

| Phase 1/2 交付 | Phase 3 使用位置 | 使用方式 |
|---|---|---|
| `gemm_naive_ikj(M,N,K,A,B,C)` | executor.cpp:102 | `execute_matmul` 和 `execute_matmul_pool` |
| `softmax_2d(data, rows, cols)` | executor.cpp:115, 216 | `execute_softmax` 和 `_pool` |
| `layernorm(x,N,eps,γ,β,out)` | executor.cpp:130, 240 | `execute_layernorm` 和 `_pool` |
| `gelu_vector_exact(in,out,N)` | executor.cpp:140, 258 | `execute_gelu` 和 `_pool` |
| `scaled_dot_product_attention(Q,K,V,...)` | executor.cpp:153, 278 | `execute_attention` 和 `_pool` |

**调用方式**：所有算子通过 `if-else` 字符串匹配分发——**没有用虚函数或注册表**。这是 Phase 3 的一个简化选择：节点数 < 10 时 if-else 比虚函数表更快（没有间接跳转开销）。如果节点类型超过 20 种，应该改成 `std::unordered_map<std::string, std::function<...>>`。

### 9.2 Phase 3 为 Phase 4/5 留了什么接口

**ComputGraph 和拓扑排序代码完全不需要改**。Phase 4 只需：

```cpp
// Phase 4 新文件：executor_cuda.cpp
static bool execute_matmul_cuda(const Node& node, ComputeGraph& graph, CudaMemoryPool& pool) {
    Tensor* A = ...;  // 同上
    Tensor* B = ...;
    // 唯一的区别：调 cuda_gemm 而不是 gemm_naive_ikj
    cuda_gemm(M, N, K_A, A->data, B->data, C_data, cuda_stream);
}
```

**MemoryPool → CudaMemoryPool 的升级路径**：
- 构造函数：`_aligned_malloc` → `cudaMalloc`
- `allocate()`：返回 `float*` → 返回 host 指针（用 `cudaMallocHost`）或 device 指针
- `deallocate()`：`in_use = false` → 逻辑不变
- Compaction：`std::memmove` → `cudaMemcpy`

**Flash Attention（Phase 5）接入方式**：
当前的 attention 算子是 3 步：`Q×K^T → softmax → ×V`。Phase 5 可以把 Attention 节点直接分发到 `flash_attention_kernel`——**图引擎不关心你是一个算子还是三个**，只看 `op_type == "Attention"`。

---

## 十、汇报要点总结

向技术领导汇报时，建议按以下结构组织：

**一句话总结**：
Phase 3 用 ~950 行 C++ 实现了完整的计算图引擎，包含 JSON 解析、DAG 拓扑排序、算子调度（5 类算子 × 2 种内存模式）和带生命周期分析的内存池，重用了 Phase 1/2 的全部算子，并为 Phase 4/5 的 CUDA 加速预留了清晰的扩展接口。

**三个技术亮点**：
1. Kahn 拓扑排序 + 环检测，确保任意合法 DAG 都能正确执行
2. 内存池的 best-fit reuse + compaction，TensorRT 同款技术
3. 手写 JSON 解析器（零外部依赖），但通过 forward-compatible 设计支持配置扩展

**当前局限性**（诚实列出）：
- Pool 模式有额外的 memcpy（Phase 1 接口限制），需 Phase 4 重构消除
- 算子调度用 if-else 字符串匹配（简单但扩展性有限）
- 不区分训练/推理模式（当前只做推理）
