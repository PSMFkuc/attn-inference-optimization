# Phase 2 直观指南：用例子讲透五个算子

> 这份文档不讲公式推导。我们用具体数字例子告诉你：**输入数据长什么样，经过算子变成了什么样，为什么这样变。**

---

## 先忘掉公式，想想你要干嘛

Transformer 的 Attention Block 在干什么？大白话就是：

> 一句话进来，给每个词算一个"重点看谁"的注意力分数，然后按这个分数把信息聚合起来，最后整理一下输出。

比如输入 "我 爱 北京"（3 个 token），整个流程是：

```
"我" "爱" "北京"
  │    │    │
  ▼    ▼    ▼    (投影到三个空间)
 Q    K    V     (Q=问什么, K=答什么, V=有什么内容)
  │    │    │
  └────┼────┘
       ▼
  "谁和谁关系近"矩阵  (3×3，比如"爱"→"北京"关系最紧)
       │
       ▼ 归一化
  "注意力权重"矩阵    (每行和为1，概率分布)
       │
       ▼ 乘以 V
  "聚合后的信息"      (每个词吸收了其他词的内容)
       │
       ▼ LayerNorm + FFN
  "整理好的输出"
```

Phase 2 就是做这个流程里的每一步。你已经在 Phase 1 做好矩阵乘法了，现在补剩下的四个。

---

## 算子 0：回顾——你 Phase 1 已经有的 MatMul

你已经有了 `gemm_naive_ikj(M, N, K, A, B, C)` 和 `gemm_tiled(...)`。

在 Phase 2 的 Attention 里矩阵乘法出现三次：

### 第一处：Q = X × Wq

```
输入 X:     [seq_len × d_model]    比如 3×768
权重 Wq:    [d_model × d_k]        比如 768×64
输出 Q:     [seq_len × d_k]        比如 3×64
```

### 第二处：S = Q × K^T

这是关键——K 需要**转置**。你的 gemm 假设 B 是 K×N 布局（行主序），但这里 B 实际上是 K 矩阵，你需要的是它的转置 `K^T`。

```
Q:      [seq_len × d_k]    3×64
K^T:    [d_k × seq_len]    64×3  (不是 3×64，是 64×3)
S:      [seq_len × seq_len] 3×3
```

两种做法：
- **做法 A**：先转置 K 到新矩阵 `K_T[d_k][seq_len]`，再调你的 gemm（多一次拷贝，简单）
- **做法 B**：写一个新的 gemm 变体 `gemm_transB`，内层循环访问 B 时按 `B[j*ldb + k]` 而不是 `B[k*N + j]`（零拷贝，高效）

Phase 2 先用做法 A，Phase 3 再优化成做法 B。

### 第三处：O = W × V

```
W (attention weights):  [seq_len × seq_len]    3×3
V:                      [seq_len × d_v]         3×64
O:                      [seq_len × d_v]         3×64
```

直接调你的 gemm 就行。

---

## 算子 1：Softmax——把"分数"变"概率"

### 不抽象的讲：它在干嘛

你有一行分数 `[2.0, 1.0, 0.5]`，想变成概率。Softmax 做三件事：
1. 放大差距（指数函数让大的更大、小的更小）
2. 全部变正数
3. 归一化到和为 1

### 用一个具体例子跟踪数据

```
输入: [2.0, 1.0, 0.5]    ← 三个"谁和谁关系近"的分数

第 1 步：减最大值（数值稳定）
max = 2.0
减后: [0.0, -1.0, -1.5]

第 2 步：指数变换
exp(0.0)  = 1.00
exp(-1.0) = 0.368
exp(-1.5) = 0.223
变成: [1.00, 0.368, 0.223]

第 3 步：归一化
sum = 1.00 + 0.368 + 0.223 = 1.591
[1.00/1.591, 0.368/1.591, 0.223/1.591]
= [0.629, 0.231, 0.140]    ← 概率分布，和为 1

检查：0.629 + 0.231 + 0.140 = 1.000  ✅
```

### 为什么减最大值不影响结果？

```
原始：softmax(x_i) = exp(x_i) / exp(x_0)+exp(x_1)+exp(x_2)
减 m 后：(exp(x_i-m)) / (exp(x_0-m)+exp(x_1-m)+exp(x_2-m))

提取公因子：
分子：exp(x_i) · exp(-m)
分母：exp(-m) · (exp(x_0)+exp(x_1)+exp(x_2))
分子分母的 exp(-m) 约掉 → 结果一模一样
```

**减 m 只是"把计算过程搬到安全范围"，不影响最终结果。**

### 完整代码（带逐行解释）

```cpp
#include <cmath>
#include <algorithm>
#include <cfloat>    // FLT_MIN 等

// softmax 对一个 float 数组做 softmax，结果覆盖回原数组
void softmax(float* x, int N) {
    // 第 1 步：找最大值
    // 为什么用 float 最小值初始化？确保任何真实数据都比它大
    float max_val = -FLT_MAX;  // FLT_MAX 约 3.4e38，-FLT_MAX = 极小
    for (int i = 0; i < N; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    // 此时 max_val = 数组中的最大值

    // 第 2 步：指数变换 + 累加和
    // 同时做 exp(x_i - max) 和 sum，一遍循环搞定，减少访存
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        x[i] = std::exp(x[i] - max_val);  // 减去 max 防溢出
        sum += x[i];                       // 累加
    }
    // 每个 x[i] 现在是 exp(x_i - max)，都是 (0, 1] 里的数，不会溢出

    // 第 3 步：除以 sum 归一化
    // 用乘法代替除法更快：预计算 1/sum
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < N; i++) {
        x[i] *= inv_sum;  // 乘法比除法快 5-20 倍
    }
}
```

### 对矩阵做 Softmax（Attention 里是每行单独做）

Attention 里的 QK^T 结果是 `[seq_len × seq_len]` 矩阵，需要**每行单独**做 softmax：

```cpp
void softmax_2d(float* matrix, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        softmax(&matrix[r * cols], cols);  // 每行独立做
    }
}
```

比如 3×3 矩阵：
```
输入:                     输出:
[ 2.0  1.0  0.5]         [0.629 0.231 0.140]  ← 每行和为 1
[ 1.0  3.0  2.0]  →      [0.090 0.665 0.245]  ← 每行和为 1
[ 0.5  1.0  1.0]         [0.220 0.362 0.418]  ← 每行和为 1
```

### 数值稳定性测试（面试必备知识点）

问题：如果所有元素相同（都是 1000），会不会崩溃？

```
输入: [1000, 1000, 1000]

不做减 max：
  exp(1000) = inf（溢出！）→ sum = inf → 除以 inf = NaN ❌

做减 max：
  max = 1000
  减后: [0, 0, 0]
  exp(0) = 1.0
  sum = 3.0
  结果: [0.333, 0.333, 0.333] ✅
```

这就是为什么减 max 是**必须的**——不是优化，是正确性问题。

---

## 算子 2：LayerNorm——把"散乱"的数据拉回标准范围

### 它在干嘛

训练中每层的值范围会越来越偏（有的层输出 0.001，有的输出 1000），不归一化就"层间不通"。LayerNorm 强制每行的数据分布回到 **均值=0，标准差=1**，然后按可学习的 γ、β 拉伸平移。

### 用一个例子跟踪

```
输入一个样本（假设 d_model=4，没有 γ/β）:
x = [1.0, 3.0, 2.0, 6.0]    ← 均值偏大，分布散

第 1 步：算均值
μ = (1.0+3.0+2.0+6.0) / 4 = 12.0/4 = 3.0

第 2 步：中心化（每个减均值）
[1.0-3.0, 3.0-3.0, 2.0-3.0, 6.0-3.0]
= [-2.0, 0.0, -1.0, 3.0]         ← 现在均值为 0

第 3 步：算方差
σ² = ((-2)²+0²+(-1)²+3²) / 4 = (4+0+1+9)/4 = 14/4 = 3.5
标准差 σ = √3.5 ≈ 1.871

第 4 步：除以标准差
[-2.0/1.871, 0.0, -1.0/1.871, 3.0/1.871]
= [-1.069, 0.0, -0.535, 1.604]     ← 均值=0，标准差≈1

验证:
均值 = (-1.069+0.0-0.535+1.604)/4 ≈ 0
方差 = (1.143+0.0+0.286+2.573)/4 ≈ 1
```

### 代码（三遍循环，清晰版）

```cpp
void layernorm(const float* x, int N, float eps,
               const float* gamma, const float* beta,
               float* out) {
    // 第 1 遍：算均值
    float mean = 0.0f;
    for (int i = 0; i < N; i++) {
        mean += x[i];
    }
    mean /= N;  // μ = Σx / N

    // 第 2 遍：算方差（需要先知道均值，所以不能和上一遍合并）
    float var = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= N;  // σ² = Σ(x-μ)² / N

    // 第 3 遍：归一化 + 仿射
    float inv_std = 1.0f / std::sqrt(var + eps); // 倒数，循环里用乘法
    for (int i = 0; i < N; i++) {
        float normalized = (x[i] - mean) * inv_std;  // (x-μ)/σ
        out[i] = gamma[i] * normalized + beta[i];    // γ·norm + β
    }
}
```

### γ 和 β 是什么？

两个可学习的向量（长 N）。γ 控制"缩放"，β 控制"平移"：

```
γ = [2.0, 1.0, 1.0, 0.5]  → 第 1 维放大 2 倍，第 4 维缩小一半
β = [0.1, 0.0, 0.0, -0.1] → 第 1 维上移 0.1，第 4 维下移 0.1
```

**为什么需要它们？** 纯归一化到均值=0 方差=1 不一定最优——模型可能想"这个特征应该更突出"或"这个应该再小一点"。γ 和 β 给模型这个自由度。

推理时 γ 和 β 是固定的（从训练好的模型读取）。Phase 2 你可以用 `gamma=[1,1,1,1], beta=[0,0,0,0]` 当占位（identity 映射，不影响归一化结果）。

---

## 算子 3：GELU——更聪明的 ReLU

### 它在干嘛

ReLU 说：x < 0 直接归零，简单粗暴。GELU 说：x < 0 时"可能激活，可能不激活"，概率取决于 x 离 0 多远。

### 数据对比

```
x:     -3.0  -2.0  -1.0  -0.5   0.0   0.5   1.0   2.0   3.0
ReLU:   0     0     0     0     0    0.5   1.0   2.0   3.0    ← x<0 直接死
GELU:  -0.004 -0.045 -0.159 -0.154 0.0  0.346 0.841 1.955 2.996  ← x<0 有小值
```

GELU 在 x≈0 附近是**平滑的**，允许小的负值输出。这对梯度传播更好——ReLU 在 x<0 时梯度为 0（神经元"死掉"），GELU 不会。

### 两条实现路径

**精确版（用 erf，C++ 标准库）**：
```cpp
float gelu(float x) {
    // erf：高斯误差函数，C++ 标准库提供
    return 0.5f * x * (1.0f + std::erf(x / M_SQRT2));
    // M_SQRT2 = √2 ≈ 1.41421356
}
```

**近似版（用 tanh，更快）**：
```cpp
float gelu_approx(float x) {
    const float c1 = 0.7978845608f;   // √(2/π)
    const float c2 = 0.044715f;        // 三次项系数

    float x3 = x * x * x;              // x³
    float inner = c1 * (x + c2 * x3);  // √(2/π)·(x + 0.044715·x³)
    return 0.5f * x * (1.0f + std::tanh(inner));
}
```

**什么时候用哪个？**
- 训练（Phase 2）：用精确版，保证和论文一致
- 推理（Phase 4+）：用近似版，精度差 < 0.1%，速度快 ~30%

Phase 2 两个都实现，跑个对比测试。

### GELU 的向量版本

在 Attention 里是整向量一起做：

```cpp
void gelu_vector(const float* in, float* out, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = gelu(in[i]);  // 逐元素，每个元素独立
    }
}
```

---

## 算子 4：Scaled Dot-Product Attention——把前面四个串起来

这是你的第一个"组合算子"。它不引入新的数学运算，只是**按正确顺序调用前面的算子**。

### 用具体数字跟着走一遍

假设 `seq_len=3, d_k=4`（超小，为了看清每个值）：

```
# 输入：已经投影好的 Q, K, V
Q = | 1.0  0.5  0.2  0.0 |   (token "我" 的问向量)
    | 0.3  1.0  0.5  0.1 |   (token "爱" 的问向量)
    | 0.1  0.2  1.0  0.5 |   (token "北京"的问向量)
    3×4 矩阵

K = | 0.8  0.3  0.1  0.0 |
    | 0.2  0.9  0.4  0.1 |
    | 0.0  0.1  0.8  0.4 |
    3×4 矩阵

V = | 2.0  1.0  0.0  0.5 |
    | 1.0  3.0  0.5  0.0 |
    | 0.0  1.0  2.0  1.0 |
    3×4 矩阵
```

### 步骤 1：S = Q × K^T / √d_k  （算"谁关心谁"）

首先做矩阵乘法 Q(3×4) × K^T(4×3)：

```
Q × K^T = | 1.0*0.8+0.5*0.3+0.2*0.1+0.0*0.0  ... |   (3×3)
         = | 0.97  0.67  0.25 |
           | 0.69  1.02  0.54 |
           | 0.22  0.54  0.99 |
```

然后除以 √4 = 2：

```
S = scores/2 = | 0.485  0.335  0.125 |
               | 0.345  0.510  0.270 |
               | 0.110  0.270  0.495 |
```

这个 3×3 矩阵 S 的含义：`S[1][2] = 0.270` 表示"爱"对"北京"的**原始关注分数**是 0.270。

### 步骤 2：对每行做 softmax（把分数变权重）

```
第 0 行 [0.485, 0.335, 0.125]：
  max=0.485, 减后 [0, -0.15, -0.36]
  exp: [1.0, 0.861, 0.698]
  sum=2.559
  → [0.391, 0.336, 0.273]   ← "我"对自己的关注最高(39.1%)

第 1 行 [0.345, 0.510, 0.270]：
  max=0.510, 减后 [-0.165, 0, -0.240]
  exp: [0.848, 1.0, 0.787]
  sum=2.635
  → [0.322, 0.379, 0.299]   ← "爱"对自己最高，三词差不多

第 2 行 [0.110, 0.270, 0.495]：
  max=0.495, 减后 [-0.385, -0.225, 0]
  exp: [0.681, 0.799, 1.0]
  sum=2.480
  → [0.274, 0.322, 0.404]   ← "北京"对自己最高
```

权重矩阵 W：
```
W = | 0.391  0.336  0.273 |
    | 0.322  0.379  0.299 |
    | 0.274  0.322  0.404 |
```

**每一行加起来都是 1。** 这是 Softmax 的保证。

### 步骤 3：O = W × V（用权重聚合信息）

```
O[0] = 0.391·V[0] + 0.336·V[1] + 0.273·V[2]
     = 0.391·[2.0,1.0,0.0,0.5] + 0.336·[1.0,3.0,0.5,0.0] + 0.273·[0.0,1.0,2.0,1.0]
     = [0.782,0.391,0,0.196] + [0.336,1.008,0.168,0] + [0,0.273,0.546,0.273]
     = [1.118, 1.672, 0.714, 0.469]

O[1] = 0.322·V[0] + 0.379·V[1] + 0.299·V[2]
     = [0.644,0.322,0,0.161] + [0.379,1.137,0.190,0] + [0,0.299,0.598,0.299]
     = [1.023, 1.758, 0.788, 0.460]

O[2] = 0.274·V[0] + 0.322·V[1] + 0.404·V[2]
     = [0.548,0.274,0,0.137] + [0.322,0.966,0.161,0] + [0,0.404,0.808,0.404]
     = [0.870, 1.644, 0.969, 0.541]

最终输出 O (3×4):
O = | 1.118  1.672  0.714  0.469 |
    | 1.023  1.758  0.788  0.460 |
    | 0.870  1.644  0.969  0.541 |
```

**关键的直觉**：O 的每一行是 V 的各行加权和，权重来自 attention 分数。"我"的输出里混入了"爱"和"北京"的信息——这就是"注意力"机制的核心：**每个 token 的输出包含了它关注的其他 token 的信息**。

---

## 完整实现

```cpp
// attention.h
#pragma once
#include <vector>
#include <cmath>

// Phase 1 的 GEMM（你已经写好的）
// C = A · B   其中 A: M×K, B: K×N, C: M×N
void gemm_ikj(int M, int N, int K,
              const std::vector<float>& A,
              const std::vector<float>& B,
              std::vector<float>& C);

// Phase 2 的新算子
void softmax_row(float* row, int N);           // 单行 softmax
void layernorm(const float* x, int N, float eps,
               const float* gamma, const float* beta,
               float* out);
float gelu(float x);
void gelu_vector(const float* in, float* out, int N);

// 组合算子
void scaled_dot_product_attention(
    const std::vector<float>& Q,    // [seq_len × d_k]
    const std::vector<float>& K,    // [seq_len × d_k]
    const std::vector<float>& V,    // [seq_len × d_v]
    int seq_len, int d_k, int d_v,
    std::vector<float>& output);     // [seq_len × d_v]
```

```cpp
// attention.cpp
#include "attention.h"
#include <algorithm>
#include <cfloat>

// --- Softmax（对单行） ---
void softmax_row(float* row, int N) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < N; i++)
        if (row[i] > max_val) max_val = row[i];

    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        row[i] = std::exp(row[i] - max_val);
        sum += row[i];
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < N; i++)
        row[i] *= inv_sum;
}

// --- LayerNorm ---
void layernorm(const float* x, int N, float eps,
               const float* gamma, const float* beta,
               float* out) {
    float mean = 0.0f;
    for (int i = 0; i < N; i++) mean += x[i];
    mean /= N;

    float var = 0.0f;
    for (int i = 0; i < N; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= N;

    float inv_std = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < N; i++)
        out[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i];
}

// --- GELU ---
float gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x / 1.414213562f));
}

void gelu_vector(const float* in, float* out, int N) {
    for (int i = 0; i < N; i++)
        out[i] = gelu(in[i]);
}

// --- Scaled Dot-Product Attention ---
void scaled_dot_product_attention(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    int seq_len, int d_k, int d_v,
    std::vector<float>& output)
{
    // 第 1 步：scores = Q × K^T / √d_k
    // K^T 就是 K 的转置，维度从 [seq_len×d_k] 变成 [d_k×seq_len]
    std::vector<float> K_T(d_k * seq_len);
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < d_k; j++)
            K_T[j * seq_len + i] = K[i * d_k + j];
    // 现在 K_T 是 d_k × seq_len 布局

    std::vector<float> scores(seq_len * seq_len);
    gemm_ikj(seq_len, seq_len, d_k, Q, K_T, scores);
    // scores = Q(seq×d_k) × K^T(d_k×seq) = seq×seq

    float scale = 1.0f / std::sqrt((float)d_k);
    for (int i = 0; i < seq_len * seq_len; i++)
        scores[i] *= scale;

    // 第 2 步：逐行 softmax
    std::vector<float> weights = scores;  // 拷贝，softmax 会原地修改
    for (int r = 0; r < seq_len; r++)
        softmax_row(&weights[r * seq_len], seq_len);

    // 第 3 步：output = weights × V
    gemm_ikj(seq_len, d_v, seq_len, weights, V, output);
    // weights: seq×seq, V: seq×d_v → output: seq×d_v
}
```

---

## 如何验证你的实现

### 和 PyTorch 逐算子对比

```python
import torch
import numpy as np

# === 测试 Softmax ===
x = torch.tensor([2.0, 1.0, 0.5])
ref = torch.softmax(x, dim=0)
print("PyTorch softmax:", ref)
# tensor([0.6285, 0.2312, 0.1402])

# === 测试 LayerNorm ===
x = torch.randn(4)  # 4 维随机向量
gamma = torch.ones(4)
beta = torch.zeros(4)
ref = torch.nn.functional.layer_norm(x, (4,), gamma, beta, eps=1e-5)
print("PyTorch LayerNorm:", ref)

# === 测试 GELU ===
x = torch.tensor([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])
ref = torch.nn.functional.gelu(x)
print("PyTorch GELU:", ref)

# === 测试 Attention ===
seq_len, d_k, d_v = 3, 4, 4
Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)
V = torch.randn(seq_len, d_v)

# 手动实现 attention 作为参考
scale = d_k ** 0.5
scores = Q @ K.T / scale      # Q × K^T / √d_k
weights = torch.softmax(scores, dim=1)  # 逐行 softmax
ref = weights @ V              # W × V
print("Manual attention output:\n", ref)

# 用 PyTorch 内置 multi-head attention（single head）验证概念
mha = torch.nn.MultiheadAttention(d_k, 1, batch_first=True)
output, _ = mha(Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0))
print("PyTorch MHA output (single head):\n", output.squeeze(0))
```

### 导出二进制文件给 C++ 读

```python
# 保存参考数据
np.save("softmax_input.npy", x.numpy())
np.save("softmax_ref.npy", ref.numpy())
# ... 同理保存其他算子的输入输出
```

C++ 端用对应的 loader 读出对比。容差标准：
- Softmax: 1e-5
- LayerNorm: 1e-5
- GELU: 1e-5（精确版）/ 1e-3（近似版）
- Attention: 1e-4（累积了 GEMM 和 Softmax 的误差）

---

## Phase 2 正确性 > 性能

再说一遍：**Phase 2 的首要目标是每个算子都和 PyTorch 对齐，不是性能。**

理由很简单——如果 Phase 2 的 Softmax 有细微偏差，Phase 5 的 Flash Attention 所有性能数据都会在错误的基础上产生，等于白做。

你现在要做的就是：
1. 实现 5 个算子（MatMul 已有，Softmax/LayerNorm/GELU/Attention）
2. 每个都写单元测试，对 PyTorch 到 1e-5
3. 用这些算子拼出一个完整的单头 Attention block 并跑通

下一步，告诉我你想看什么：是我直接帮你在 `phase2_operators/` 下面搭好完整的骨架代码（带 CMake 和测试），还是你先按这个文档自己写，有问题再找我？
