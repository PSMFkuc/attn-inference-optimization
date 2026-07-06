# Phase 2 预览：核心 Attention 算子库

> 这份文档是 Phase 2 的"地图"，让你动手前先知道全貌。具体算子的逐行教学，等你完成 Phase 1 后我们再深入。

---

## Phase 2 在整个项目里的位置

```
Phase 1: GEMM 优化      →  学会"性能思维"
Phase 2: Attention 算子  →  搭建 Attention 的"积木"  ← 你在这里
Phase 3: 计算图引擎     →  把积木拼成能跑的框架
```

Phase 2 产出的是**一组正确、模块化、单元测试覆盖的算子函数**。它们会在 Phase 3 被计算图引擎调用，在 Phase 4 被改写成 CUDA。所以这一阶段**重点是正确性，不是性能**——性能留给后面。

---

## 你要实现哪些算子？为什么是这些？

一个完整的 Attention block（Transformer 的核心）长这样：

```
输入 X
  │
  ├──► Wq ──► Q ┐
  ├──► Wk ──► K ├─► Attention ──► ┐
  └──► Wv ──► V ┘                  │
                                   ├─► + (残差) ─► LayerNorm ─► FFN(含GELU) ─► + (残差) ─► LayerNorm ─► 输出
                                   ┘
```

拆开看，需要的原子算子：
1. **MatMul**（线性投影 QKV、Attention 里的 QK^T、attention·V）—— Phase 1 已实现，复用
2. **Softmax**（attention 权重归一化）
3. **LayerNorm**（每个 sublayer 后归一化）
4. **GELU**（FFN 的激活函数）
5. **Scaled Dot-Product Attention**（把上面组合起来）

下面逐个讲"为什么需要它"和"实现时的关键点"。

---

## 算子 1：Softmax

### 数学
```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```
把任意实数向量变成"概率分布"（和为 1，每个非负）。

### Attention 里干嘛用？
`QK^T` 的结果是一个相似度矩阵，数值范围不固定。Softmax 把它变成"权重"——每行加起来为 1，表示"这个 token 对其他 token 的关注度分配"。

### 实现的关键陷阱：数值稳定性

**naive 写法是错的**：
```cpp
// 错误！x_i 很大时 exp(x_i) 会溢出变成 inf
float sum = 0;
for (int i = 0; i < N; ++i) sum += exp(x[i]);
for (int i = 0; i < N; ++i) out[i] = exp(x[i]) / sum;
```

如果 x = [1000, 1001, 1002]，`exp(1000)` 远超 float 上限（~3.4e38），直接 inf。

**正确做法：减去最大值**
```cpp
// 减最大值，让最大的 exp(0)=1，其余都 < 1，不会溢出
float max_val = *std::max_element(x, x + N);  // 找最大值
float sum = 0;
for (int i = 0; i < N; ++i) {
    out[i] = exp(x[i] - max_val);  // 数值稳定
    sum += out[i];
}
for (int i = 0; i < N; ++i) out[i] /= sum;
```

**为什么减最大值不影响结果？** 数学推导：
```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
             = exp(x_i - m) · exp(m) / [Σ exp(x_j - m) · exp(m)]
             = exp(x_i - m) / Σ exp(x_j - m)
```
`exp(m)` 被约掉了。所以减任意常数结果不变，减 max 是为了数值安全。

**这个 trick 面试必问，也是 Flash Attention 的核心思想之一。**

### Softmax 的"在线算法"（进阶，Phase 5 用到）
标准 softmax 要两遍遍历（一遍算 sum，一遍归一化）。Flash Attention 需要一遍搞定，用"在线 softmax"算法。Phase 2 先实现标准两遍版，Phase 5 再升级。

---

## 算子 2：Layer Normalization

### 数学
```
y_i = γ · (x_i - μ) / sqrt(σ² + ε) + β
其中 μ = mean(x), σ² = var(x), ε 是小常数防除零
γ, β 是可学习参数
```
对每个样本独立做归一化（不同于 BatchNorm 跨样本归一化）。

### Attention 里干嘛用？
每个 sublayer（attention 或 FFN）的输出加残差后，做 LayerNorm 稳定数值分布。Transformer 用 LayerNorm 而不是 BatchNorm，因为：
- 序列长度可变，BatchNorm 跨样本统计不方便
- 推理时 batch size 常为 1，BatchNorm 退化

### 实现关键点
```cpp
void layernorm(const float* x, const float* gamma, const float* beta,
               float* out, int N, float eps = 1e-5f) {
    // 1. 算均值
    float mean = 0;
    for (int i = 0; i < N; ++i) mean += x[i];
    mean /= N;

    // 2. 算方差
    float var = 0;
    for (int i = 0; i < N; ++i) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= N;

    // 3. 归一化 + 仿射变换
    float inv_std = 1.0f / sqrtf(var + eps);  // 预计算倒数，避免循环内除法
    for (int i = 0; i < N; ++i) {
        out[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i];
    }
}
```
**为什么预计算 `inv_std`？** 除法比乘法慢 5-20 倍。循环里用乘法代替除法是基本优化。

---

## 算子 3：GELU 激活函数

### 数学
```
GELU(x) = x · Φ(x)    其中 Φ(x) 是标准正态的 CDF
精确式：GELU(x) = 0.5x · [1 + erf(x / √2)]
近似式：GELU(x) ≈ 0.5x · [1 + tanh(√(2/π) · (x + 0.044715·x³))]
```

### 为什么用 GELU 不用 ReLU？
- ReLU：x<0 直接 0，硬截断，信息丢失
- GELU：平滑过渡，x<0 的小负值也能保留一点（概率性激活）
- Transformer 论文用 GELU，已成事实标准

### 实现关键点
```cpp
#include <cmath>

// 精确版（用 erf）。C++ 标准库的 erff 足够快。
float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x / 1.41421356f));  // 1.4142 = sqrt(2)
}

// 近似版（用 tanh）。推理框架常用，更快，精度足够。
float gelu_tanh(float x) {
    const float c = 0.7978845608f;  // sqrt(2/π)
    float inner = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}
```
**选哪个？** 训练用精确版，推理用近似版（速度优先，精度差 < 1e-3 可接受）。你的项目两个都实现，对比精度。

---

## 算子 4：Scaled Dot-Product Attention

### 数学
```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

### 为什么除以 √d_k？
`Q·K^T` 的方差随 d_k 增大而增大（点积是 d_k 个乘加，方差累加）。方差大方差大意味着有些值很大，softmax 后会"尖峰化"——一个权重接近 1，其余接近 0，梯度消失。除以 √d_k 把方差拉回 ~1，softmax 输出分布健康。

**这个解释面试高频，务必理解。**

### 实现就是组合前面的算子
```cpp
void attention(const float* Q, const float* K, const float* V,
               float* out, int seq_len, int d_k) {
    // 步骤 1: scores = Q · K^T / √d_k   [seq_len × seq_len]
    std::vector<float> scores(seq_len * seq_len);
    float scale = 1.0f / sqrtf((float)d_k);
    // 复用 Phase 1 的 GEMM：Q(seq×d) × K^T(d×seq) = scores(seq×seq)
    // 注意 K^T：要么转置 K，要么 GEMM 时用 K 的转置布局
    matmul_transposed_b(Q, K, scores, seq_len, d_k, seq_len);  // 你实现的
    for (auto& s : scores) s *= scale;

    // 步骤 2: 对 scores 每行做 softmax
    std::vector<float> weights(seq_len * seq_len);
    for (int i = 0; i < seq_len; ++i) {
        softmax_row(&scores[i * seq_len], &weights[i * seq_len], seq_len);
    }

    // 步骤 3: out = weights · V   [seq_len × d_k]
    matmul(weights, V, out, seq_len, seq_len, d_k);
}
```

---

## Phase 2 的工程要求

### 1. 每个算子都要单元测试，对齐 PyTorch
```cpp
TEST(SoftmaxTest, MatchesPyTorch) {
    // 用 Python 生成 PyTorch 参考结果，C++ 读入对比
    // 容差 1e-5（Phase 2 算子精度要求高，比 GEMM 严）
}
```

### 2. 模块化：每个算子独立 .h/.cpp
```
phase2_operators/src/
├── softmax.h / softmax.cpp
├── layernorm.h / layernorm.cpp
├── gelu.h / gelu.cpp
├── attention.h / attention.cpp
└── matmul.h / matmul.cpp   ← 复用 Phase 1 的 GEMM
```

### 3. Python 参考生成脚本
```python
# scripts/gen_reference.py
import torch
import numpy as np

# 为每个算子生成参考答案
x = torch.randn(128)  # 随机输入
# Softmax
ref = torch.softmax(x, dim=0).numpy()
x.numpy().tofile("softmax_in.bin")
ref.tofile("softmax_ref.bin")
# LayerNorm, GELU, Attention 同理
```

---

## Phase 2 完成标志

- [ ] 5 个算子（Softmax/LayerNorm/GELU/MatMul/Attention）全部实现
- [ ] 每个算子有单元测试，对齐 PyTorch 到 1e-5
- [ ] 能用这些算子手动拼出一个完整的 Attention block 并跑通
- [ ] 代码模块化，能被 Phase 3 的计算图引擎直接调用

**Phase 2 比较直接，重点是"严谨"——每个算子都验证对，因为后面所有阶段都依赖它们。一个错的 Softmax 会让 Phase 4/5 的所有性能数据失去意义。**

---

完成后打开 `docs/04-phase3-graph-engine.md`，我们设计计算图引擎——那才是"框架"的灵魂。
