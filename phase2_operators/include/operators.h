#pragma once
#include <vector>
#include <cstddef>

// ============================================================
// Phase 2 算子库：接口声明
// 每个算子的"合同"——函数签名、参数含义、调用约定
// ============================================================

// ---- Softmax ----
// 对一维数组做 softmax，原地修改
// x: 输入/输出，N: 数组长度
void softmax(float* x, int N);

// 对矩阵逐行做 softmax（Attention 里用这个）
// matrix: 行主序矩阵 [rows × cols]
void softmax_2d(float* matrix, int rows, int cols);

// ---- LayerNorm ----
// x: 输入，N: 特征维度（d_model）
// eps: 防止除零的小常数（默认 1e-5）
// gamma, beta: 可学习参数，维度都是 N
// out: 输出
void layernorm(const float* x, int N, float eps,
               const float* gamma, const float* beta,
               float* out);

// ---- GELU ----
// 精确版（用 erf），对标论文实现
float gelu_exact(float x);

// 近似版（用 tanh），推理场景更快
float gelu_approx(float x);

// 向量版本
void gelu_vector_exact(const float* in, float* out, int N);
void gelu_vector_approx(const float* in, float* out, int N);

// ---- Scaled Dot-Product Attention ----
// Q: [seq_len × d_k], K: [seq_len × d_k], V: [seq_len × d_v]
// output: [seq_len × d_v]
void scaled_dot_product_attention(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    int seq_len, int d_k, int d_v,
    std::vector<float>& output);
