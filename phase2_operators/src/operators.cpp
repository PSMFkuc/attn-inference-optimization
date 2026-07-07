#include "operators.h"
#include <cmath>
#include <cfloat>
#include <algorithm>

// ============================================================
// Softmax 实现
// ============================================================

// 单行 softmax，原地修改
void softmax(float* x, int N) {
    // 第一步：找最大值，用于数值稳定
    // 为什么是必须的？exp(1000) 直接溢出成 inf，减去 max 后最大是 exp(0)=1
    float max_val = -FLT_MAX;
    for (int i = 0; i < N; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    // 第二步：exp(x_i - max) 同时累加 sum
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }

    // 第三步：除以 sum（用乘法代替除法）
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < N; i++) {
        x[i] *= inv_sum;
    }
}

// 逐行 softmax
void softmax_2d(float* matrix, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        softmax(&matrix[r * cols], cols);
    }
}

// ============================================================
// LayerNorm 实现
// ============================================================
void layernorm(const float* x, int N, float eps,
               const float* gamma, const float* beta,
               float* out) {
    // 第一遍：算均值 μ = Σx / N
    float mean = 0.0f;
    for (int i = 0; i < N; i++) mean += x[i];
    mean /= N;

    // 第二遍：算方差 σ² = Σ(x-μ)² / N
    // 必须等均值算完才能做这步
    float var = 0.0f;
    for (int i = 0; i < N; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= N;

    // 预计算 1/√(σ²+ε)，循环里用乘法而不是除法
    float inv_std = 1.0f / std::sqrt(var + eps);

    // 第三遍：归一化 + 仿射变换 y = γ·(x-μ)/σ + β
    for (int i = 0; i < N; i++) {
        out[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i];
    }
}

// ============================================================
// GELU 实现
// ============================================================

float gelu_exact(float x) {
    // GELU(x) = 0.5·x·(1 + erf(x/√2))
    return 0.5f * x * (1.0f + std::erf(x / 1.414213562f));
}

float gelu_approx(float x) {
    // tanh 近似：GELU(x) ≈ 0.5·x·[1 + tanh(√(2/π)·(x + 0.044715·x³))]
    const float c1 = 0.7978845608f;  // √(2/π)
    const float c2 = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(c1 * (x + c2 * x3)));
}

void gelu_vector_exact(const float* in, float* out, int N) {
    for (int i = 0; i < N; i++) out[i] = gelu_exact(in[i]);
}

void gelu_vector_approx(const float* in, float* out, int N) {
    for (int i = 0; i < N; i++) out[i] = gelu_approx(in[i]);
}

// ============================================================
// Scaled Dot-Product Attention
// ============================================================

// 外部声明：Phase 1 的 GEMM（链接时绑定）
extern void gemm_ikj(int M, int N, int K,
                     const std::vector<float>& A,
                     const std::vector<float>& B,
                     std::vector<float>& C);

void scaled_dot_product_attention(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    int seq_len, int d_k, int d_v,
    std::vector<float>& output)
{
    // === 第 1 步：scores = Q × K^T / √d_k ===
    // 手动转置 K：从 [seq_len × d_k] 变 [d_k × seq_len]
    // 行主序下：K[i][j] = K[i*d_k + j]
    //          K^T[j][i] = K_T[j*seq_len + i] = K[i*d_k + j]
    std::vector<float> K_T(d_k * seq_len);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_k; j++) {
            K_T[j * seq_len + i] = K[i * d_k + j];
        }
    }

    // scores = Q(seq×d_k) × K^T(d_k×seq)
    std::vector<float> scores(seq_len * seq_len);
    gemm_ikj(seq_len, seq_len, d_k, Q, K_T, scores);

    // 除以 √d_k：缩放因子，防止点积方差过大导致 softmax 饱和
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
    for (int i = 0; i < seq_len * seq_len; i++) {
        scores[i] *= scale;
    }

    // === 第 2 步：逐行 softmax → 注意力权重 ===
    std::vector<float> weights(scores.begin(), scores.end());
    softmax_2d(weights.data(), seq_len, seq_len);

    // === 第 3 步：output = weights × V ===
    // weights: seq×seq, V: seq×d_v → output: seq×d_v
    output.resize(seq_len * d_v);
    gemm_ikj(seq_len, d_v, seq_len, weights, V, output);
}
