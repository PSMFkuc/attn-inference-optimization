#include <gtest/gtest.h>
#include "operators.h"
#include <cmath>
#include <vector>
#include <numeric>
#include <fstream>
#include <string>

// ============================================================
// 辅助函数：读二进制文件
// ============================================================
static std::vector<float> read_bin(const std::string& path, int count) {
    std::vector<float> data(count);
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        ADD_FAILURE() << "Cannot open: " << path;
        return data;
    }
    f.read(reinterpret_cast<char*>(data.data()), count * sizeof(float));
    return data;
}

static float max_abs_error(const std::vector<float>& a,
                           const std::vector<float>& b) {
    float max_err = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        max_err = std::max(max_err, std::abs(a[i] - b[i]));
    }
    return max_err;
}

// ============================================================
// Softmax 测试
// ============================================================
TEST(SoftmaxTest, BasicOrdering) {
    // 大的输入对应大的输出
    float x[] = {2.0f, 1.0f, 0.5f};
    softmax(x, 3);
    EXPECT_GT(x[0], x[1]);
    EXPECT_GT(x[1], x[2]);
}

TEST(SoftmaxTest, SumToOne) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    softmax(x, 5);
    float s = std::accumulate(x, x + 5, 0.0f);
    EXPECT_NEAR(s, 1.0f, 1e-5f);
}

TEST(SoftmaxTest, NumericalStability) {
    // 全是很大的数，不应产生 NaN 或 inf
    float x[] = {1000.0f, 1000.0f, 1000.0f};
    softmax(x, 3);
    for (int i = 0; i < 3; i++) {
        EXPECT_FALSE(std::isnan(x[i]));
        EXPECT_FALSE(std::isinf(x[i]));
        EXPECT_NEAR(x[i], 1.0f / 3.0f, 1e-5f);
    }
}

TEST(SoftmaxTest, AllNegative) {
    float x[] = {-5.0f, -3.0f, -1.0f};
    softmax(x, 3);
    float s = std::accumulate(x, x + 3, 0.0f);
    EXPECT_NEAR(s, 1.0f, 1e-5f);
    EXPECT_GT(x[2], x[1]);  // -1 最大
    EXPECT_GT(x[1], x[0]);  // -3 其次
}

// ============================================================
// LayerNorm 测试
// ============================================================
TEST(LayerNormTest, IdentityGammaBeta) {
    float x[] = {1.0f, 3.0f, 2.0f, 6.0f};
    float gamma[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float out[4];
    layernorm(x, 4, 1e-5f, gamma, beta, out);

    // 输出均值应接近 0
    float mean = std::accumulate(out, out + 4, 0.0f) / 4;
    EXPECT_NEAR(mean, 0.0f, 1e-5f);

    // 输出标准差应接近 1
    float var = 0.0f;
    for (int i = 0; i < 4; i++) var += out[i] * out[i];
    var /= 4;
    EXPECT_NEAR(var, 1.0f, 1e-4f);
}

TEST(LayerNormTest, GammaBetaApplied) {
    float x[] = {2.0f, 4.0f, 6.0f, 8.0f};
    float gamma[] = {2.0f, 1.0f, 1.0f, 1.0f};
    float beta[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float out[4];
    layernorm(x, 4, 1e-5f, gamma, beta, out);
    // gamma=2, beta=1 对第一个元素：应该被放大和上移
    EXPECT_GT(out[0], out[1]);  // 放大了第一维
}

// ============================================================
// GELU 测试
// ============================================================
TEST(GeluTest, AtZero) {
    EXPECT_NEAR(gelu_exact(0.0f), 0.0f, 1e-5f);
    EXPECT_NEAR(gelu_approx(0.0f), 0.0f, 1e-3f);
}

TEST(GeluTest, PositiveLarge) {
    // x 很大时 GELU 接近恒等（和 ReLU 行为类似）
    float g = gelu_exact(10.0f);
    EXPECT_NEAR(g, 10.0f, 0.01f);
}

TEST(GeluTest, NegativeLarge) {
    // x 很负时 GELU 接近 0
    float g = gelu_exact(-10.0f);
    EXPECT_NEAR(g, 0.0f, 0.01f);
}

TEST(GeluTest, ExactVsApprox) {
    // 近似版和精确版差异应小于 0.002
    for (float x = -5.0f; x <= 5.0f; x += 0.1f) {
        float diff = std::abs(gelu_exact(x) - gelu_approx(x));
        EXPECT_LT(diff, 0.002f) << "at x=" << x;
    }
}

TEST(GeluTest, Monotonic) {
    // GELU 应该单调递增
    float prev = gelu_exact(-5.0f);
    for (float x = -4.9f; x <= 5.0f; x += 0.1f) {
        float curr = gelu_exact(x);
        EXPECT_GE(curr, prev) << "at x=" << x;
        prev = curr;
    }
}

// ============================================================
// Attention 测试
// ============================================================
TEST(AttentionTest, OutputShape) {
    int seq_len = 3, d_k = 4, d_v = 4;
    std::vector<float> Q(seq_len * d_k, 1.0f);
    std::vector<float> K(seq_len * d_k, 1.0f);
    std::vector<float> V(seq_len * d_v, 1.0f);
    std::vector<float> out;

    scaled_dot_product_attention(Q, K, V, seq_len, d_k, d_v, out);

    EXPECT_EQ(out.size(), seq_len * d_v);
    // V 全是 1，权重每行和为 1，所以每行输出也都应该是 1
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_v; j++) {
            EXPECT_NEAR(out[i * d_v + j], 1.0f, 1e-4f);
        }
    }
}

TEST(AttentionTest, WeightSumToOne) {
    // 验证 attention 权重每行和为 1
    // 通过全 0 Q 和全 1 K：QK^T=0，softmax 后均匀分布
    int seq_len = 4, d_k = 8, d_v = 4;
    std::vector<float> Q(seq_len * d_k, 0.0f);
    std::vector<float> K(seq_len * d_k, 1.0f);
    std::vector<float> V(seq_len * d_v, 1.0f);
    std::vector<float> out;

    scaled_dot_product_attention(Q, K, V, seq_len, d_k, d_v, out);

    // QK^T 全为 0 → softmax 每行均匀 → 每行都 = 1/seq_len
    // V 全 1 → out 每行都是 1
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_v; j++) {
            EXPECT_NEAR(out[i * d_v + j], 1.0f, 1e-4f);
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
