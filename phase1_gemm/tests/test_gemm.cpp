#include <gtest/gtest.h>
#include "gemm_naive.h"
#include "gemm_tiled.h"
#include "gemm_simd.h"
#include "gemm_simd_unroll.h"
#include "gemm_parallel.h"
#include <vector>
#include <cmath>
#include <random>

// ===========================================================================
// 正确性测试：每个优化版本必须对齐朴素参考实现
// ===========================================================================
// 原则：先正确，再快速。一个跑得飞快但结果错误的 kernel 毫无价值。
// 每次优化前后都要跑一次正确性测试。

// 参考实现：最朴素的 ijk，作为 ground truth。
// 为什么单独写一个？因为我们要对比"被测函数"和"独立实现的参考"，
// 不能让被测函数自己和自己比（那永远"正确"）。
static void reference_gemm(int M, int N, int K,
                           const std::vector<float>& A,
                           const std::vector<float>& B,
                           std::vector<float>& C) {
    C.assign(M * N, 0.0f);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

// 测试 ikj 版本和参考实现一致
TEST(GemmTest, IKJ_MatchesReference) {
    const int M = 64, N = 64, K = 64;
    std::vector<float> A(M * K), B(K * N), C_test(M * N), C_ref(M * N);

    // 用有规律的数据填充（不要全 0 或全 1，测不出 bug）
    // 这里用 i%7 这种模式，覆盖正数、不同量级
    for (int i = 0; i < M * K; ++i) A[i] = (float)(i % 7) * 0.1f - 0.3f;
    for (int i = 0; i < K * N; ++i) B[i] = (float)(i % 5) * 0.1f - 0.2f;

    reference_gemm(M, N, K, A, B, C_ref);
    gemm_naive_ikj(M, N, K, A, B, C_test);

    // EXPECT_NEAR：浮点数容差比较。为什么不用 ==？
    //   float 精度有限（~7 位有效数字），运算顺序不同会有微小差异。
    //   == 永远失败。1e-4 是 float GEMM 的合理容差。
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_test[i], C_ref[i], 1e-4f)
            << "mismatch at index " << i
            << " (i=" << i / N << ", j=" << i % N << ")";
    }
}

// 测试分块版本和参考实现一致
TEST(GemmTest, Tiled_MatchesReference) {
    // 用非 block-size 整数倍的尺寸，测边界处理
    const int M = 70, N = 50, K = 90;  // 故意不是 32 的倍数
    std::vector<float> A(M * K), B(K * N), C_test(M * N), C_ref(M * N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    reference_gemm(M, N, K, A, B, C_ref);
    gemm_tiled(M, N, K, A, B, C_test, 32, 32, 32);

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_test[i], C_ref[i], 1e-3f)
            << "mismatch at index " << i;
    }
}

// 参数化测试：测多个尺寸，确保没有 size-specific bug
TEST(GemmTest, Tiled_MultipleSizes) {
    int sizes[] = {16, 32, 33, 64, 100, 128};
    for (int sz : sizes) {
        std::vector<float> A(sz*sz), B(sz*sz), C_test(sz*sz), C_ref(sz*sz);
        for (int i = 0; i < sz*sz; ++i) {
            A[i] = (float)(i % 11) * 0.05f;
            B[i] = (float)(i % 13) * 0.05f;
        }
        reference_gemm(sz, sz, sz, A, B, C_ref);
        gemm_tiled(sz, sz, sz, A, B, C_test);
        for (int i = 0; i < sz*sz; ++i) {
            EXPECT_NEAR(C_test[i], C_ref[i], 1e-3f)
                << "size=" << sz << " mismatch at " << i;
        }
    }
}

// Test SIMD version matches reference
TEST(GemmTest, SIMD_MatchesReference) {
    const int M = 64, N = 64, K = 64;  // N is multiple of 8, best case for SIMD
    std::vector<float> A(M * K), B(K * N), C_test(M * N), C_ref(M * N);

    for (int i = 0; i < M * K; ++i) A[i] = (float)(i % 7) * 0.1f - 0.3f;
    for (int i = 0; i < K * N; ++i) B[i] = (float)(i % 5) * 0.1f - 0.2f;

    reference_gemm(M, N, K, A, B, C_ref);
    gemm_simd(M, N, K, A, B, C_test);

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_test[i], C_ref[i], 1e-4f)
            << "SIMD mismatch at index " << i
            << " (i=" << i / N << ", j=" << i % N << ")";
    }
}

// Test SIMD with non-multiple-of-8 sizes (remainder loop test)
TEST(GemmTest, SIMD_NonMultipleOf8) {
    const int M = 70, N = 50, K = 90;  // 50 is NOT a multiple of 8, tests remainder
    std::vector<float> A(M * K), B(K * N), C_test(M * N), C_ref(M * N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    reference_gemm(M, N, K, A, B, C_ref);
    gemm_simd(M, N, K, A, B, C_test);

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_test[i], C_ref[i], 1e-3f)
            << "SIMD (non-multiple-of-8) mismatch at index " << i;
    }
}

// Test 4x-unrolled SIMD version matches reference
TEST(GemmTest, SIMD_Unroll_MatchesReference) {
    const int M = 64, N = 64, K = 64;  // N is multiple of 8 and 32
    std::vector<float> A(M * K), B(K * N), C_test(M * N), C_ref(M * N);

    for (int i = 0; i < M * K; ++i) A[i] = (float)(i % 7) * 0.1f - 0.3f;
    for (int i = 0; i < K * N; ++i) B[i] = (float)(i % 5) * 0.1f - 0.2f;

    reference_gemm(M, N, K, A, B, C_ref);
    gemm_simd_unroll(M, N, K, A, B, C_test);

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_test[i], C_ref[i], 1e-4f)
            << "SIMD-Unroll mismatch at index " << i;
    }
}

// Test unrolled SIMD with non-multiple-of-32 sizes (tests 2x + scalar tail)
TEST(GemmTest, SIMD_Unroll_NonMultipleOf32) {
    const int M = 70, N = 50, K = 90;  // 50 is NOT 32-aligned, tests 2x tail + scalar
    std::vector<float> A(M * K), B(K * N), C_test(M * N), C_ref(M * N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    reference_gemm(M, N, K, A, B, C_ref);
    gemm_simd_unroll(M, N, K, A, B, C_test);

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_test[i], C_ref[i], 1e-3f)
            << "SIMD-Unroll (non-32-aligned) mismatch at index " << i;
    }
}

// Test parallel ikj matches reference
TEST(GemmTest, Parallel_IKJ_MatchesReference) {
    const int M = 128, N = 128, K = 128;
    std::vector<float> A(M * K), B(K * N), C_test(M * N), C_ref(M * N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    reference_gemm(M, N, K, A, B, C_ref);
    gemm_parallel_ikj(M, N, K, A, B, C_test);

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_test[i], C_ref[i], 1e-3f)
            << "Parallel-IKJ mismatch at index " << i;
    }
}

// Test parallel tiled matches reference
TEST(GemmTest, Parallel_Tiled_MatchesReference) {
    const int M = 256, N = 256, K = 256;
    std::vector<float> A(M * K), B(K * N), C_test(M * N), C_ref(M * N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    reference_gemm(M, N, K, A, B, C_ref);
    gemm_parallel_tiled(M, N, K, A, B, C_test);

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_test[i], C_ref[i], 1e-3f)
            << "Parallel-Tiled mismatch at index " << i;
    }
}
