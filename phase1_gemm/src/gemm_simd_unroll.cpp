#include "gemm_simd_unroll.h"
#include <immintrin.h>
#include <cstring>

// ===========================================================================
// SIMD GEMM with 4x Loop Unrolling
// ===========================================================================
//
// Before (no unroll):
//   for j in 0..N step 8:
//       load B[j..j+7]
//       load C[j..j+7]
//       FMA: C = a_ik * B + C
//       store C[j..j+7]
//       add rcx, 32; cmp rcx, rdi; jne loop   ← 33% of time!
//
// After (4x unroll):
//   for j in 0..N step 32:
//       load B[j..j+7],  FMA, store C[j..j+7]     // unroll 1
//       load B[j+8..15], FMA, store C[j+8..15]    // unroll 2
//       load B[j+16..23],FMA, store C[j+16..23]   // unroll 3
//       load B[j+24..31],FMA, store C[j+24..31]   // unroll 4
//       add rcx, 128; cmp rcx, rdi; jne loop      ← ~8% of time
//
// Key insight from perf annotate:
//   - add+cmp+jne was 33% of inner loop
//   - With 4x unroll, we do 4x more work per check → ~8% overhead
//   - Expected speedup: 1 / (1 - 0.25) ≈ 1.33x (25-33% faster)
//
// Register usage (16 YMM registers total):
//   a_ik:  1 register (broadcast)
//   B[0-3]: 4 registers (one per unrolled iteration)
//   C[0-3]: 4 registers (one per unrolled iteration)
//   Total:  9 registers → safe, 7 registers left for compiler
// ===========================================================================

void gemm_simd_unroll(int M, int N, int K,
                      const std::vector<float>& A,
                      const std::vector<float>& B,
                      std::vector<float>& C) {

    C.resize(M * N);
    std::memset(C.data(), 0, M * N * sizeof(float));

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {

            // Broadcast a_ik to all 8 lanes
            __m256 a_ik = _mm256_set1_ps(A[i * K + k]);

            // --- 4x unrolled SIMD inner loop ---
            // Process 32 floats per iteration (4 * 8)
            int j = 0;
            for (; j + 31 < N; j += 32) {

                // Unroll 1: j+0..j+7
                __m256 b0 = _mm256_loadu_ps(&B[k * N + j]);
                __m256 c0 = _mm256_loadu_ps(&C[i * N + j]);
                c0 = _mm256_fmadd_ps(a_ik, b0, c0);
                _mm256_storeu_ps(&C[i * N + j], c0);

                // Unroll 2: j+8..j+15
                __m256 b1 = _mm256_loadu_ps(&B[k * N + j + 8]);
                __m256 c1 = _mm256_loadu_ps(&C[i * N + j + 8]);
                c1 = _mm256_fmadd_ps(a_ik, b1, c1);
                _mm256_storeu_ps(&C[i * N + j + 8], c1);

                // Unroll 3: j+16..j+23
                __m256 b2 = _mm256_loadu_ps(&B[k * N + j + 16]);
                __m256 c2 = _mm256_loadu_ps(&C[i * N + j + 16]);
                c2 = _mm256_fmadd_ps(a_ik, b2, c2);
                _mm256_storeu_ps(&C[i * N + j + 16], c2);

                // Unroll 4: j+24..j+31
                __m256 b3 = _mm256_loadu_ps(&B[k * N + j + 24]);
                __m256 c3 = _mm256_loadu_ps(&C[i * N + j + 24]);
                c3 = _mm256_fmadd_ps(a_ik, b3, c3);
                _mm256_storeu_ps(&C[i * N + j + 24], c3);
            }

            // --- 2x unrolled tail: handle j where 8 <= remaining < 32 ---
            for (; j + 7 < N; j += 8) {
                __m256 b = _mm256_loadu_ps(&B[k * N + j]);
                __m256 c = _mm256_loadu_ps(&C[i * N + j]);
                c = _mm256_fmadd_ps(a_ik, b, c);
                _mm256_storeu_ps(&C[i * N + j], c);
            }

            // --- Scalar tail: handle remaining columns (N % 8) ---
            for (; j < N; ++j) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}
