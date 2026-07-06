#include "gemm_simd.h"
#include <immintrin.h>  // AVX2 intrinsics
#include <cstring>      // memset

// ===========================================================================
// SIMD GEMM: AVX2 + FMA
// ===========================================================================
//
// Structure: same ikj loop order as gemm_naive_ikj, but inner j-loop is
// vectorized to process 8 floats at a time.
//
//     for i in 0..M:
//         for k in 0..K:
//             a_ik = A[i*K+k]           // scalar, broadcast to 8 lanes
//             for j in 0..N step 8:     // SIMD: process 8 floats at once
//                 C[j..j+7] += a_ik * B[j..j+7]   // FMA instruction
//
// Why FMA (_mm256_fmadd_ps)?
//   FMA = Fused Multiply-Add. One instruction does: d = a * b + c
//   Without FMA: 2 instructions (multiply then add), 2 cycles
//   With FMA:    1 instruction (fused),        1 cycle
//   Throughput: 2 FMA per cycle per core = 16 single-precision FLOPs/cycle
//
// Why unaligned load/store (_mm256_loadu_ps / _mm256_storeu_ps)?
//   Aligned (load_ps) requires 32-byte aligned addresses. std::vector<float>
//   may not be 32-byte aligned. Unaligned (loadu) works on any address.
//   On modern CPUs (Haswell+), unaligned has zero penalty when data crosses
//   no cache line boundary. We use unaligned for safety.
// ===========================================================================

void gemm_simd(int M, int N, int K,
               const std::vector<float>& A,
               const std::vector<float>& B,
               std::vector<float>& C) {

    // Step 1: Initialize C to zero.
    // Use memset for speed (sets all bytes to 0).
    // Note: this only works for float 0.0f because IEEE 754 defines
    // 0.0f as all-zero bits.
    C.resize(M * N);
    std::memset(C.data(), 0, M * N * sizeof(float));

    // Step 2: Main computation with AVX2 SIMD.
    // Loop order: i -> k -> j (same as ikj, cache-friendly for B and C).

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {

            // Load scalar a_ik and broadcast to all 8 lanes of a YMM register.
            // _mm256_set1_ps(x): [x, x, x, x, x, x, x, x]
            __m256 a_ik = _mm256_set1_ps(A[i * K + k]);

            // --- SIMD inner loop: process 8 floats per iteration ---
            // Step j by 8 because AVX2 handles 8 floats at once.
            int j = 0;
            for (; j + 7 < N; j += 8) {

                // Load 8 consecutive floats from B[k*N + j..j+7]
                __m256 b = _mm256_loadu_ps(&B[k * N + j]);

                // Load 8 consecutive floats from C[i*N + j..j+7]
                __m256 c = _mm256_loadu_ps(&C[i * N + j]);

                // FMA: c = a_ik * b + c  (fused multiply-add)
                // This single instruction replaces both multiply and add.
                c = _mm256_fmadd_ps(a_ik, b, c);

                // Store result back to C[i*N + j..j+7]
                _mm256_storeu_ps(&C[i * N + j], c);
            }

            // --- Remainder loop: handle leftover columns (N % 8) ---
            // If N is not a multiple of 8, the last few columns need
            // scalar processing. This is the "tail" of the loop.
            for (; j < N; ++j) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

// ===========================================================================
// Thinking questions (answer after running benchmarks):
//
// 1. Why is the speedup NOT exactly 8x despite 8-wide SIMD?
//    Hint: memory bandwidth, cache, loop overhead, remainder handling.
//
// 2. Why is the load from C (not just B) also needed in the inner loop?
//    Hint: C accumulates across k, we can't just overwrite it.
//
// 3. What happens if N is exactly a multiple of 8?
//    The remainder loop never executes — best case for SIMD.
//
// 4. Why is _mm256_fmadd_ps faster than _mm256_mul_ps + _mm256_add_ps?
//    FMA is a single micro-op. mul+add is two micro-ops. Half the work.
// ===========================================================================
