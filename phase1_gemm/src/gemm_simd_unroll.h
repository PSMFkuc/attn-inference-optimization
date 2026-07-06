#ifndef GEMM_SIMD_UNROLL_H
#define GEMM_SIMD_UNROLL_H

#include <vector>

// ===========================================================================
// SIMD GEMM with 4x loop unrolling
// ===========================================================================
// Builds on gemm_simd and unrolls the inner j-loop by 4x:
//   Each iteration processes 4 * 8 = 32 floats before the next loop check.
//
// Why unroll?
//   perf annotate showed that add+cmp+jne (loop control) consumed 33% of
//   inner loop time. Unrolling 4x reduces this overhead from 33% to ~8%.
//
// Register pressure:
//   AVX2 has 16 YMM registers. 4x unrolling uses:
//     4 for B loads, 4 for C loads, 1 for a_ik = 9 registers → safe.
// ===========================================================================

void gemm_simd_unroll(int M, int N, int K,
                      const std::vector<float>& A,
                      const std::vector<float>& B,
                      std::vector<float>& C);

#endif // GEMM_SIMD_UNROLL_H
