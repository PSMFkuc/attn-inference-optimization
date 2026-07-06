#ifndef GEMM_SIMD_H
#define GEMM_SIMD_H

#include <vector>

// ===========================================================================
// SIMD-optimized GEMM using AVX2 intrinsics
// ===========================================================================
// Principle:
//   AVX2 registers are 256 bits = 8 floats.
//   One instruction processes 8 floats simultaneously.
//   Combined with FMA (Fused Multiply-Add), each cycle does 16 FLOPs.
//
// This implementation builds on ikj loop order and replaces the inner
// j-loop with SIMD vectorized operations.
//
// Constraint: N must be a multiple of 8 for the main loop.
//   A tail (remainder) loop handles the last N%8 elements.
// ===========================================================================

void gemm_simd(int M, int N, int K,
               const std::vector<float>& A,
               const std::vector<float>& B,
               std::vector<float>& C);

#endif // GEMM_SIMD_H
