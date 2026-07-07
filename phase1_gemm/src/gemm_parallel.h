#ifndef GEMM_PARALLEL_H
#define GEMM_PARALLEL_H

#include <vector>

// ===========================================================================
// Multi-threaded GEMM using OpenMP
// ===========================================================================
//
// Principle:
//   The outer i-loop iterates over rows of C, each row is independent.
//   We parallelize the i-loop with #pragma omp parallel for.
//   Each thread computes a subset of C's rows.
//
//   for i in 0..M:          ← parallelize this
//       for k in 0..K:
//           for j in 0..N:
//               C[i*N+j] += A[i*K+k] * B[k*N+j]
//
// Why parallelize i and not k or j?
//   - i: each thread writes to different rows of C → no write conflict ✅
//   - k: all threads accumulate into same C elements → race condition ❌
//   - j: writes to adjacent C elements → false sharing ❌
//
// False sharing risk:
//   Adjacent rows share the same cache line (64 bytes = 16 floats).
//   If thread 0 writes C[0] and thread 1 writes C[1], they fight over
//   the same cache line → performance drops.
//   Mitigation: each row is N floats, much larger than a cache line,
//   so the risk is low for M x N matrices with N >= 64.
//
// Expected speedup:
//   i5-12450H has 4 P-cores. Theoretical 4x, actual ~3x (Amdahl + memory BW).
// ===========================================================================

// Parallel ikj (multi-threaded loop-reorder GEMM)
void gemm_parallel_ikj(int M, int N, int K,
                       const std::vector<float>& A,
                       const std::vector<float>& B,
                       std::vector<float>& C);

// Parallel tiled (multi-threaded + cache-blocked GEMM, block=128)
void gemm_parallel_tiled(int M, int N, int K,
                         const std::vector<float>& A,
                         const std::vector<float>& B,
                         std::vector<float>& C);

#endif // GEMM_PARALLEL_H
