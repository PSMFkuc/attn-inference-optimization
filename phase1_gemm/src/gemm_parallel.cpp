#include "gemm_parallel.h"
#include <cstring>
#include <omp.h>
#ifdef _WIN32
#include <windows.h>
#endif

// ===========================================================================
// Thread affinity setup: restrict OpenMP to P-cores only
// ===========================================================================
// i5-12450H logical core layout:
//   Cores 0-7  = P-cores (4 physical × 2 hyperthreads)
//   Cores 8-11 = E-cores (4 physical, no hyperthreading)
//
// Strategy: Create an OpenMP place partition that only includes P-cores.
//   - OMP_PLACES = "{0:4}:2:1" means 4 places, starting at core 0, stride 2
//     (skip hyperthreads, use only physical P-cores)
//   - OMP_NUM_THREADS = 4 (match 4 P-cores)
//
// This prevents OpenMP from scheduling threads on E-cores, avoiding the
// performance penalty of slower E-cores dragging down parallel sections.
// ===========================================================================

static void setup_omp_pcore_affinity() {
#ifdef _WIN32
    // Restrict OpenMP to P-cores only (logical cores 0,2,4,6 = physical P-cores)
    // Skip E-cores (8-11) and hyperthreads (1,3,5,7)
    _putenv("OMP_PLACES=cores");
    _putenv("OMP_PROC_BIND=close");
    _putenv("OMP_NUM_THREADS=4");

    // Use Windows API to set affinity mask for the current process
    // to P-cores only (cores 0,2,4,6 → mask 0x55 = 01010101)
    HANDLE hProcess = GetCurrentProcess();
    DWORD_PTR processMask, systemMask;
    GetProcessAffinityMask(hProcess, &processMask, &systemMask);

    // Try to restrict to P-cores (mask 0x55 = cores 0,2,4,6)
    DWORD_PTR pcoreMask = 0x55;
    SetProcessAffinityMask(hProcess, pcoreMask & systemMask);
#endif
}

// Parallel ikj GEMM
// ===========================================================================
// Parallelization strategy:
//   Split the outer i-loop (rows of C) across threads.
//   Each thread gets a contiguous chunk of rows.
//   No write conflicts because each row belongs to exactly one thread.
//
// OpenMP scheduling:
//   schedule(static): pre-assign equal chunks to threads. Best when
//   all iterations take same time (our case: all rows equal size).
//
// Thread count:
//   Default = all cores. We let OpenMP decide (usually = physical cores).
//   Can override: export OMP_NUM_THREADS=4
// ===========================================================================

void gemm_parallel_ikj(int M, int N, int K,
                       const std::vector<float>& A,
                       const std::vector<float>& B,
                       std::vector<float>& C) {

    setup_omp_pcore_affinity();

    C.resize(M * N);
    std::memset(C.data(), 0, M * N * sizeof(float));

    // Parallelize the outer i-loop. Each thread processes its own chunk of rows.
    // No data dependency between rows → safe to parallelize.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a_ik = A[i * K + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

// ===========================================================================
// Parallel tiled GEMM (block=128x128x128, best from block size scan)
// ===========================================================================
// Two-level parallelism:
//   1. Outer ii-loop (row blocks of C) → parallelized with OpenMP
//   2. Inner ikj loops → sequential per thread (cache-friendly within block)
//
// Why parallelize the block-level ii loop, not the fine-grained i loop?
//   - Larger work units per thread → less synchronization overhead
//   - Each thread works on a contiguous set of rows → better cache locality
//   - Block size 128 means each thread gets (M/num_threads) rows
//
// Memory bandwidth bottleneck:
//   With 4 threads reading B simultaneously, memory bandwidth may saturate.
//   B is read-only (shared), so no false sharing on B. But total bandwidth
//   demand is 4x single-thread → may become the new bottleneck.
// ===========================================================================

void gemm_parallel_tiled(int M, int N, int K,
                         const std::vector<float>& A,
                         const std::vector<float>& B,
                         std::vector<float>& C) {

    setup_omp_pcore_affinity();

    C.resize(M * N);
    std::memset(C.data(), 0, M * N * sizeof(float));

    const int BM = 128, BN = 128, BK = 128;

    // Parallelize the outer block-level ii loop.
    // Each thread gets a set of row-blocks to work on.
    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < M; ii += BM) {
        int i_end = (ii + BM < M) ? (ii + BM) : M;

        for (int kk = 0; kk < K; kk += BK) {
            int k_end = (kk + BK < K) ? (kk + BK) : K;

            for (int jj = 0; jj < N; jj += BN) {
                int j_end = (jj + BN < N) ? (jj + BN) : N;

                // Inner ikj triple loop (sequential within each thread)
                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        float a_ik = A[i * K + k];
                        for (int j = jj; j < j_end; ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}
