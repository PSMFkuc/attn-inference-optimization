#include "gemm_naive.h"
#include "gemm_tiled.h"
#include "gemm_simd.h"
#include "gemm_simd_unroll.h"
#include "gemm_parallel.h"
#include "timer.h"
#include <cstdio>
#include <random>
#include <algorithm>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#endif

// ===========================================================================
// Performance Benchmark (Production-grade)
// ===========================================================================
// Measurement principles (industrial standard):
//   1. Core pinning: prevent OS thread migration causing cache warm/cold alternation
//   2. Warmup: skip first run with cold start
//   3. Multiple trials, take minimum: not average! minimum is cleanest
//   4. Multiple sizes: small matrix = cache-in performance, large = cache miss impact
//   5. GFLOPS + theoretical peak efficiency: know "how much CPU capability is used"
// ===========================================================================

// --- Compute GFLOPS ---
// Why 2*M*N*K? Each C[i][j] does K multiplies + K adds = 2K FLOPs.
static double compute_gflops(int M, int N, int K, double ms) {
    double flops = 2.0 * M * N * K;
    return flops / (ms / 1000.0) / 1e9;
}

// --- CPU theoretical peak GFLOPS (single core) ---
// AVX2 = 256 bits = 8 floats, dual FMA = 16 ops/cycle
// Formula: frequency(GHz) * 16 ops/cycle
// Note: i5-12450H is hybrid, P-core 4.4GHz, E-core 3.3GHz
//       We pin to P-core, so use P-core frequency
static double theoretical_peak_gflops() {
    // i5-12450H P-core max turbo: 4.4 GHz
    // AVX2 FMA: 16 single-precision ops/cycle
    return 4.4 * 16.0;  // = 70.4 GFLOPS (single-core theoretical peak)
}

// --- CPU info detection (Windows) ---
static void print_cpu_info() {
#ifdef _WIN32
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
        "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
        0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        char cpuName[256] = {0};
        DWORD size = sizeof(cpuName);
        RegQueryValueExA(hKey, "ProcessorNameString", nullptr, nullptr,
                         (LPBYTE)cpuName, &size);
        RegCloseKey(hKey);
        printf("[CPU] %s\n", cpuName);
    }

    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    printf("[CPU] Cores: %lu, P-core AVX2 peak (single core): %.1f GFLOPS\n",
           sysInfo.dwNumberOfProcessors, theoretical_peak_gflops());
    printf("[CPU] Note: i5-12450H = 4P-cores (4.4GHz) + 4E-cores (3.3GHz), "
           "benchmark pinned to P-core\n");
#else
    printf("[CPU] Non-Windows platform, theoretical peak based on AVX2: ~70 GFLOPS\n");
#endif
}

// --- Pin to specified logical core ---
// i5-12450H logical core layout (typically):
//   0-7  = P-core hyperthreads (4 cores x 2 threads), pick Core 0 = mask 0x1
//   8-11 = E-cores (4 cores, no hyperthreading)
// Pin to Core 0 (P-core) to avoid scheduling to E-core, ensuring stable performance
static void pin_to_pcore() {
#ifdef _WIN32
    DWORD_PTR mask = 0x1;  // Use core 0 only
    if (SetProcessAffinityMask(GetCurrentProcess(), mask)) {
        printf("[PIN] Process pinned to logical core 0 (P-core)\n");
    } else {
        printf("[PIN] WARNING: Failed to pin to core (error %lu)\n", GetLastError());
    }
#else
    printf("[PIN] Non-Windows: pinning not implemented, results may vary\n");
#endif
}

// --- Benchmark function: run multiple times, take minimum ---
// auto_trials: automatically adjust trial count based on matrix size.
//   Small matrices (< 256) run more trials to reduce timing noise.
//   Large matrices run fewer trials because each run is slow.
static double bench_once(int M, int N, int K,
                         const std::vector<float>& A,
                         const std::vector<float>& B,
                         std::vector<float>& C,
                         void (*func)(int, int, int,
                                      const std::vector<float>&,
                                      const std::vector<float>&,
                                      std::vector<float>&),
                         int trials = 0) {
    // Auto-detect trial count based on size
    if (trials <= 0) {
        int sz = std::max({M, N, K});
        if (sz <= 64)       trials = 50;
        else if (sz <= 128) trials = 30;
        else if (sz <= 256) trials = 20;
        else                trials = 10;
    }

    // Warmup: first run data not in cache, time is longer
    func(M, N, K, A, B, C);

    double best = 1e9;
    for (int t = 0; t < trials; ++t) {
        auto start = now();
        func(M, N, K, A, B, C);
        auto end = now();
        double ms = elapsed_ms(start, end);
        best = std::min(best, ms);
    }
    return best;
}

// --- Tiled version has different signature (extra block size params), separate benchmark ---
static double bench_tiled(int M, int N, int K,
                          const std::vector<float>& A,
                          const std::vector<float>& B,
                          std::vector<float>& C,
                          int BM, int BN, int BK,
                          int trials = 0) {
    // Auto-detect trial count
    if (trials <= 0) {
        int sz = std::max({M, N, K});
        if (sz <= 64)       trials = 50;
        else if (sz <= 128) trials = 30;
        else if (sz <= 256) trials = 20;
        else                trials = 10;
    }

    // Warmup
    gemm_tiled(M, N, K, A, B, C, BM, BN, BK);

    double best = 1e9;
    for (int t = 0; t < trials; ++t) {
        auto start = now();
        gemm_tiled(M, N, K, A, B, C, BM, BN, BK);
        auto end = now();
        best = std::min(best, elapsed_ms(start, end));
    }
    return best;
}

int main() {
    // --- Environment setup ---
    // Note: Do NOT pin process to single core when testing multi-threaded code.
    // pin_to_pcore() restricts the entire process to core 0, which forces
    // all OpenMP threads to share one core → worse than single-threaded.
    // For multi-threaded benchmarks, we rely on OpenMP's thread affinity.
    // pin_to_pcore();  // DISABLED for parallel benchmarks
    print_cpu_info();        // Print CPU info
    printf("\n");

    double peak = theoretical_peak_gflops();  // Single-core theoretical peak

    // Fixed random seed for reproducibility
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Benchmark sizes from small to large, observe cache miss effects
    int sizes[] = {64, 128, 256, 512, 1024};

    // Print table header
    printf("%-8s | %-8s %-8s | %-8s %-8s | %-8s %-8s | %-8s %-8s | %-8s %-8s | %-8s %-8s | %-10s\n",
           "size", "ijk", "GFLOPS", "ikj", "GFLOPS",
           "tiled", "GFLOPS", "par-ikj", "GFLOPS", "par-tiled", "GFLOPS",
           "simd", "GFLOPS", "best/peak%");
    printf("%.140s\n",
           "--------------------------------------------------"
           "--------------------------------------------------"
           "--------------------------------------------------"
           "--------------------");

    for (int sz : sizes) {
        int M = sz, N = sz, K = sz;
        std::vector<float> A(M * K), B(K * N), C(M * N);
        for (auto& x : A) x = dist(rng);
        for (auto& x : B) x = dist(rng);

        double ms_ijk = bench_once(M, N, K, A, B, C, gemm_naive_ijk);
        double ms_ikj = bench_once(M, N, K, A, B, C, gemm_naive_ikj);
        double ms_tiled = bench_tiled(M, N, K, A, B, C, 128, 128, 128);
        double ms_par_ikj = bench_once(M, N, K, A, B, C, gemm_parallel_ikj, 5);
        double ms_par_tiled = bench_once(M, N, K, A, B, C, gemm_parallel_tiled, 5);
        double ms_simd = bench_once(M, N, K, A, B, C, gemm_simd);

        double gflops_ijk = compute_gflops(sz, sz, sz, ms_ijk);
        double gflops_ikj = compute_gflops(sz, sz, sz, ms_ikj);
        double gflops_tiled = compute_gflops(sz, sz, sz, ms_tiled);
        double gflops_par_ikj = compute_gflops(sz, sz, sz, ms_par_ikj);
        double gflops_par_tiled = compute_gflops(sz, sz, sz, ms_par_tiled);
        double gflops_simd = compute_gflops(sz, sz, sz, ms_simd);

        double best_gflops = std::max({gflops_ijk, gflops_ikj, gflops_tiled, gflops_par_ikj, gflops_par_tiled, gflops_simd});
        double efficiency = best_gflops / peak * 100.0;

        printf("%-8d | %-8.2f %-8.2f | %-8.2f %-8.2f | %-8.2f %-8.2f | %-8.2f %-8.2f | %-8.2f %-8.2f | %-8.2f %-8.2f | %-9.1f%%\n",
               sz,
               ms_ijk, gflops_ijk,
               ms_ikj, gflops_ikj,
               ms_tiled, gflops_tiled,
               ms_par_ikj, gflops_par_ikj,
               ms_par_tiled, gflops_par_tiled,
               ms_simd, gflops_simd,
               efficiency);
    }

    // --- Summary analysis ---
    printf("\n");
    printf("==========================================================================\n");
    printf("  Performance Analysis Report\n");
    printf("==========================================================================\n");
    printf("  CPU single-core theoretical peak (AVX2 FMA, P-core 4.4GHz): %.1f GFLOPS\n", peak);
    printf("  Current best implementation efficiency: see last column of table above\n\n");
    printf("  Efficiency interpretation:\n");
    printf("    < 10%%  : Severe memory bottleneck (high cache miss), optimize memory access first\n");
    printf("    10-30%% : Moderate, tiling helps but more room to improve\n");
    printf("    30-60%% : Good, auto-vectorization may be partially active\n");
    printf("    > 60%%  : Excellent, near hand-tuned level\n");
    printf("==========================================================================\n");

    // =========================================================================
    // Block Size Tuning Scan
    // =========================================================================
    // Test 7-8 representative block sizes to find optimal tiling parameter.
    // Selection rationale:
    //   L1 D-Cache = 48KB. Working set = (BM*BK + BK*BN + BM*BN) * 4 bytes.
    //   Sizes chosen to span L1-safe (< 48KB), L1-edge, and L2-range.
    //   All BN values are multiples of 8 (SIMD-friendly).
    // =========================================================================

    struct BlockConfig {
        int BM, BN, BK;
        const char* label;
    };

    BlockConfig block_configs[] = {
        { 16, 16, 16,  "16x16x16  L1-safe (3KB)"},
        { 32, 32, 32,  "32x32x32  L1-comfort (12KB)"},
        { 48, 48, 48,  "48x48x48  L1-edge (28KB)"},
        { 64, 64, 64,  "64x64x64  L1-critical (48KB)"},
        { 96, 96, 96,  "96x96x96  L2 (96KB)"},
        {128,128,128,  "128x128x128 L2 (192KB)"},
        { 64, 64,256,  "64x64x256  asym:long-K"},
        {128, 16,256,  "128x16x256 asym:register-favored"},
    };

    int tune_sizes[] = {256, 512, 1024};  // Only larger matrices benefit from tiling

    printf("\n");
    printf("==========================================================================\n");
    printf("  Block Size Tuning Scan (GFLOPS)\n");
    printf("==========================================================================\n");

    // Print header
    printf("%-22s", "block config");
    for (int sz : tune_sizes) printf(" | %-6d", sz);
    printf("\n");
    printf("%.80s\n", "----------------------------------------"
           "----------------------------------------");

    for (auto& cfg : block_configs) {
        printf("%-22s", cfg.label);

        for (int sz : tune_sizes) {
            std::vector<float> A(sz * sz), B(sz * sz), C(sz * sz);
            for (auto& x : A) x = dist(rng);
            for (auto& x : B) x = dist(rng);

            double ms = bench_tiled(sz, sz, sz, A, B, C, cfg.BM, cfg.BN, cfg.BK, 0);
            double gflops = compute_gflops(sz, sz, sz, ms);
            printf(" | %-6.2f", gflops);
        }
        printf("\n");
    }

    printf("\n  Interpreting the scan:\n");
    printf("  - If GFLOPS peaks at 32-48: L1-sized blocks are optimal.\n");
    printf("  - If GFLOPS peaks at 64-96:  L2 is enough, L1 too tight.\n");
    printf("  - Asymmetric configs (long-K) may win if K-dimension is the bottleneck.\n");
    printf("  - Pick the config with the best GFLOPS at size=1024 for final report.\n");
    printf("==========================================================================\n");

    return 0;
}
