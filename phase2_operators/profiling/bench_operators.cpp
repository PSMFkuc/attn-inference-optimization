#include "operators.h"
#include "gemm_naive.h"
#include <cstdio>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>

// ===========================================================================
// Phase 2 Operator Benchmark
// ===========================================================================
// Measures throughput for each operator at representative sizes.

static double now_ms() {
    auto t = std::chrono::high_resolution_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(t).count();
}

template<typename Func>
static double bench(Func f, const char* label, int trials = 10) {
    f(); // warmup
    double best = 1e9;
    for (int t = 0; t < trials; t++) {
        double t0 = now_ms();
        f();
        best = std::min(best, now_ms() - t0);
    }
    printf("  %-30s %10.4f ms\n", label, best);
    return best;
}

int main() {
    printf("============================================================\n");
    printf("  Phase 2 Operator Benchmark\n");
    printf("============================================================\n\n");

    // ============================================================
    // Softmax benchmark
    // ============================================================
    printf("[Softmax]\n");
    int softmax_N = 4096;
    std::vector<float> sm_data(softmax_N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : sm_data) x = dist(rng);

    bench([&]() { softmax(sm_data.data(), softmax_N); },
          "softmax(N=4096)");

    // 2D softmax: 128 rows x 128 cols
    int sm_rows = 128, sm_cols = 128;
    std::vector<float> sm_2d(sm_rows * sm_cols);
    for (auto& x : sm_2d) x = dist(rng);
    bench([&]() { softmax_2d(sm_2d.data(), sm_rows, sm_cols); },
          "softmax_2d(128x128)");

    // ============================================================
    // LayerNorm benchmark
    // ============================================================
    printf("\n[LayerNorm]\n");
    int ln_N = 4096;
    std::vector<float> ln_in(ln_N), ln_out(ln_N);
    std::vector<float> ln_gamma(ln_N, 1.0f), ln_beta(ln_N, 0.0f);
    for (auto& x : ln_in) x = dist(rng);

    bench([&]() { layernorm(ln_in.data(), ln_N, 1e-5f,
                            ln_gamma.data(), ln_beta.data(), ln_out.data()); },
          "layernorm(N=4096)");

    // ============================================================
    // GELU benchmark
    // ============================================================
    printf("\n[GELU]\n");
    int gelu_N = 4096;
    std::vector<float> gelu_in(gelu_N), gelu_out(gelu_N);
    for (auto& x : gelu_in) x = dist(rng) * 5.0f;  // range [-5, 5]

    bench([&]() { gelu_vector_exact(gelu_in.data(), gelu_out.data(), gelu_N); },
          "gelu_vector_exact(N=4096)");

    bench([&]() { gelu_vector_approx(gelu_in.data(), gelu_out.data(), gelu_N); },
          "gelu_vector_approx(N=4096)");

    // ============================================================
    // Attention benchmark
    // ============================================================
    printf("\n[Scaled Dot-Product Attention]\n");
    int seq_lens[] = {32, 64, 128};

    for (int seq_len : seq_lens) {
        int d_k = 64, d_v = 64;
        std::vector<float> Q(seq_len * d_k), K(seq_len * d_k), V(seq_len * d_v), out;
        for (auto& x : Q) x = dist(rng);
        for (auto& x : K) x = dist(rng);
        for (auto& x : V) x = dist(rng);

        char label[64];
        snprintf(label, sizeof(label), "attention(seq=%d, d_k=%d, d_v=%d)",
                 seq_len, d_k, d_v);
        bench([&]() { scaled_dot_product_attention(Q, K, V, seq_len, d_k, d_v, out); },
              label, 5);
    }

    printf("\n============================================================\n");
    printf("  Benchmark complete.\n");
    printf("============================================================\n");
    return 0;
}
