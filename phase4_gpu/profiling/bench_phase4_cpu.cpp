#include "backend.h"
#include "gemm_naive.h"
#include "gemm_parallel.h"
#include "parallel_graph_executor.h"
#include "quantization.h"
#include "graph.h"

#include <chrono>
#include <cstdio>
#include <random>

using Clock = std::chrono::high_resolution_clock;

static float rand_float() {
    static std::mt19937 rng(2026);
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    return dist(rng);
}

static Tensor make_tensor(const char* name, int rows, int cols, float scale = 1.0f) {
    Tensor t = Tensor::zeros(name, {rows, cols});
    for (float& v : t.data) {
        v = rand_float() * scale;
    }
    return t;
}

template <typename Fn>
static double time_ms(Fn&& fn, int iters = 3) {
    double best = 1e100;
    for (int i = 0; i < iters; ++i) {
        auto start = Clock::now();
        fn();
        auto end = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (ms < best) best = ms;
    }
    return best;
}

static ComputeGraph make_attention_graph(int seq_len, int d_model) {
    ComputeGraph graph;
    graph.set_tensor("x", make_tensor("x", seq_len, d_model));
    graph.set_tensor("Wq", make_tensor("Wq", d_model, d_model, 0.1f));
    graph.set_tensor("Wk", make_tensor("Wk", d_model, d_model, 0.1f));
    graph.set_tensor("Wv", make_tensor("Wv", d_model, d_model, 0.1f));
    graph.add_node(Node{"q_proj", "MatMul", {"x", "Wq"}, {"q_proj"}, {}});
    graph.add_node(Node{"k_proj", "MatMul", {"x", "Wk"}, {"k_proj"}, {}});
    graph.add_node(Node{"v_proj", "MatMul", {"x", "Wv"}, {"v_proj"}, {}});
    graph.add_node(Node{"attn_out", "Attention", {"q_proj", "k_proj", "v_proj"}, {"attn_out"}, {}});
    graph.add_node(Node{"ln_out", "LayerNorm", {"attn_out"}, {"ln_out"}, {{"eps", 1e-5f}}});
    return graph;
}

static void bench_gemm_paths() {
    std::printf("\n=== GEMM Paths ===\n");
    std::printf("%-8s %-12s %-12s %-12s %-12s\n",
                "size", "naive(ms)", "parallel(ms)", "int8w(ms)", "int8_err");

    for (int n : {128, 256, 512}) {
        Tensor A = make_tensor("A", n, n);
        Tensor B = make_tensor("B", n, n, 0.25f);
        std::vector<float> ref;
        std::vector<float> parallel_out;
        std::vector<float> int8_out;
        auto Bq = phase4::quantize_weight_per_output_channel(B.data, n, n);

        double naive_ms = time_ms([&]() {
            gemm_naive_ikj(n, n, n, A.data, B.data, ref);
        }, 2);
        double parallel_ms = time_ms([&]() {
            gemm_parallel_tiled(n, n, n, A.data, B.data, parallel_out);
        }, 3);
        double int8_ms = time_ms([&]() {
            phase4::gemm_fp32_int8_weight(n, n, n, A.data, Bq, int8_out);
        }, 3);
        float err = phase4::max_abs_error(ref, int8_out);

        std::printf("%-8d %-12.3f %-12.3f %-12.3f %-12.5f\n",
                    n, naive_ms, parallel_ms, int8_ms, err);
    }
}

static void bench_graph_schedule() {
    std::printf("\n=== Graph Schedule ===\n");
    std::printf("%-16s %-12s %-12s %-12s\n",
                "shape", "serial(ms)", "level(ms)", "levels");

    for (auto [seq_len, d_model] : {std::pair<int, int>{64, 64},
                                    std::pair<int, int>{128, 64}}) {
        phase4::CpuParallelBackend backend;
        phase4::ParallelGraphExecutor executor(backend, false);
        ComputeGraph graph = make_attention_graph(seq_len, d_model);
        auto sorted = graph.topological_sort();
        auto levels = executor.build_execution_levels(sorted);

        double serial_ms = time_ms([&]() {
            ComputeGraph g = graph;
            executor.execute_serial(sorted, g);
        }, 3);
        double level_ms = time_ms([&]() {
            ComputeGraph g = graph;
            executor.execute_level_parallel(sorted, g);
        }, 3);

        std::printf("%dx%-13d %-12.3f %-12.3f %-12zu\n",
                    seq_len, d_model, serial_ms, level_ms, levels.size());
    }
}

int main() {
    std::printf("============================================================\n");
    std::printf("  Phase 4 CPU Inference Acceleration Benchmark\n");
    std::printf("============================================================\n");

    bench_gemm_paths();
    bench_graph_schedule();
    return 0;
}
