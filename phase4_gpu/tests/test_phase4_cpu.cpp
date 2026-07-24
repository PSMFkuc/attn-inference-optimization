#include "backend.h"
#include "parallel_graph_executor.h"
#include "quantization.h"
#include "gemm_naive.h"
#include "graph.h"

#include <cmath>
#include <cstdio>
#include <random>

using phase4::CpuParallelBackend;
using phase4::ParallelGraphExecutor;

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { std::fprintf(stderr, "FAIL: %s\n", msg); tests_failed++; } \
} while (0)

static float rand_float() {
    static std::mt19937 rng(123);
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

static float max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return INFINITY;
    float diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        diff = std::max(diff, std::fabs(a[i] - b[i]));
    }
    return diff;
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

static void test_execution_levels() {
    std::printf("\n=== Test 1: DAG Levelization ===\n");
    CpuParallelBackend backend;
    ParallelGraphExecutor executor(backend);
    ComputeGraph graph = make_attention_graph(16, 32);
    auto sorted = graph.topological_sort();
    auto levels = executor.build_execution_levels(sorted);

    CHECK(levels.size() == 3, "attention graph has 3 execution levels");
    CHECK(levels[0].size() == 3, "q/k/v projection nodes share level 0");
    CHECK(levels[1].size() == 1, "attention is level 1");
    CHECK(levels[2].size() == 1, "layernorm is level 2");
}

static void test_serial_matches_level_parallel() {
    std::printf("\n=== Test 2: Serial vs Level-Parallel Execution ===\n");
    CpuParallelBackend backend;
    ParallelGraphExecutor executor(backend);

    ComputeGraph serial_graph = make_attention_graph(16, 32);
    ComputeGraph parallel_graph = serial_graph;
    auto sorted = serial_graph.topological_sort();

    bool ok_serial = executor.execute_serial(sorted, serial_graph);
    bool ok_parallel = executor.execute_level_parallel(sorted, parallel_graph);
    CHECK(ok_serial, "serial Phase4 execution succeeds");
    CHECK(ok_parallel, "level-parallel Phase4 execution succeeds");

    Tensor* serial_out = serial_graph.get_tensor("ln_out");
    Tensor* parallel_out = parallel_graph.get_tensor("ln_out");
    CHECK(serial_out != nullptr, "serial output exists");
    CHECK(parallel_out != nullptr, "parallel output exists");
    if (serial_out && parallel_out) {
        CHECK(max_diff(serial_out->data, parallel_out->data) < 1e-5f,
              "serial and level-parallel outputs match");
    }
}

static void test_rowwise_layernorm() {
    std::printf("\n=== Test 3: Row-wise LayerNorm Semantics ===\n");
    CpuParallelBackend backend;
    Node node{"ln", "LayerNorm", {"x"}, {"ln"}, {{"eps", 1e-5f}}};
    std::vector<Tensor> inputs = {make_tensor("x", 8, 16)};
    Tensor out;
    std::string error;
    bool ok = backend.execute(node, inputs, out, error);
    CHECK(ok, "LayerNorm backend execution succeeds");

    const int rows = out.shape[0];
    const int cols = out.shape[1];
    for (int r = 0; r < rows; ++r) {
        float mean = 0.0f;
        for (int c = 0; c < cols; ++c) mean += out.data[r * cols + c];
        mean /= cols;
        float var = 0.0f;
        for (int c = 0; c < cols; ++c) {
            const float d = out.data[r * cols + c] - mean;
            var += d * d;
        }
        var /= cols;
        CHECK(std::fabs(mean) < 1e-4f, "row LayerNorm mean is near zero");
        CHECK(std::fabs(var - 1.0f) < 1e-3f, "row LayerNorm variance is near one");
    }
}

static void test_quantized_weight_gemm() {
    std::printf("\n=== Test 4: INT8 Weight GEMM ===\n");
    const int M = 16, K = 32, N = 24;
    Tensor A = make_tensor("A", M, K);
    Tensor B = make_tensor("B", K, N, 0.25f);

    std::vector<float> ref;
    gemm_naive_ikj(M, N, K, A.data, B.data, ref);

    auto Bq = phase4::quantize_weight_per_output_channel(B.data, K, N);
    std::vector<float> out;
    phase4::gemm_fp32_int8_weight(M, N, K, A.data, Bq, out);

    const float err = phase4::max_abs_error(ref, out);
    std::printf("  max_abs_error=%.6f\n", err);
    CHECK(err < 0.05f, "INT8 weight GEMM stays within expected error");
}

int main() {
    std::printf("============================================================\n");
    std::printf("  Phase 4 CPU Inference Acceleration Tests\n");
    std::printf("============================================================\n");

    test_execution_levels();
    test_serial_matches_level_parallel();
    test_rowwise_layernorm();
    test_quantized_weight_gemm();

    std::printf("\n============================================================\n");
    std::printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    std::printf("============================================================\n");
    return tests_failed == 0 ? 0 : 1;
}
