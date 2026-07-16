#include "graph.h"
#include "json_parser.h"
#include "executor.h"
#include <cstdio>
#include <cassert>
#include <random>
#include <cmath>

// ===========================================================================
// Phase 3 Step 2: End-to-end Attention Block test
// ===========================================================================
// Tests:
//   1. Full pipeline: JSON -> Graph -> Topological sort -> Execute
//   2. Verify output dimensions
//   3. Verify numerical correctness (LayerNorm output ~N(0,1))
//   4. Manual graph execution (code-built, no JSON)
// ===========================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { fprintf(stderr, "FAIL: %s\n", msg); tests_failed++; } \
} while(0)

static float rand_float() {
    static std::mt19937 rng(42);
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    return dist(rng);
}

// ============================================================
// Test 1: Full pipeline with Attention Block JSON
// ============================================================
static void test_full_pipeline() {
    printf("\n=== Test 1: Full Pipeline (JSON -> Execute) ===\n");

    ComputeGraph graph;

    // Step 1: Parse JSON
    // Resolve path relative to project root (build/ -> ../configs/)
    bool ok = JsonParser::parse_file("../configs/attention_block.json", graph);
    if (!ok) ok = JsonParser::parse_file("../../configs/attention_block.json", graph);
    CHECK(ok, "JSON parsed successfully");
    if (!ok) return;  // skip remaining checks if JSON failed

    // Step 2: Create input tensors (128 tokens, 64-dim embeddings)
    int seq_len = 128, d_model = 64;
    Tensor x = Tensor::zeros("x", {seq_len, d_model});
    for (auto& v : x.data) v = rand_float();
    graph.set_tensor("x", std::move(x));

    // Weights: Wq, Wk, Wv (d_model x d_model)
    for (const char* wname : {"Wq", "Wk", "Wv"}) {
        Tensor w = Tensor::zeros(wname, {d_model, d_model});
        for (auto& v : w.data) v = rand_float() * 0.1f;  // small init
        graph.set_tensor(wname, std::move(w));
    }

    // Step 3: Topological sort
    auto sorted = graph.topological_sort();
    CHECK(sorted.size() == 5, "Topological sort: 5 nodes");

    // Step 4: Execute
    Executor exec;
    ok = exec.execute(sorted, graph);
    CHECK(ok, "Executor ran all nodes");

    // Step 5: Verify output
    const Tensor* out = graph.get_tensor("ln_out");
    CHECK(out != nullptr, "Output tensor ln_out exists");
    if (out) {
        CHECK(out->shape.size() == 2, "Output has 2 dimensions");
        CHECK(out->shape[0] == seq_len, "Output rows = seq_len");
        CHECK(out->shape[1] == d_model, "Output cols = d_model");

        // LayerNorm output should have mean ~0 and var ~1
        float mean = 0, var = 0;
        for (auto v : out->data) mean += v;
        mean /= out->data.size();
        for (auto v : out->data) var += (v - mean) * (v - mean);
        var /= out->data.size();

        printf("  LayerNorm output: mean=%.6f, var=%.6f\n", mean, var);
        CHECK(std::abs(mean) < 1e-4f, "LayerNorm output mean ~0");
        CHECK(std::abs(var - 1.0f) < 0.05f, "LayerNorm output var ~1");
    }
}

// ============================================================
// Test 2: Simple 3-node graph (MatMul -> GELU -> LayerNorm)
// ============================================================
static void test_simple_graph() {
    printf("\n=== Test 2: Simple 3-Node Graph ===\n");

    ComputeGraph graph;

    // Input: 32x32 matrix
    int M = 32, N = 32;
    Tensor x = Tensor::zeros("x", {M, N});
    for (auto& v : x.data) v = rand_float();
    graph.set_tensor("x", std::move(x));

    Tensor w = Tensor::zeros("w", {N, N});
    for (auto& v : w.data) v = rand_float() * 0.1f;
    graph.set_tensor("w", std::move(w));

    // Build graph manually
    Node proj{"proj", "MatMul", {"x", "w"}, {"proj"}};
    Node act{"act", "GELU", {"proj"}, {"act"}};
    Node norm{"norm", "LayerNorm", {"act"}, {"norm"}};
    graph.add_node(proj);
    graph.add_node(act);
    graph.add_node(norm);

    auto sorted = graph.topological_sort();
    CHECK(sorted.size() == 3, "Simple graph: 3 nodes");

    Executor exec;
    bool ok = exec.execute(sorted, graph);
    CHECK(ok, "Simple graph executed");

    const Tensor* out = graph.get_tensor("norm");
    CHECK(out != nullptr, "Output exists");
    if (out) {
        float mean = 0, var = 0;
        for (auto v : out->data) mean += v;
        mean /= out->data.size();
        for (auto v : out->data) var += (v - mean) * (v - mean);
        var /= out->data.size();
        CHECK(std::abs(mean) < 1e-4f, "Simple graph: mean ~0");
        CHECK(std::abs(var - 1.0f) < 0.05f, "Simple graph: var ~1");
    }
}

// ============================================================
// Test 3: Verify intermediate tensors
// ============================================================
static void test_intermediate_tensors() {
    printf("\n=== Test 3: Intermediate Tensor Verification ===\n");

    ComputeGraph graph;
    bool ok = JsonParser::parse_file("../configs/attention_block.json", graph);
    if (!ok) ok = JsonParser::parse_file("../../configs/attention_block.json", graph);
    CHECK(ok, "JSON parsed for intermediate test");
    if (!ok) return;

    int seq_len = 16, d_model = 32;
    Tensor x = Tensor::zeros("x", {seq_len, d_model});
    for (auto& v : x.data) v = rand_float();
    graph.set_tensor("x", std::move(x));

    for (const char* wname : {"Wq", "Wk", "Wv"}) {
        Tensor w = Tensor::zeros(wname, {d_model, d_model});
        for (auto& v : w.data) v = rand_float() * 0.1f;
        graph.set_tensor(wname, std::move(w));
    }

    auto sorted = graph.topological_sort();
    Executor exec;
    exec.execute(sorted, graph);

    // Check intermediate tensors exist
    CHECK(graph.get_tensor("q_proj") != nullptr, "q_proj tensor exists");
    CHECK(graph.get_tensor("k_proj") != nullptr, "k_proj tensor exists");
    CHECK(graph.get_tensor("v_proj") != nullptr, "v_proj tensor exists");
    CHECK(graph.get_tensor("attn_out") != nullptr, "attn_out tensor exists");
    CHECK(graph.get_tensor("ln_out") != nullptr, "ln_out tensor exists");

    // Check shapes
    CHECK(graph.get_tensor("q_proj")->shape[0] == seq_len, "q_proj rows");
    CHECK(graph.get_tensor("q_proj")->shape[1] == d_model, "q_proj cols");
    CHECK(graph.get_tensor("attn_out")->shape[0] == seq_len, "attn_out rows");
}

int main() {
    printf("============================================================\n");
    printf("  Phase 3 Step 2: End-to-End Execution Tests\n");
    printf("============================================================\n");

    test_full_pipeline();
    test_simple_graph();
    test_intermediate_tensors();

    printf("\n============================================================\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("============================================================\n");

    return (tests_failed == 0) ? 0 : 1;
}
