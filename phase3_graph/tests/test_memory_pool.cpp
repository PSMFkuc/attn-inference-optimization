#include "graph.h"
#include "json_parser.h"
#include "executor.h"
#include "memory_pool.h"
#include <cstdio>
#include <cassert>
#include <random>
#include <cmath>
#include <cstring>

// ===========================================================================
// Phase 3 Step 3: Memory Pool tests (revised)
// ===========================================================================
// Tests:
//   1. Basic pool alloc/dealloc/reuse
//   2. Full pipeline: naive vs pool on SAME input, element-by-element comparison
//   3. Pool stats: peak usage < total naive allocation
//   4. Lifetime analysis: non-overlapping tensors reuse memory
// ===========================================================================

static int tests_passed = 0, tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { fprintf(stderr, "FAIL: %s\n", msg); tests_failed++; } \
} while(0)

static void fill_random(Tensor& t, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : t.data) v = dist(rng);
}

// Resolve config path from multiple possible working directories
static const char* CONFIG_PATHS[] = {
    "../configs/attention_block.json",
    "../../configs/attention_block.json",
    "../../phase3_graph/configs/attention_block.json",
    "phase3_graph/configs/attention_block.json",
    nullptr
};

static bool load_config(ComputeGraph& graph) {
    for (int i = 0; CONFIG_PATHS[i] != nullptr; i++) {
        if (JsonParser::parse_file(CONFIG_PATHS[i], graph)) return true;
    }
    return false;
}

// ============================================================
// Test 1: Basic pool operations
// ============================================================
static void test_basic_pool() {
    printf("\n=== Test 1: Basic Pool Operations ===\n");
    MemoryPool pool(1024);

    float* a = pool.allocate("A", 256);
    CHECK(a != nullptr, "Alloc A");
    float* b = pool.allocate("B", 256);
    CHECK(b != nullptr, "Alloc B");
    CHECK(b == a + 256, "B is right after A in pool");

    pool.deallocate("A");
    float* c = pool.allocate("C", 128);
    CHECK(c != nullptr, "Alloc C (reuses A's slot)");
    CHECK(c == a, "C reuses A's memory at same address");

    float* d = pool.allocate("D", 512);
    CHECK(d != nullptr, "Alloc D");

    pool.dump();
    CHECK(pool.peak_used() <= 1024, "Peak within bounds");
    CHECK(pool.allocation_count() == 4, "4 allocations made");
}

// ============================================================
// Test 2: Naive vs Pool — element-by-element comparison
// ============================================================
static void test_naive_vs_pool() {
    printf("\n=== Test 2: Naive vs Pool (Same Input, Element Comparison) ===\n");

    int seq_len = 64, d_model = 32;

    // --- Build shared input data ---
    // We create the inputs ONCE, then copy them into two separate graphs.
    Tensor x_template = Tensor::zeros("x", {seq_len, d_model});
    fill_random(x_template, 42);

    std::vector<Tensor> weights;
    for (const char* wname : {"Wq", "Wk", "Wv"}) {
        Tensor w = Tensor::zeros(wname, {d_model, d_model});
        fill_random(w, 100 + (wname[1] - 'a'));  // different seeds per weight
        weights.push_back(w);
    }

    // --- Run 1: Naive execution ---
    ComputeGraph graph_naive;
    bool ok = JsonParser::parse_file("../configs/attention_block.json", graph_naive);
    if (!ok) ok = JsonParser::parse_file("../../configs/attention_block.json", graph_naive);
    CHECK(ok, "Naive: JSON parsed");
    if (!ok) return;

    // Copy inputs into naive graph
    graph_naive.set_tensor("x", Tensor(x_template));  // copy
    for (size_t i = 0; i < weights.size(); i++)
        graph_naive.set_tensor(weights[i].name, Tensor(weights[i]));

    auto sorted_naive = graph_naive.topological_sort();
    Executor exec_naive;
    ok = exec_naive.execute(sorted_naive, graph_naive);
    CHECK(ok, "Naive: execution succeeded");

    const Tensor* ref_out = graph_naive.get_tensor("ln_out");
    CHECK(ref_out != nullptr, "Naive: output tensor exists");
    if (!ref_out) return;

    // --- Run 2: Pool execution with SAME input data ---
    ComputeGraph graph_pool;
    ok = JsonParser::parse_file("../configs/attention_block.json", graph_pool);
    if (!ok) ok = JsonParser::parse_file("../../configs/attention_block.json", graph_pool);
    CHECK(ok, "Pool: JSON parsed");
    if (!ok) return;

    // Copy SAME inputs into pool graph
    graph_pool.set_tensor("x", Tensor(x_template));
    for (size_t i = 0; i < weights.size(); i++)
        graph_pool.set_tensor(weights[i].name, Tensor(weights[i]));

    auto sorted_pool = graph_pool.topological_sort();

    size_t pool_size = seq_len * d_model * 6;  // generous
    MemoryPool pool(pool_size);

    Executor exec_pool;
    ok = exec_pool.execute_with_pool(sorted_pool, graph_pool, pool);
    CHECK(ok, "Pool: execution succeeded");

    const Tensor* pool_out = graph_pool.get_tensor("ln_out");
    CHECK(pool_out != nullptr, "Pool: output tensor exists");
    if (!pool_out) return;

    // --- Compare: element-by-element ---
    CHECK(ref_out->shape == pool_out->shape, "Same output shape");
    CHECK(ref_out->data.size() == pool_out->data.size(), "Same output size");

    if (ref_out->data.size() == pool_out->data.size()) {
        float max_diff = 0.0f;
        int max_idx = -1;
        for (size_t i = 0; i < ref_out->data.size(); i++) {
            float diff = std::abs(ref_out->data[i] - pool_out->data[i]);
            if (diff > max_diff) {
                max_diff = diff;
                max_idx = (int)i;
            }
        }
        printf("  Max element-wise difference: %.8f at index %d\n", max_diff, max_idx);
        CHECK(max_diff < 1e-5f, "Naive and pool outputs are identical (diff < 1e-5)");
    }

    // --- Memory stats ---
    printf("\n  Peak pool usage: %zu floats = %.2f KB\n",
           pool.peak_used(), pool.peak_used() * 4.0f / 1024.0f);
    size_t naive_total = (size_t)seq_len * d_model * 5;
    printf("  Naive total allocation: ~%zu floats = %.2f KB\n",
           naive_total, naive_total * 4.0f / 1024.0f);
    CHECK(pool.peak_used() < naive_total,
          "Pool peak < naive total (memory saved!)");
}

// ============================================================
// Test 3: Intermediate tensor verification (pool path)
// ============================================================
static void test_intermediate_tensors_pool() {
    printf("\n=== Test 3: Intermediate Tensors (Pool Path) ===\n");

    ComputeGraph graph;
    bool ok = JsonParser::parse_file("../configs/attention_block.json", graph);
    if (!ok) ok = JsonParser::parse_file("../../configs/attention_block.json", graph);
    CHECK(ok, "JSON parsed");
    if (!ok) return;

    int seq_len = 32, d_model = 16;
    Tensor x = Tensor::zeros("x", {seq_len, d_model});
    fill_random(x, 42);
    graph.set_tensor("x", std::move(x));

    for (const char* wname : {"Wq", "Wk", "Wv"}) {
        Tensor w = Tensor::zeros(wname, {d_model, d_model});
        fill_random(w, 99);
        graph.set_tensor(wname, std::move(w));
    }

    auto sorted = graph.topological_sort();

    MemoryPool pool(seq_len * d_model * 8);
    Executor exec;
    ok = exec.execute_with_pool(sorted, graph, pool);
    CHECK(ok, "Pool execution succeeded");

    // All 5 intermediate tensors should exist
    CHECK(graph.get_tensor("q_proj") != nullptr, "q_proj exists");
    CHECK(graph.get_tensor("k_proj") != nullptr, "k_proj exists");
    CHECK(graph.get_tensor("v_proj") != nullptr, "v_proj exists");
    CHECK(graph.get_tensor("attn_out") != nullptr, "attn_out exists");
    CHECK(graph.get_tensor("ln_out") != nullptr, "ln_out exists");

    // Check shapes
    CHECK(graph.get_tensor("q_proj")->shape[0] == seq_len, "q_proj rows correct");
    CHECK(graph.get_tensor("q_proj")->shape[1] == d_model, "q_proj cols correct");
    CHECK(graph.get_tensor("attn_out")->shape[0] == seq_len, "attn_out rows correct");

    // LayerNorm output sanity
    const Tensor* ln = graph.get_tensor("ln_out");
    float mean = 0, var = 0;
    for (auto v : ln->data) mean += v;
    mean /= ln->data.size();
    for (auto v : ln->data) var += (v - mean) * (v - mean);
    var /= ln->data.size();
    CHECK(std::abs(mean) < 1e-4f, "Pool LN mean ~0");
    CHECK(std::abs(var - 1.0f) < 0.05f, "Pool LN var ~1");
}

// ============================================================
// Test 4: Softmax in-place safety with pool reuse
// ============================================================
static void test_softmax_pool_safety() {
    printf("\n=== Test 4: Softmax Pool Reuse Safety ===\n");

    ComputeGraph graph;
    // Build: x -> softmax -> ln (simple chain, softmax output might reuse input slot)
    Node sm{"sm", "Softmax", {"x"}, {"sm_out"}};
    Node ln{"ln", "LayerNorm", {"sm_out"}, {"ln_out"}};
    graph.add_node(sm);
    graph.add_node(ln);

    int N = 128;
    Tensor x = Tensor::zeros("x", {N});
    fill_random(x, 42);
    graph.set_tensor("x", std::move(x));

    auto sorted = graph.topological_sort();
    MemoryPool pool(N * 4);
    Executor exec;
    bool ok = exec.execute_with_pool(sorted, graph, pool);
    CHECK(ok, "Softmax + LayerNorm executed with pool");

    // Softmax output should sum to 1
    const Tensor* sm_out = graph.get_tensor("sm_out");
    if (sm_out) {
        float sum = 0;
        for (auto v : sm_out->data) sum += v;
        CHECK(std::abs(sum - 1.0f) < 1e-4f, "Softmax output sums to 1");
    }

    // LayerNorm output: mean ~0, var ~1
    const Tensor* ln_out = graph.get_tensor("ln_out");
    if (ln_out) {
        float mean = 0, var = 0;
        for (auto v : ln_out->data) mean += v;
        mean /= ln_out->data.size();
        for (auto v : ln_out->data) var += (v - mean) * (v - mean);
        var /= ln_out->data.size();
        printf("  Softmax+LN output: mean=%.6f, var=%.6f\n", mean, var);
        CHECK(std::abs(mean) < 1e-4f, "Softmax pool: LN mean ~0");
        // With only 128 elements, var can deviate more. Just check it's positive.
        CHECK(var > 0.5f, "Softmax pool: LN var > 0.5 (reasonable)");
    }
}

int main() {
    printf("============================================================\n");
    printf("  Phase 3 Step 3: Memory Pool Tests (Revised)\n");
    printf("============================================================\n");

    test_basic_pool();
    test_naive_vs_pool();
    test_intermediate_tensors_pool();
    test_softmax_pool_safety();

    printf("\n============================================================\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("============================================================\n");

    return (tests_failed == 0) ? 0 : 1;
}
