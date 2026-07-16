#include "graph.h"
#include "json_parser.h"
#include <cstdio>
#include <cassert>
#include <string>

// ===========================================================================
// Phase 3 Step 1: Graph data structure + JSON parser + Topological sort
// ===========================================================================
// Tests:
//   1. Parse JSON config → build graph
//   2. Verify node count and types
//   3. Topological sort → verify execution order
//   4. Manual graph construction + cycle detection
// ===========================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { fprintf(stderr, "FAIL: %s\n", msg); tests_failed++; } \
} while(0)

// ============================================================
// Test 1: JSON parsing
// ============================================================
static void test_json_parsing() {
    printf("\n=== Test 1: JSON Parsing ===\n");
    ComputeGraph graph;

    bool ok = JsonParser::parse_file("../configs/attention_block.json", graph);
    CHECK(ok, "JSON parse succeeded");
    CHECK(graph.node_count() == 5, "Graph has 5 nodes");

    // Verify specific nodes
    const Node* q = graph.find_node("q_proj");
    CHECK(q != nullptr, "q_proj node exists");
    CHECK(q && q->op_type == "MatMul", "q_proj is MatMul");

    const Node* attn = graph.find_node("attn_out");
    CHECK(attn != nullptr, "attn_out node exists");
    CHECK(attn && attn->op_type == "Attention", "attn_out is Attention");
    CHECK(attn && attn->inputs.size() == 3, "Attention has 3 inputs");

    const Node* ln = graph.find_node("ln_out");
    CHECK(ln != nullptr, "ln_out node exists");
    CHECK(ln && ln->op_type == "LayerNorm", "ln_out is LayerNorm");
    CHECK(ln && ln->params.count("eps") > 0, "LayerNorm has eps param");

    printf("  Graph dump:\n");
    graph.dump();
}

// ============================================================
// Test 2: Topological sort
// ============================================================
static void test_topological_sort() {
    printf("\n=== Test 2: Topological Sort ===\n");
    ComputeGraph graph;
    JsonParser::parse_file("../configs/attention_block.json", graph);

    auto sorted = graph.topological_sort();
    CHECK(sorted.size() == 5, "Topological sort has 5 nodes");
    CHECK(!sorted.empty(), "Sort result is non-empty");

    printf("  Execution order:\n");
    for (const auto* n : sorted) {
        printf("    %s (%s)\n", n->id.c_str(), n->op_type.c_str());
    }

    // Verify ordering constraints:
    // - Projections (q/k/v_proj) must come before attn_out
    // - attn_out must come before ln_out
    std::unordered_map<std::string, int> order;
    for (size_t i = 0; i < sorted.size(); i++) {
        order[sorted[i]->id] = (int)i;
    }

    CHECK(order["q_proj"] < order["attn_out"], "q_proj before attn_out");
    CHECK(order["k_proj"] < order["attn_out"], "k_proj before attn_out");
    CHECK(order["v_proj"] < order["attn_out"], "v_proj before attn_out");
    CHECK(order["attn_out"] < order["ln_out"], "attn_out before ln_out");
}

// ============================================================
// Test 3: Manual graph construction (no JSON)
// ============================================================
static void test_manual_graph() {
    printf("\n=== Test 3: Manual Graph Construction ===\n");
    ComputeGraph graph;

    // Build a simple graph: A -> B -> C
    Node a{"A", "Softmax", {"x"}, {"A"}};
    Node b{"B", "LayerNorm", {"A"}, {"B"}};
    Node c{"C", "GELU", {"B"}, {"C"}};
    graph.add_node(a);
    graph.add_node(b);
    graph.add_node(c);

    auto sorted = graph.topological_sort();
    CHECK(sorted.size() == 3, "Manual graph has 3 nodes");
    CHECK(sorted[0]->id == "A", "First is A");
    CHECK(sorted[1]->id == "B", "Second is B");
    CHECK(sorted[2]->id == "C", "Third is C");
}

// ============================================================
// Test 4: Cycle detection
// ============================================================
static void test_cycle_detection() {
    printf("\n=== Test 4: Cycle Detection ===\n");
    ComputeGraph graph;

    // A -> B -> A (cycle!)
    Node a{"A", "Softmax", {"B"}, {"A"}};  // A depends on B
    Node b{"B", "Softmax", {"A"}, {"B"}};  // B depends on A
    graph.add_node(a);
    graph.add_node(b);

    auto sorted = graph.topological_sort();
    // Should return empty vector for cyclic graph
    CHECK(sorted.empty(), "Cycle detected: sort returns empty");
}

int main() {
    printf("============================================================\n");
    printf("  Phase 3 Step 1: Graph Engine Tests\n");
    printf("============================================================\n");

    test_json_parsing();
    test_topological_sort();
    test_manual_graph();
    test_cycle_detection();

    printf("\n============================================================\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("============================================================\n");

    return (tests_failed == 0) ? 0 : 1;
}
