#include "executor.h"
#include "operators.h"       // Phase 2: softmax, layernorm, gelu, attention
#include "gemm_naive.h"      // Phase 1: gemm_naive_ikj
#include <cstdio>
#include <cstring>
#include <cassert>
#include <unordered_set>

// ===========================================================================
// Helper: find a tensor, with error message
// ===========================================================================

static Tensor* require_tensor(ComputeGraph& graph, const std::string& name,
                              const std::string& node_id) {
    Tensor* t = graph.get_tensor(name);
    if (!t) {
        fprintf(stderr, "[Executor] Node '%s': input tensor '%s' not found!\n",
                node_id.c_str(), name.c_str());
    }
    return t;
}

// ===========================================================================
// Lifetime analysis (revised)
// ===========================================================================
// For each intermediate tensor, compute:
//   first_use: index of the first node that reads this tensor
//   last_use:  index of the last node that reads this tensor
//
// A tensor can be freed (its memory reclaimed) after last_use.
// Two tensors with non-overlapping lifetimes can share the same memory.
//
// Revision: output tensor sizes are now set to 0 initially (unknown).
// The actual size is filled in when the tensor is first allocated in
// the pool. This avoids inaccurate pre-estimation.

struct TensorLifetime {
    int first_use;
    int last_use;
    bool is_input;   // input tensors are never freed
};

static std::unordered_map<std::string, TensorLifetime>
analyze_lifetimes(const std::vector<const Node*>& sorted_nodes,
                  const ComputeGraph& graph) {
    std::unordered_map<std::string, TensorLifetime> lifetimes;

    for (int idx = 0; idx < (int)sorted_nodes.size(); idx++) {
        const Node* node = sorted_nodes[idx];

        // Each node's inputs: update last_use for these tensors
        for (const auto& input : node->inputs) {
            Tensor* t = const_cast<ComputeGraph&>(graph).get_tensor(input);
            if (!t) continue;  // skip weights/inputs not in tensor store yet

            auto it = lifetimes.find(input);
            if (it == lifetimes.end()) {
                TensorLifetime lt;
                lt.first_use = idx;
                lt.last_use = idx;
                lt.is_input = false;
                lifetimes[input] = lt;
            } else {
                it->second.last_use = idx;
            }
        }

        // Each node's outputs: register with first_use = idx
        for (const auto& output : node->outputs) {
            TensorLifetime lt;
            lt.first_use = idx;
            lt.last_use = idx;   // will be updated when consumed
            lt.is_input = false;
            lifetimes[output] = lt;
        }
    }

    return lifetimes;
}

// ===========================================================================
// Operator dispatch (naive allocation path, unchanged)
// ===========================================================================

static bool execute_matmul(const Node& node, ComputeGraph& graph) {
    assert(node.inputs.size() >= 2);
    Tensor* A = require_tensor(graph, node.inputs[0], node.id);
    Tensor* B = require_tensor(graph, node.inputs[1], node.id);
    if (!A || !B) return false;

    int M = A->shape[0];
    int K_A = A->shape.size() >= 2 ? A->shape[1] : 1;
    int K_B = B->shape.size() >= 2 ? B->shape[0] : 1;
    int N = B->shape.size() >= 2 ? B->shape[1] : B->data.size();

    if (K_A != K_B) {
        fprintf(stderr, "[Executor] MatMul '%s': dimension mismatch\n", node.id.c_str());
        return false;
    }

    Tensor C = Tensor::zeros(node.outputs[0], {M, N});
    gemm_naive_ikj(M, N, K_A, A->data, B->data, C.data);
    graph.set_tensor(node.outputs[0], std::move(C));
    return true;
}

static bool execute_softmax(const Node& node, ComputeGraph& graph) {
    assert(node.inputs.size() >= 1);
    Tensor* in = require_tensor(graph, node.inputs[0], node.id);
    if (!in) return false;
    Tensor out = *in;
    out.name = node.outputs[0];
    int rows = 1, cols = out.data.size();
    if (out.shape.size() >= 2) { rows = out.shape[0]; cols = out.shape[1]; }
    softmax_2d(out.data.data(), rows, cols);
    graph.set_tensor(node.outputs[0], std::move(out));
    return true;
}

static bool execute_layernorm(const Node& node, ComputeGraph& graph) {
    assert(node.inputs.size() >= 1);
    Tensor* in = require_tensor(graph, node.inputs[0], node.id);
    if (!in) return false;
    int N = in->data.size();
    float eps = 1e-5f;
    auto it = node.params.find("eps");
    if (it != node.params.end()) eps = it->second;
    std::vector<float> gamma(N, 1.0f), beta(N, 0.0f);
    Tensor out = Tensor::zeros(node.outputs[0], in->shape);
    layernorm(in->data.data(), N, eps, gamma.data(), beta.data(), out.data.data());
    graph.set_tensor(node.outputs[0], std::move(out));
    return true;
}

static bool execute_gelu(const Node& node, ComputeGraph& graph) {
    assert(node.inputs.size() >= 1);
    Tensor* in = require_tensor(graph, node.inputs[0], node.id);
    if (!in) return false;
    Tensor out = Tensor::zeros(node.outputs[0], in->shape);
    gelu_vector_exact(in->data.data(), out.data.data(), in->data.size());
    graph.set_tensor(node.outputs[0], std::move(out));
    return true;
}

static bool execute_attention(const Node& node, ComputeGraph& graph) {
    assert(node.inputs.size() >= 3);
    Tensor* Q = require_tensor(graph, node.inputs[0], node.id);
    Tensor* K = require_tensor(graph, node.inputs[1], node.id);
    Tensor* V = require_tensor(graph, node.inputs[2], node.id);
    if (!Q || !K || !V) return false;
    int seq_len = Q->shape[0], d_k = Q->shape[1], d_v = V->shape[1];
    std::vector<float> out_data;
    scaled_dot_product_attention(Q->data, K->data, V->data, seq_len, d_k, d_v, out_data);
    Tensor out;
    out.name = node.outputs[0];
    out.shape = {seq_len, d_v};
    out.data = std::move(out_data);
    graph.set_tensor(node.outputs[0], std::move(out));
    return true;
}

// ===========================================================================
// Pool-based operator dispatch (uses memory pool instead of vector<float>)
// ===========================================================================

static bool execute_matmul_pool(const Node& node, ComputeGraph& graph, MemoryPool& pool) {
    assert(node.inputs.size() >= 2);
    Tensor* A = require_tensor(graph, node.inputs[0], node.id);
    Tensor* B = require_tensor(graph, node.inputs[1], node.id);
    if (!A || !B) return false;

    int M = A->shape[0];
    int K_A = A->shape.size() >= 2 ? A->shape[1] : 1;
    int K_B = B->shape.size() >= 2 ? B->shape[0] : 1;
    int N = B->shape.size() >= 2 ? B->shape[1] : B->data.size();

    if (K_A != K_B) return false;

    float* C_data = pool.allocate(node.outputs[0], M * N);
    if (!C_data) return false;

    Tensor C;
    C.name = node.outputs[0];
    C.shape = {M, N};
    C.data.assign(C_data, C_data + M * N);
    gemm_naive_ikj(M, N, K_A, A->data, B->data, C.data);
    std::memcpy(C_data, C.data.data(), M * N * sizeof(float));
    graph.set_tensor(node.outputs[0], std::move(C));
    return true;
}

static bool execute_softmax_pool(const Node& node, ComputeGraph& graph, MemoryPool& pool) {
    assert(node.inputs.size() >= 1);
    Tensor* in = require_tensor(graph, node.inputs[0], node.id);
    if (!in) return false;

    int n = in->data.size();
    float* out_data = pool.allocate(node.outputs[0], n);
    if (!out_data) return false;

    // Safety: if input and output share the same pool slot (reuse scenario),
    // we must copy first before in-place modification.
    // Check: are they the same pointer?
    bool same_buffer = (out_data == in->data.data());
    if (same_buffer) {
        // Softmax reads all elements before writing, so same-buffer is safe
        // because softmax does 3 passes: find max, exp+sum, normalize.
        // Pass 1 only reads, pass 2 reads then overwrites, pass 3 reads then overwrites.
        // Actually: softmax is in-place on x[] itself. We pass out_data which
        // already has the input data copied. So it's safe regardless.
    }
    std::memcpy(out_data, in->data.data(), n * sizeof(float));

    int rows = 1, cols = n;
    if (in->shape.size() >= 2) { rows = in->shape[0]; cols = in->shape[1]; }
    softmax_2d(out_data, rows, cols);

    Tensor out;
    out.name = node.outputs[0];
    out.shape = in->shape;
    out.data.assign(out_data, out_data + n);
    graph.set_tensor(node.outputs[0], std::move(out));
    return true;
}

static bool execute_layernorm_pool(const Node& node, ComputeGraph& graph, MemoryPool& pool) {
    assert(node.inputs.size() >= 1);
    Tensor* in = require_tensor(graph, node.inputs[0], node.id);
    if (!in) return false;

    int N = in->data.size();
    float eps = 1e-5f;
    auto it = node.params.find("eps");
    if (it != node.params.end()) eps = it->second;

    float* out_data = pool.allocate(node.outputs[0], N);
    if (!out_data) return false;

    std::vector<float> gamma(N, 1.0f), beta(N, 0.0f);
    layernorm(in->data.data(), N, eps, gamma.data(), beta.data(), out_data);

    Tensor out;
    out.name = node.outputs[0];
    out.shape = in->shape;
    out.data.assign(out_data, out_data + N);
    graph.set_tensor(node.outputs[0], std::move(out));
    return true;
}

static bool execute_gelu_pool(const Node& node, ComputeGraph& graph, MemoryPool& pool) {
    assert(node.inputs.size() >= 1);
    Tensor* in = require_tensor(graph, node.inputs[0], node.id);
    if (!in) return false;

    int N = in->data.size();
    float* out_data = pool.allocate(node.outputs[0], N);
    if (!out_data) return false;
    gelu_vector_exact(in->data.data(), out_data, N);

    Tensor out;
    out.name = node.outputs[0];
    out.shape = in->shape;
    out.data.assign(out_data, out_data + N);
    graph.set_tensor(node.outputs[0], std::move(out));
    return true;
}

static bool execute_attention_pool(const Node& node, ComputeGraph& graph, MemoryPool& pool) {
    assert(node.inputs.size() >= 3);
    Tensor* Q = require_tensor(graph, node.inputs[0], node.id);
    Tensor* K = require_tensor(graph, node.inputs[1], node.id);
    Tensor* V = require_tensor(graph, node.inputs[2], node.id);
    if (!Q || !K || !V) return false;

    int seq_len = Q->shape[0], d_k = Q->shape[1], d_v = V->shape[1];

    std::vector<float> out_data;
    scaled_dot_product_attention(Q->data, K->data, V->data, seq_len, d_k, d_v, out_data);

    float* pool_data = pool.allocate(node.outputs[0], out_data.size());
    if (!pool_data) return false;
    std::memcpy(pool_data, out_data.data(), out_data.size() * sizeof(float));

    Tensor out;
    out.name = node.outputs[0];
    out.shape = {seq_len, d_v};
    out.data.assign(pool_data, pool_data + out_data.size());
    graph.set_tensor(node.outputs[0], std::move(out));
    return true;
}

// ===========================================================================
// Execute (naive allocation)
// ===========================================================================

bool Executor::execute(const std::vector<const Node*>& sorted_nodes,
                       ComputeGraph& graph) {
    printf("[Executor] Running %zu nodes (naive alloc)...\n", sorted_nodes.size());

    for (const Node* node : sorted_nodes) {
        printf("  [%s] op=%s", node->id.c_str(), node->op_type.c_str());

        bool ok = false;
        if (node->op_type == "MatMul")       ok = execute_matmul(*node, graph);
        else if (node->op_type == "Softmax")  ok = execute_softmax(*node, graph);
        else if (node->op_type == "LayerNorm") ok = execute_layernorm(*node, graph);
        else if (node->op_type == "GELU")     ok = execute_gelu(*node, graph);
        else if (node->op_type == "Attention") ok = execute_attention(*node, graph);
        else { fprintf(stderr, " -> UNKNOWN OP\n"); return false; }

        if (!ok) { fprintf(stderr, " -> FAILED\n"); return false; }

        Tensor* out = graph.get_tensor(node->outputs[0]);
        if (out) {
            printf(" -> shape=[");
            for (size_t i = 0; i < out->shape.size(); i++)
                printf("%d%s", out->shape[i], i+1<out->shape.size()?"x":"");
            printf("]\n");
        }
    }
    printf("[Executor] Done.\n");
    return true;
}

// ===========================================================================
// Execute with memory pool
// ===========================================================================

bool Executor::execute_with_pool(const std::vector<const Node*>& sorted_nodes,
                                 ComputeGraph& graph,
                                 MemoryPool& pool) {
    printf("[Executor-Pool] Running %zu nodes (pooled alloc)...\n", sorted_nodes.size());

    // Step 1: Analyze lifetimes (without size estimation)
    auto lifetimes = analyze_lifetimes(sorted_nodes, graph);
    printf("[Executor-Pool] Analyzed %zu tensor lifetimes\n", lifetimes.size());

    // Step 2: Execute nodes, freeing tensors after last use
    for (int idx = 0; idx < (int)sorted_nodes.size(); idx++) {
        const Node* node = sorted_nodes[idx];
        printf("  [%s] op=%s", node->id.c_str(), node->op_type.c_str());

        // Execute the node with pool allocation
        bool ok = false;
        if (node->op_type == "MatMul")       ok = execute_matmul_pool(*node, graph, pool);
        else if (node->op_type == "Softmax")  ok = execute_softmax_pool(*node, graph, pool);
        else if (node->op_type == "LayerNorm") ok = execute_layernorm_pool(*node, graph, pool);
        else if (node->op_type == "GELU")     ok = execute_gelu_pool(*node, graph, pool);
        else if (node->op_type == "Attention") ok = execute_attention_pool(*node, graph, pool);
        else { fprintf(stderr, " -> UNKNOWN OP\n"); return false; }

        if (!ok) { fprintf(stderr, " -> FAILED\n"); return false; }

        // Step 3: Free tensors whose last use is this node
        int freed = 0;
        for (auto& [name, lt] : lifetimes) {
            if (!lt.is_input && lt.last_use == idx) {
                pool.deallocate(name);
                freed++;
            }
        }

        Tensor* out = graph.get_tensor(node->outputs[0]);
        if (out) {
            printf(" -> shape=[");
            for (size_t i = 0; i < out->shape.size(); i++)
                printf("%d%s", out->shape[i], i+1<out->shape.size()?"x":"");
            printf("] (freed %d)", freed);
        }
        printf("\n");
    }

    printf("[Executor-Pool] Done. ");
    pool.dump();
    return true;
}

const Tensor* Executor::get_output(const ComputeGraph& graph,
                                   const std::string& name) const {
    auto& g = const_cast<ComputeGraph&>(graph);
    return g.get_tensor(name);
}
