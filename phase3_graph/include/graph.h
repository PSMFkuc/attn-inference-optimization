#ifndef PHASE3_GRAPH_H
#define PHASE3_GRAPH_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <variant>

// ===========================================================================
// Tensor: a named multi-dimensional array of floats
// ===========================================================================
// Minimal tensor abstraction. In production you'd want strides, device
// placement, dtype enum, etc. For Phase 3, a flat float vector + shape
// is enough to feed into Phase 2 operators.

struct Tensor {
    std::string name;
    std::vector<int> shape;       // e.g. [128, 64] for 128x64 matrix
    std::vector<float> data;

    int total_size() const {
        int sz = 1;
        for (int d : shape) sz *= d;
        return sz;
    }

    // Create an uninitialized tensor of given shape
    static Tensor zeros(const std::string& name,
                        const std::vector<int>& shape) {
        Tensor t;
        t.name = name;
        t.shape = shape;
        int n = 1;
        for (int d : shape) n *= d;
        t.data.resize(n, 0.0f);
        return t;
    }
};

// ===========================================================================
// Graph Node: one operation in the compute graph
// ===========================================================================

struct Node {
    std::string id;                      // unique name, e.g. "q_proj"
    std::string op_type;                 // "MatMul", "Softmax", "Attention"...
    std::vector<std::string> inputs;     // names of input tensors
    std::vector<std::string> outputs;    // names of output tensors

    // Operator-specific parameters (key-value pairs from JSON)
    std::unordered_map<std::string, float> params;
};

// ===========================================================================
// Compute Graph: a DAG of nodes
// ===========================================================================

class ComputeGraph {
public:
    // --- Build ---
    void add_node(const Node& node);
    void add_weight(const std::string& name, const Tensor& tensor);

    // --- Query ---
    const Node* find_node(const std::string& id) const;
    size_t node_count() const { return nodes_.size(); }

    // --- Topological sort (Kahn's algorithm) ---
    // Returns nodes in execution order.
    // Returns empty vector if graph has a cycle (should never happen).
    std::vector<const Node*> topological_sort() const;

    // --- Tensor store ---
    Tensor* get_tensor(const std::string& name);
    void set_tensor(const std::string& name, Tensor t);

    // Debug
    void dump() const;

private:
    std::vector<Node> nodes_;
    std::unordered_map<std::string, size_t> node_index_;  // id -> index
    std::unordered_map<std::string, Tensor> tensors_;
};

#endif // PHASE3_GRAPH_H
