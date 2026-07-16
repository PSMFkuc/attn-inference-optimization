#include "graph.h"
#include <queue>
#include <cstdio>

// ===========================================================================
// ComputeGraph implementation
// ===========================================================================

void ComputeGraph::add_node(const Node& node) {
    size_t idx = nodes_.size();
    nodes_.push_back(node);
    node_index_[node.id] = idx;
}

void ComputeGraph::add_weight(const std::string& name, const Tensor& tensor) {
    tensors_[name] = tensor;
}

const Node* ComputeGraph::find_node(const std::string& id) const {
    auto it = node_index_.find(id);
    return (it != node_index_.end()) ? &nodes_[it->second] : nullptr;
}

Tensor* ComputeGraph::get_tensor(const std::string& name) {
    auto it = tensors_.find(name);
    return (it != tensors_.end()) ? &it->second : nullptr;
}

void ComputeGraph::set_tensor(const std::string& name, Tensor t) {
    tensors_[name] = std::move(t);
}

// ===========================================================================
// Topological sort using Kahn's algorithm (BFS based on in-degree)
// ===========================================================================
// Time complexity: O(V + E) where V = nodes, E = edges (input->output deps)
//
// Algorithm:
//   1. Compute in-degree for each node (= number of upstream nodes it depends on)
//   2. Push all nodes with in-degree=0 to queue
//   3. Pop a node, add to result, decrease in-degree of its consumers
//   4. If any consumer's in-degree becomes 0, push it to queue
//   5. Repeat until queue is empty
//
// This works because the graph is a DAG (directed acyclic graph).
// If the graph had a cycle, some nodes would never reach in-degree=0,
// and we'd detect it by comparing result.size() != nodes_.size().

std::vector<const Node*> ComputeGraph::topological_sort() const {
    // Step 1: build name->node mapping for fast lookup
    std::unordered_map<std::string, const Node*> name_to_node;
    for (const auto& n : nodes_) {
        name_to_node[n.id] = &n;
    }

    // Step 2: compute in-degree for each node
    // A node's in-degree = number of its inputs that are produced by other nodes
    std::unordered_map<std::string, int> in_degree;
    for (const auto& n : nodes_) {
        in_degree[n.id] = 0;  // initialize
    }
    for (const auto& n : nodes_) {
        for (const auto& input : n.inputs) {
            // If this input is produced by some node in the graph, it's a dependency
            if (name_to_node.count(input)) {
                in_degree[n.id]++;
            }
        }
    }

    // Step 3: initialize queue with all nodes that have in-degree 0
    std::queue<const Node*> q;
    for (const auto& n : nodes_) {
        if (in_degree[n.id] == 0) {
            q.push(&n);
        }
    }

    // Step 4: BFS
    std::vector<const Node*> sorted;
    while (!q.empty()) {
        const Node* current = q.front();
        q.pop();
        sorted.push_back(current);

        // For each node that depends on current's output, decrease its in-degree
        for (const auto& other : nodes_) {
            for (const auto& input : other.inputs) {
                if (input == current->id) {
                    in_degree[other.id]--;
                    if (in_degree[other.id] == 0) {
                        q.push(&other);
                    }
                }
            }
        }
    }

    // Step 5: cycle detection
    if (sorted.size() != nodes_.size()) {
        fprintf(stderr, "[ERROR] Graph has a cycle! Sorted %zu of %zu nodes.\n",
                sorted.size(), nodes_.size());
        return {};
    }

    return sorted;
}

// ===========================================================================
// Debug dump
// ===========================================================================

void ComputeGraph::dump() const {
    printf("=== ComputeGraph: %zu nodes ===\n", nodes_.size());
    for (const auto& n : nodes_) {
        printf("  [%s] op=%s\n", n.id.c_str(), n.op_type.c_str());
        printf("    inputs:  ");
        for (const auto& in : n.inputs) printf("%s ", in.c_str());
        printf("\n    outputs: ");
        for (const auto& out : n.outputs) printf("%s ", out.c_str());
        printf("\n");
        if (!n.params.empty()) {
            printf("    params:  ");
            for (const auto& [k, v] : n.params) printf("%s=%.4f ", k.c_str(), v);
            printf("\n");
        }
    }
    printf("  Tensors in store: %zu\n", tensors_.size());
}
