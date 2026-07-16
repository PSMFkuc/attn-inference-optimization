#ifndef PHASE3_EXECUTOR_H
#define PHASE3_EXECUTOR_H

#include "graph.h"
#include "memory_pool.h"
#include <functional>
#include <unordered_map>
#include <string>
#include <memory>

// ===========================================================================
// Operator executor: dispatches graph nodes to Phase 2 operator functions
// ===========================================================================

class Executor {
public:
    // Run all nodes in sorted order using naive memory allocation
    // (each output tensor gets its own std::vector<float>)
    bool execute(const std::vector<const Node*>& sorted_nodes,
                 ComputeGraph& graph);

    // Run with memory pool: pre-allocate a buffer, reuse memory for
    // intermediate tensors whose lifetimes don't overlap.
    bool execute_with_pool(const std::vector<const Node*>& sorted_nodes,
                           ComputeGraph& graph,
                           MemoryPool& pool);

    // Get tensor from graph's store (convenience)
    const Tensor* get_output(const ComputeGraph& graph,
                             const std::string& name) const;
};

#endif // PHASE3_EXECUTOR_H
