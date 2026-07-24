#ifndef PHASE4_PARALLEL_GRAPH_EXECUTOR_H
#define PHASE4_PARALLEL_GRAPH_EXECUTOR_H

#include "backend.h"
#include "graph.h"
#include <string>
#include <vector>

namespace phase4 {

class ParallelGraphExecutor {
public:
    explicit ParallelGraphExecutor(const Backend& backend, bool verbose = true);

    bool execute_serial(const std::vector<const Node*>& sorted_nodes,
                        ComputeGraph& graph) const;

    bool execute_level_parallel(const std::vector<const Node*>& sorted_nodes,
                                ComputeGraph& graph) const;

    std::vector<std::vector<const Node*>>
    build_execution_levels(const std::vector<const Node*>& sorted_nodes) const;

private:
    const Backend& backend_;
    bool verbose_;
};

} // namespace phase4

#endif // PHASE4_PARALLEL_GRAPH_EXECUTOR_H
