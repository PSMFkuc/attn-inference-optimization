#include "parallel_graph_executor.h"

#include <cstdio>
#include <future>
#include <unordered_map>

namespace phase4 {

namespace {

struct JobResult {
    const Node* node = nullptr;
    Tensor output;
    bool ok = false;
    std::string error;
};

bool gather_inputs(const Node& node,
                   ComputeGraph& graph,
                   std::vector<Tensor>& inputs,
                   std::string& error) {
    inputs.clear();
    inputs.reserve(node.inputs.size());

    for (const std::string& name : node.inputs) {
        Tensor* tensor = graph.get_tensor(name);
        if (!tensor) {
            error = "input tensor '" + name + "' not found for node '" + node.id + "'";
            return false;
        }
        inputs.push_back(*tensor);
    }
    return true;
}

} // namespace

ParallelGraphExecutor::ParallelGraphExecutor(const Backend& backend, bool verbose)
    : backend_(backend), verbose_(verbose) {}

bool ParallelGraphExecutor::execute_serial(const std::vector<const Node*>& sorted_nodes,
                                           ComputeGraph& graph) const {
    if (verbose_) {
        std::printf("[Phase4] Running %zu nodes with backend=%s (serial schedule)\n",
                    sorted_nodes.size(), backend_.name());
    }

    for (const Node* node : sorted_nodes) {
        std::vector<Tensor> inputs;
        std::string error;
        if (!gather_inputs(*node, graph, inputs, error)) {
            std::fprintf(stderr, "[Phase4] %s\n", error.c_str());
            return false;
        }

        Tensor output;
        if (!backend_.execute(*node, inputs, output, error)) {
            std::fprintf(stderr, "[Phase4] node '%s' failed: %s\n",
                         node->id.c_str(), error.c_str());
            return false;
        }
        const std::string output_name = output.name;
        graph.set_tensor(output_name, std::move(output));
    }
    return true;
}

std::vector<std::vector<const Node*>>
ParallelGraphExecutor::build_execution_levels(const std::vector<const Node*>& sorted_nodes) const {
    std::unordered_map<std::string, int> producer_level;
    std::vector<std::vector<const Node*>> levels;

    for (const Node* node : sorted_nodes) {
        int level = 0;
        for (const std::string& input : node->inputs) {
            auto it = producer_level.find(input);
            if (it != producer_level.end()) {
                level = std::max(level, it->second + 1);
            }
        }

        if (static_cast<int>(levels.size()) <= level) {
            levels.resize(level + 1);
        }
        levels[level].push_back(node);

        for (const std::string& output : node->outputs) {
            producer_level[output] = level;
        }
        producer_level[node->id] = level;
    }

    return levels;
}

bool ParallelGraphExecutor::execute_level_parallel(const std::vector<const Node*>& sorted_nodes,
                                                   ComputeGraph& graph) const {
    const auto levels = build_execution_levels(sorted_nodes);
    if (verbose_) {
        std::printf("[Phase4] Running %zu nodes in %zu levels with backend=%s\n",
                    sorted_nodes.size(), levels.size(), backend_.name());
    }

    for (size_t level_idx = 0; level_idx < levels.size(); ++level_idx) {
        const auto& level = levels[level_idx];
        std::vector<std::vector<Tensor>> level_inputs(level.size());

        for (size_t i = 0; i < level.size(); ++i) {
            std::string error;
            if (!gather_inputs(*level[i], graph, level_inputs[i], error)) {
                std::fprintf(stderr, "[Phase4] %s\n", error.c_str());
                return false;
            }
        }

        std::vector<std::future<JobResult>> futures;
        futures.reserve(level.size());
        for (size_t i = 0; i < level.size(); ++i) {
            const Node* node = level[i];
            futures.push_back(std::async(std::launch::async,
                [this, node, inputs = std::move(level_inputs[i])]() mutable {
                    JobResult result;
                    result.node = node;
                    result.ok = backend_.execute(*node, inputs, result.output, result.error);
                    return result;
                }));
        }

        for (auto& future : futures) {
            JobResult result = future.get();
            if (!result.ok) {
                std::fprintf(stderr, "[Phase4] node '%s' failed at level %zu: %s\n",
                             result.node->id.c_str(), level_idx, result.error.c_str());
                return false;
            }
            const std::string output_name = result.output.name;
            graph.set_tensor(output_name, std::move(result.output));
        }

        if (verbose_) {
            std::printf("  level %zu: %zu node(s)\n", level_idx, level.size());
        }
    }

    return true;
}

} // namespace phase4
