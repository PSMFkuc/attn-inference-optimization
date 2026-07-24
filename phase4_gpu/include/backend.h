#ifndef PHASE4_BACKEND_H
#define PHASE4_BACKEND_H

#include "graph.h"
#include <string>
#include <vector>

namespace phase4 {

class Backend {
public:
    virtual ~Backend() = default;

    virtual const char* name() const = 0;

    virtual bool execute(const Node& node,
                         const std::vector<Tensor>& inputs,
                         Tensor& output,
                         std::string& error) const = 0;
};

class CpuParallelBackend final : public Backend {
public:
    const char* name() const override { return "cpu-parallel"; }

    bool execute(const Node& node,
                 const std::vector<Tensor>& inputs,
                 Tensor& output,
                 std::string& error) const override;
};

} // namespace phase4

#endif // PHASE4_BACKEND_H
