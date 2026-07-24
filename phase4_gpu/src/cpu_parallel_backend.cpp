#include "backend.h"
#include "gemm_parallel.h"
#include "operators.h"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace phase4 {

static bool require_input_count(const Node& node,
                                const std::vector<Tensor>& inputs,
                                size_t expected,
                                std::string& error) {
    if (inputs.size() < expected) {
        error = "node '" + node.id + "' expects at least " +
                std::to_string(expected) + " input tensors";
        return false;
    }
    return true;
}

static bool execute_matmul(const Node& node,
                           const std::vector<Tensor>& inputs,
                           Tensor& output,
                           std::string& error) {
    if (!require_input_count(node, inputs, 2, error)) return false;

    const Tensor& A = inputs[0];
    const Tensor& B = inputs[1];
    if (A.shape.size() < 2 || B.shape.size() < 2) {
        error = "MatMul expects rank-2 tensors";
        return false;
    }

    const int M = A.shape[0];
    const int K_A = A.shape[1];
    const int K_B = B.shape[0];
    const int N = B.shape[1];
    if (K_A != K_B) {
        error = "MatMul dimension mismatch";
        return false;
    }

    output.name = node.outputs.empty() ? node.id : node.outputs[0];
    output.shape = {M, N};
    gemm_parallel_tiled(M, N, K_A, A.data, B.data, output.data);
    return true;
}

static bool execute_softmax(const Node& node,
                            const std::vector<Tensor>& inputs,
                            Tensor& output,
                            std::string& error) {
    if (!require_input_count(node, inputs, 1, error)) return false;
    const Tensor& in = inputs[0];

    output.name = node.outputs.empty() ? node.id : node.outputs[0];
    output.shape = in.shape;
    output.data = in.data;

    int rows = 1;
    int cols = static_cast<int>(output.data.size());
    if (output.shape.size() >= 2) {
        rows = output.shape[0];
        cols = output.shape[1];
    }

#pragma omp parallel for schedule(static) if(rows > 1)
    for (int r = 0; r < rows; ++r) {
        softmax(&output.data[r * cols], cols);
    }
    return true;
}

static bool execute_layernorm(const Node& node,
                              const std::vector<Tensor>& inputs,
                              Tensor& output,
                              std::string& error) {
    if (!require_input_count(node, inputs, 1, error)) return false;
    const Tensor& in = inputs[0];
    if (in.shape.empty()) {
        error = "LayerNorm expects a non-empty tensor shape";
        return false;
    }

    const int cols = in.shape.back();
    const int rows = static_cast<int>(in.data.size()) / cols;
    float eps = 1e-5f;
    auto it = node.params.find("eps");
    if (it != node.params.end()) eps = it->second;

    output.name = node.outputs.empty() ? node.id : node.outputs[0];
    output.shape = in.shape;
    output.data.resize(in.data.size());

    std::vector<float> gamma(cols, 1.0f);
    std::vector<float> beta(cols, 0.0f);

#pragma omp parallel for schedule(static) if(rows > 1)
    for (int r = 0; r < rows; ++r) {
        layernorm(&in.data[r * cols], cols, eps,
                  gamma.data(), beta.data(),
                  &output.data[r * cols]);
    }
    return true;
}

static bool execute_gelu(const Node& node,
                         const std::vector<Tensor>& inputs,
                         Tensor& output,
                         std::string& error) {
    if (!require_input_count(node, inputs, 1, error)) return false;
    const Tensor& in = inputs[0];

    output.name = node.outputs.empty() ? node.id : node.outputs[0];
    output.shape = in.shape;
    output.data.resize(in.data.size());

    const int n = static_cast<int>(in.data.size());
#pragma omp parallel for schedule(static) if(n > 2048)
    for (int i = 0; i < n; ++i) {
        output.data[i] = gelu_approx(in.data[i]);
    }
    return true;
}

static bool execute_attention(const Node& node,
                              const std::vector<Tensor>& inputs,
                              Tensor& output,
                              std::string& error) {
    if (!require_input_count(node, inputs, 3, error)) return false;
    const Tensor& Q = inputs[0];
    const Tensor& K = inputs[1];
    const Tensor& V = inputs[2];
    if (Q.shape.size() < 2 || K.shape.size() < 2 || V.shape.size() < 2) {
        error = "Attention expects rank-2 Q/K/V tensors";
        return false;
    }

    const int seq_len = Q.shape[0];
    const int d_k = Q.shape[1];
    const int d_v = V.shape[1];
    if (K.shape[0] != seq_len || K.shape[1] != d_k || V.shape[0] != seq_len) {
        error = "Attention dimension mismatch";
        return false;
    }

    std::vector<float> K_T(d_k * seq_len);
#pragma omp parallel for collapse(2) schedule(static) if(seq_len * d_k > 4096)
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_k; ++j) {
            K_T[j * seq_len + i] = K.data[i * d_k + j];
        }
    }

    std::vector<float> scores;
    gemm_parallel_tiled(seq_len, seq_len, d_k, Q.data, K_T, scores);

    const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
#pragma omp parallel for schedule(static) if(seq_len * seq_len > 4096)
    for (int i = 0; i < seq_len * seq_len; ++i) {
        scores[i] *= scale;
    }

#pragma omp parallel for schedule(static) if(seq_len > 1)
    for (int r = 0; r < seq_len; ++r) {
        softmax(&scores[r * seq_len], seq_len);
    }

    output.name = node.outputs.empty() ? node.id : node.outputs[0];
    output.shape = {seq_len, d_v};
    gemm_parallel_tiled(seq_len, d_v, seq_len, scores, V.data, output.data);
    return true;
}

bool CpuParallelBackend::execute(const Node& node,
                                 const std::vector<Tensor>& inputs,
                                 Tensor& output,
                                 std::string& error) const {
    if (node.op_type == "MatMul") return execute_matmul(node, inputs, output, error);
    if (node.op_type == "Softmax") return execute_softmax(node, inputs, output, error);
    if (node.op_type == "LayerNorm") return execute_layernorm(node, inputs, output, error);
    if (node.op_type == "GELU") return execute_gelu(node, inputs, output, error);
    if (node.op_type == "Attention") return execute_attention(node, inputs, output, error);

    error = "unknown op type '" + node.op_type + "'";
    return false;
}

} // namespace phase4
