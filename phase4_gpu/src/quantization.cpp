#include "quantization.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace phase4 {

namespace {

int8_t clamp_to_i8(int value) {
    value = std::max(-127, std::min(127, value));
    return static_cast<int8_t>(value);
}

float safe_scale(float max_abs) {
    return max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
}

} // namespace

QuantizedTensor quantize_symmetric_per_tensor(const std::vector<float>& values,
                                              const std::vector<int>& shape) {
    float max_abs = 0.0f;
    for (float v : values) {
        max_abs = std::max(max_abs, std::fabs(v));
    }

    QuantizedTensor q;
    q.shape = shape;
    q.values.resize(values.size());
    q.scales.clear();
    q.scales.push_back(safe_scale(max_abs));
    q.per_channel = false;

    const float inv_scale = 1.0f / q.scales[0];
    for (size_t i = 0; i < values.size(); ++i) {
        q.values[i] = clamp_to_i8(static_cast<int>(std::lround(values[i] * inv_scale)));
    }
    return q;
}

QuantizedTensor quantize_weight_per_output_channel(const std::vector<float>& weights,
                                                   int rows,
                                                   int cols) {
    QuantizedTensor q;
    q.shape = {rows, cols};
    q.values.resize(weights.size());
    q.scales.resize(cols, 1.0f);
    q.per_channel = true;

    for (int col = 0; col < cols; ++col) {
        float max_abs = 0.0f;
        for (int row = 0; row < rows; ++row) {
            max_abs = std::max(max_abs, std::fabs(weights[row * cols + col]));
        }

        q.scales[col] = safe_scale(max_abs);
        const float inv_scale = 1.0f / q.scales[col];
        for (int row = 0; row < rows; ++row) {
            const int idx = row * cols + col;
            q.values[idx] = clamp_to_i8(static_cast<int>(std::lround(weights[idx] * inv_scale)));
        }
    }

    return q;
}

void gemm_fp32_int8_weight(int M,
                           int N,
                           int K,
                           const std::vector<float>& A,
                           const QuantizedTensor& B,
                           std::vector<float>& C) {
    C.assign(static_cast<size_t>(M) * N, 0.0f);

#pragma omp parallel for schedule(static) if(M * N > 4096)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * static_cast<float>(B.values[k * N + j]);
            }
            const float scale = B.per_channel ? B.scales[j] : B.scales[0];
            C[i * N + j] = acc * scale;
        }
    }
}

float max_abs_error(const std::vector<float>& a,
                    const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<float>::infinity();
    }

    float max_err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_err = std::max(max_err, std::fabs(a[i] - b[i]));
    }
    return max_err;
}

} // namespace phase4
