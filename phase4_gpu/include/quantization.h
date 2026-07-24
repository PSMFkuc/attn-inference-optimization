#ifndef PHASE4_QUANTIZATION_H
#define PHASE4_QUANTIZATION_H

#include <cstdint>
#include <vector>

namespace phase4 {

struct QuantizedTensor {
    std::vector<int> shape;
    std::vector<int8_t> values;
    std::vector<float> scales;
    bool per_channel = false;
};

QuantizedTensor quantize_symmetric_per_tensor(const std::vector<float>& values,
                                              const std::vector<int>& shape);

QuantizedTensor quantize_weight_per_output_channel(const std::vector<float>& weights,
                                                   int rows,
                                                   int cols);

void gemm_fp32_int8_weight(int M,
                           int N,
                           int K,
                           const std::vector<float>& A,
                           const QuantizedTensor& B,
                           std::vector<float>& C);

float max_abs_error(const std::vector<float>& a,
                    const std::vector<float>& b);

} // namespace phase4

#endif // PHASE4_QUANTIZATION_H
