#ifndef GEMM_TILED_H
#define GEMM_TILED_H

#include <vector>

// 分块（Tiling/Blocking）优化的 GEMM。
// 核心思想：把矩阵切成小块，让一小块的工作集塞进 L1/L2 cache，
// 内层循环的 cache miss 几乎为零。
//
// 参数 BM/BN/BK 是 block size，影响 cache 利用率：
//   - 太小：cache 没塞满，循环开销占比大
//   - 太大：溢出 cache，反而 miss
//   - 最优值靠实验测（不同 CPU 不同）
//
// 经验起点：BM=BN=BK=32，3*32*32*4=12KB，塞进 L1(32KB) 留余量。

void gemm_tiled(int M, int N, int K,
                const std::vector<float>& A,
                const std::vector<float>& B,
                std::vector<float>& C,
                int BM = 32, int BN = 32, int BK = 32);

#endif // GEMM_TILED_H
