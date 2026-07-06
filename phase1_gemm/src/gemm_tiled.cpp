#include "gemm_tiled.h"
#include <algorithm>

// ===========================================================================
// 分块 GEMM：6 层循环（3 层块外 + 3 层块内）
// ===========================================================================
// 为什么是 6 层而不是 3 层？
//   原始 GEMM 是 3 层(i,j,k)。分块后，每个原维度变成"块外+块内"两层。
//   块外循环：遍历所有小块的组合
//   块内循环：在 cache 友好的小块范围内做实际计算
//
// 为什么块内还用 ikj 顺序？
//   块外顺序决定 cache 友好性（让小块塞进 cache）。
//   块内顺序决定寄存器利用（和原 GEMM 一样，ikj 最优）。
//   两层优化叠加。
// ===========================================================================
void gemm_tiled(int M, int N, int K,
                const std::vector<float>& A,
                const std::vector<float>& B,
                std::vector<float>& C,
                int BM, int BN, int BK) {
    C.assign(M * N, 0.0f);

    // 块外三层循环：步进是 block size
    for (int ii = 0; ii < M; ii += BM) {
        for (int jj = 0; jj < N; jj += BN) {
            for (int kk = 0; kk < K; kk += BK) {

                // 边界处理：M/N/K 不一定是 block size 的整数倍。
                // std::min 确保不越界。最后一小块可能比 BM/BN/BK 小。
                int i_end = std::min(ii + BM, M);
                int k_end = std::min(kk + BK, K);
                int j_end = std::min(jj + BN, N);

                // 块内三层循环：在 [ii,i_end) × [kk,k_end) × [jj,j_end) 范围内
                // 此时访问的数据：
                //   A 的 [ii,i_end)×[kk,k_end) 小块 ~ BM*BK*4 字节
                //   B 的 [kk,k_end)×[jj,j_end) 小块 ~ BK*BN*4 字节
                //   C 的 [ii,i_end)×[jj,j_end) 小块 ~ BM*BN*4 字节
                // 总共 12KB（32x32 时），远小于 L1 的 32KB → 内层 cache miss 极少
                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        float a_ik = A[i * K + k];  // 循环不变量外提
                        for (int j = jj; j < j_end; ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

// ===========================================================================
// 思考题（做完实验后回答）：
// 1. 把 BM/BN/BK 改成 16、64、128，分别测，哪个最快？为什么？
// 2. 不对称的 block（如 BM=64,BN=8,BK=256）会怎样？什么时候有用？
//    提示：考虑寄存器分块——让 C 的一个小行块完全驻留在寄存器里。
// 3. 你的 CPU L1/L2/L3 多大？怎么查？（Linux: lscpu / cat /proc/cpuinfo）
// ===========================================================================
