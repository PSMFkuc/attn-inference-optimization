#ifndef GEMM_NAIVE_H
#define GEMM_NAIVE_H

#include <vector>

// 函数接口：C = A × B
// A: M×K (行主序，即 A[i][k] 在 A[i*K + k])
// B: K×N
// C: M×N
//
// 为什么用 std::vector<float> 而不是二维 vector<vector<float>>？
//   1. vector<vector> 每行单独分配内存，行间不连续，cache 友好性差。
//   2. vector<vector> 分配多次，慢。
//   3. 一维数组 + 索引计算 (i*N + j) 是高性能代码的标准做法，内存连续、分配一次。
//
// 为什么参数用 const& 而不是值传递？
//   避免拷贝。矩阵可能几十 MB，值传递会深拷贝，浪费且慢。

// 版本 1：最朴素的 ijk 三重循环（baseline）
void gemm_naive_ijk(int M, int N, int K,
                    const std::vector<float>& A,
                    const std::vector<float>& B,
                    std::vector<float>& C);

// 版本 2：ikj 顺序，A/B 都按行访问，通常比 ijk 快 2-3 倍
void gemm_naive_ikj(int M, int N, int K,
                    const std::vector<float>& A,
                    const std::vector<float>& B,
                    std::vector<float>& C);

#endif // GEMM_NAIVE_H
