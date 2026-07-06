#!/usr/bin/env python3
"""
Phase 1 性能对比：NumPy/OpenBLAS vs 你的 C++ 实现

运行方式：
    cd AttnInferenceFramework
    python scripts/bench_numpy.py

输出：和 bench_gemm.exe 同样尺寸的 GFLOPS 表，方便直接对比。
"""

import numpy as np
import time
import os
import sys

# --- 强制单线程（公平对比你的单线程 C++ 实现）---
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 尝试限制 threadpoolctl（如果安装了的话）
try:
    import threadpoolctl
    threadpoolctl.threadpool_limits(limits=1)
except ImportError:
    pass


def compute_gflops(M: int, N: int, K: int, ms: float) -> float:
    """计算 GFLOPS。GEMM 浮点运算量 = 2 * M * N * K"""
    flops = 2.0 * M * N * K
    return flops / (ms / 1000.0) / 1e9


def bench_matmul(sz: int, warmup: int = 1, trials: int = 10) -> tuple[float, float]:
    """
    单次尺寸的 benchmark。
    返回：(best_ms, gflops)
    """
    np.random.seed(42)
    A = np.random.randn(sz, sz).astype(np.float32)
    B = np.random.randn(sz, sz).astype(np.float32)

    # 预热
    for _ in range(warmup):
        _ = A @ B

    # 多次测量取最小值
    best_ms = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        _ = A @ B
        ms = (time.perf_counter() - t0) * 1000.0
        best_ms = min(best_ms, ms)

    gflops = compute_gflops(sz, sz, sz, best_ms)
    return best_ms, gflops


def main():
    print("=" * 70)
    print("  NumPy GEMM Benchmark (Single-Thread)")
    print("=" * 70)

    # 检测 NumPy 用的 BLAS 后端
    try:
        np.show_config(mode="blas")
    except Exception:
        pass

    print(f"\n  NumPy version: {np.__version__}")
    print(f"  dtype: float32")
    print(f"  Threads: 1 (single-thread, fair comparison)")
    print()

    sizes = [64, 128, 256, 512, 1024]

    # 打印表头（格式对齐 C++ bench_gemm 输出）
    print(f"{'size':<8} | {'NumPy(ms)':<10} {'NumPy GFLOPS':<14} | "
          f"{'参考：你的 C++ ikj 预期 GFLOPS':<30}")
    print("-" * 80)

    for sz in sizes:
        ms, gflops = bench_matmul(sz)
        print(f"{sz:<8} | {ms:<10.2f} {gflops:<14.2f} | "
              f"(对比你之前 C++ 结果)")

    print()
    print("=" * 70)
    print("  如何对比")
    print("=" * 70)
    print("  1. 跑 bench_gemm.exe 拿到你的 C++ GFLOPS 表")
    print("  2. 把上表的 NumPy GFLOPS 填到右侧对比")
    print("  3. 算差距: NumPy GFLOPS / C++ GFLOPS")
    print()
    print("  典型差距解读：")
    print("    1-2x   : C++ 实现已接近 BLAS 水平，非常好")
    print("    2-5x   : 正常差距，BLAS 有汇编微内核 + packing")
    print("    > 5x   : 你的实现还有较大优化空间（先看 cache miss）")
    print("=" * 70)


if __name__ == "__main__":
    main()
