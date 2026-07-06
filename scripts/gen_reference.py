"""生成 Phase 1-2 算子的 PyTorch/NumPy 参考答案。

为什么需要这个？
  C++ 实现需要一个"权威参考"来验证正确性。PyTorch/NumPy 是工业标准，
  它们的结果可以作为 ground truth。

用法：
  python gen_reference.py
  会在 ref/ 目录生成 .bin 二进制文件，C++ 测试代码读取对比。
"""

import os
import numpy as np

REF_DIR = os.path.join(os.path.dirname(__file__), "..", "ref")
os.makedirs(REF_DIR, exist_ok=True)

np.random.seed(42)  # 固定种子，可复现


def save_bin(name, arr):
    """保存为二进制文件，C++ 用 ifstream 直接读。"""
    path = os.path.join(REF_DIR, f"{name}.bin")
    arr.astype(np.float32).tofile(path)
    print(f"  saved {name}: shape={arr.shape} -> {path}")


def gen_gemm_reference():
    """Phase 1: GEMM 参考答案。"""
    print("Phase 1: GEMM reference")
    for sz in [64, 128, 256]:
        A = np.random.randn(sz, sz).astype(np.float32)
        B = np.random.randn(sz, sz).astype(np.float32)
        C = A @ B  # NumPy matmul，工业级实现
        save_bin(f"gemm_A_{sz}", A)
        save_bin(f"gemm_B_{sz}", B)
        save_bin(f"gemm_C_{sz}", C)


def gen_softmax_reference():
    """Phase 2: Softmax 参考答案，含数值稳定性测试。"""
    print("Phase 2: Softmax reference")
    # 正常情况
    x = np.random.randn(128).astype(np.float32)
    save_bin("softmax_normal_in", x)
    save_bin("softmax_normal_out", np.exp(x) / np.exp(x).sum())

    # 数值稳定性测试：大数值，naive 实现会溢出
    x_big = (np.random.randn(128) * 100 + 1000).astype(np.float32)  # 均值 1000
    save_bin("softmax_big_in", x_big)
    # 用减最大值的方法（PyTorch 内部就这么做）
    x_shifted = x_big - x_big.max()
    save_bin("softmax_big_out", np.exp(x_shifted) / np.exp(x_shifted).sum())


def gen_layernorm_reference():
    """Phase 2: LayerNorm 参考答案。"""
    print("Phase 2: LayerNorm reference")
    N = 64
    x = np.random.randn(2, N).astype(np.float32)
    gamma = np.random.randn(N).astype(np.float32)
    beta = np.random.randn(N).astype(np.float32)
    eps = 1e-5

    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    out = gamma * (x - mean) / np.sqrt(var + eps) + beta

    save_bin("ln_x", x)
    save_bin("ln_gamma", gamma)
    save_bin("ln_beta", beta)
    save_bin("ln_out", out)


def gen_gelu_reference():
    """Phase 2: GELU 参考答案（精确版 + 近似版）。"""
    print("Phase 2: GELU reference")
    from scipy.special import erf

    x = np.random.randn(256).astype(np.float32)
    # 精确版
    exact = 0.5 * x * (1 + erf(x / np.sqrt(2)))
    # 近似版（tanh）
    c = np.sqrt(2 / np.pi)
    approx = 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x**3)))

    save_bin("gelu_in", x)
    save_bin("gelu_exact", exact)
    save_bin("gelu_approx", approx)


def gen_attention_reference():
    """Phase 2: 完整 Scaled Dot-Product Attention 参考答案。"""
    print("Phase 2: Attention reference")
    seq_len, d_k = 32, 64
    Q = np.random.randn(seq_len, d_k).astype(np.float32)
    K = np.random.randn(seq_len, d_k).astype(np.float32)
    V = np.random.randn(seq_len, d_k).astype(np.float32)

    # scores = Q @ K^T / sqrt(d_k)
    scores = (Q @ K.T) / np.sqrt(d_k)
    # softmax 每行
    scores_shifted = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores_shifted)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    # out = weights @ V
    out = weights @ V

    save_bin("attn_Q", Q)
    save_bin("attn_K", K)
    save_bin("attn_V", V)
    save_bin("attn_out", out)


if __name__ == "__main__":
    print(f"Generating reference data in {REF_DIR}\n")
    gen_gemm_reference()
    print()
    gen_softmax_reference()
    print()
    gen_layernorm_reference()
    print()
    gen_gelu_reference()
    print()
    gen_attention_reference()
    print("\nDone. C++ 测试代码可用 ifstream 读取 .bin 文件对比。")
