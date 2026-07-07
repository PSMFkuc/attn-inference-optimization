import torch
import numpy as np
from pathlib import Path

"""
生成 PyTorch 参考数据，供 C++ 测试对比使用。
运行：python gen_phase2_ref.py
输出放在 data/ 目录下
"""

Path("data").mkdir(exist_ok=True)

np.random.seed(42)

# =============================================================
# 1. Softmax 测试数据
# =============================================================
print("=== Generating Softmax references ===")

# 基本测试
x_basic = np.array([2.0, 1.0, 0.5], dtype=np.float32)
x_t = torch.tensor(x_basic)
ref = torch.softmax(x_t, dim=0).numpy()
print(f"  Input: {x_basic}")
print(f"  Ref:   {ref}, sum={ref.sum():.6f}")

# 数值稳定性测试（大数）
x_large = np.array([1000.0, 1000.0, 1000.0], dtype=np.float32)
ref_large = torch.softmax(torch.tensor(x_large), dim=0).numpy()
print(f"  Large input ref: {ref_large}")

# 全负数测试
x_neg = np.array([-5.0, -3.0, -1.0], dtype=np.float32)
ref_neg = torch.softmax(torch.tensor(x_neg), dim=0).numpy()
print(f"  Negative input ref: {ref_neg}")

# =============================================================
# 2. LayerNorm 测试数据
# =============================================================
print("\n=== Generating LayerNorm references ===")

x_ln = np.array([1.0, 3.0, 2.0, 6.0], dtype=np.float32)
gamma = np.ones(4, dtype=np.float32)
beta = np.zeros(4, dtype=np.float32)

x_t = torch.tensor(x_ln)
gamma_t = torch.tensor(gamma)
beta_t = torch.tensor(beta)

# PyTorch layer_norm 需要 normalized_shape 参数
ref_ln = torch.nn.functional.layer_norm(
    x_t, (4,), gamma_t, beta_t, eps=1e-5
).numpy()

print(f"  Input:  {x_ln}")
print(f"  Output: {ref_ln}")
mean = ref_ln.mean()
var = ref_ln.var()
print(f"  Mean={mean:.6f} (should be ~0), Var={var:.6f} (should be ~1)")

# Gamma/Beta 非恒等测试
gamma2 = np.array([2.0, 1.0, 1.0, 1.0], dtype=np.float32)
beta2 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
x_ln2 = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
ref_ln2 = torch.nn.functional.layer_norm(
    torch.tensor(x_ln2), (4,),
    torch.tensor(gamma2), torch.tensor(beta2), eps=1e-5
).numpy()
print(f"  With gamma/beta: {ref_ln2}")

# =============================================================
# 3. GELU 测试数据
# =============================================================
print("\n=== Generating GELU references ===")

# 采样点
x_gelu = np.array([-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
                  dtype=np.float32)
ref_gelu = torch.nn.functional.gelu(torch.tensor(x_gelu)).numpy()

print("  x         GELU(x)")
for xi, ri in zip(x_gelu, ref_gelu):
    print(f"  {xi:8.1f}  {ri:12.6f}")

# 密集采样（用于测试单调性和近似精度）
x_dense = np.arange(-5.0, 5.01, 0.1, dtype=np.float32)
ref_dense = torch.nn.functional.gelu(torch.tensor(x_dense)).numpy()

# =============================================================
# 4. Attention 测试数据
# =============================================================
print("\n=== Generating Attention references ===")

seq_len, d_k, d_v = 3, 4, 4
torch.manual_seed(42)
Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)
V = torch.randn(seq_len, d_v)

# 手动实现 Attention 作为参考
scale = d_k ** 0.5
scores = Q @ K.T / scale
weights = torch.softmax(scores, dim=1)
ref_attn = (weights @ V).numpy()

print(f"  Q:\n{Q.numpy()}")
print(f"  K:\n{K.numpy()}")
print(f"  V:\n{V.numpy()}")
print(f"  Scores:\n{scores.numpy()}")
print(f"  Weights:\n{weights.numpy()}")
print(f"  Weights row sums: {weights.sum(dim=1).numpy()}")
print(f"  Output:\n{ref_attn}")

# =============================================================
# 保存二进制文件
# =============================================================
print("\n=== Saving binary files ===")

# Softmax
x_basic.tofile("data/softmax_basic_in.bin")
ref.tofile("data/softmax_basic_ref.bin")

x_large.tofile("data/softmax_large_in.bin")
ref_large.tofile("data/softmax_large_ref.bin")

x_neg.tofile("data/softmax_neg_in.bin")
ref_neg.tofile("data/softmax_neg_ref.bin")

# LayerNorm
x_ln.tofile("data/layernorm_in.bin")
ref_ln.tofile("data/layernorm_ref.bin")

# GELU
x_gelu.tofile("data/gelu_in.bin")
ref_gelu.tofile("data/gelu_ref.bin")

# Attention
Q.numpy().astype(np.float32).tofile("data/attn_Q.bin")
K.numpy().astype(np.float32).tofile("data/attn_K.bin")
V.numpy().astype(np.float32).tofile("data/attn_V.bin")
ref_attn.astype(np.float32).tofile("data/attn_ref.bin")

# 保存元数据（形状信息）
with open("data/attn_meta.txt", "w") as f:
    f.write(f"seq_len={seq_len}\n")
    f.write(f"d_k={d_k}\n")
    f.write(f"d_v={d_v}\n")

print("Done! All files saved to data/")
