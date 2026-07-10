import numpy as np
from pathlib import Path

"""
Generate reference data using pure NumPy (no PyTorch dependency).
Run: python gen_phase2_ref.py
Output: data/ directory with binary files
"""

Path("data").mkdir(exist_ok=True)
np.random.seed(42)

# ============================================================
# 1. Softmax
# ============================================================
def softmax_np(x):
    x_max = np.max(x)
    e = np.exp(x - x_max)
    return e / np.sum(e)

print("=== Generating Softmax references ===")

x_basic = np.array([2.0, 1.0, 0.5], dtype=np.float32)
ref_basic = softmax_np(x_basic)
print(f"  Basic: {x_basic} -> {ref_basic}, sum={ref_basic.sum():.6f}")

x_large = np.array([1000.0, 1000.0, 1000.0], dtype=np.float32)
ref_large = softmax_np(x_large)
print(f"  Large: {x_large} -> {ref_large}")

x_neg = np.array([-5.0, -3.0, -1.0], dtype=np.float32)
ref_neg = softmax_np(x_neg)
print(f"  Negative: {x_neg} -> {ref_neg}")

# ============================================================
# 2. LayerNorm
# ============================================================
def layernorm_np(x, gamma, beta, eps=1e-5):
    mean = np.mean(x)
    var = np.var(x)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

print("\n=== Generating LayerNorm references ===")

x_ln = np.array([1.0, 3.0, 2.0, 6.0], dtype=np.float32)
gamma = np.ones(4, dtype=np.float32)
beta = np.zeros(4, dtype=np.float32)
ref_ln = layernorm_np(x_ln, gamma, beta)
print(f"  Input: {x_ln}")
print(f"  Output: {ref_ln}")
print(f"  Mean={ref_ln.mean():.6f} (expect ~0), Var={ref_ln.var():.6f} (expect ~1)")

gamma2 = np.array([2.0, 1.0, 1.0, 1.0], dtype=np.float32)
beta2 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
x_ln2 = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
ref_ln2 = layernorm_np(x_ln2, gamma2, beta2)
print(f"  With gamma/beta: {ref_ln2}")

# ============================================================
# 3. GELU
# ============================================================
def gelu_np(x):
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

print("\n=== Generating GELU references ===")

x_gelu = np.array([-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
                  dtype=np.float32)
ref_gelu = gelu_np(x_gelu)
print("  x         GELU(x)")
for xi, ri in zip(x_gelu, ref_gelu):
    print(f"  {xi:8.1f}  {ri:12.6f}")

# ============================================================
# 4. Attention
# ============================================================
print("\n=== Generating Attention references ===")

seq_len, d_k, d_v = 3, 4, 4
np.random.seed(42)
Q = np.random.randn(seq_len, d_k).astype(np.float32)
K = np.random.randn(seq_len, d_k).astype(np.float32)
V = np.random.randn(seq_len, d_v).astype(np.float32)

# Manual attention
scale = np.sqrt(d_k)
scores = Q @ K.T / scale

# Softmax per row
weights = np.zeros_like(scores)
for r in range(seq_len):
    weights[r] = softmax_np(scores[r])

ref_attn = weights @ V

print(f"  Q:\n{Q}")
print(f"  K:\n{K}")
print(f"  V:\n{V}")
print(f"  Weights row sums: {weights.sum(axis=1)}")
print(f"  Output:\n{ref_attn}")

# ============================================================
# Save binary files
# ============================================================
print("\n=== Saving binary files ===")

x_basic.tofile("data/softmax_basic_in.bin")
ref_basic.tofile("data/softmax_basic_ref.bin")
x_large.tofile("data/softmax_large_in.bin")
ref_large.tofile("data/softmax_large_ref.bin")
x_neg.tofile("data/softmax_neg_in.bin")
ref_neg.tofile("data/softmax_neg_ref.bin")

x_ln.tofile("data/layernorm_in.bin")
ref_ln.tofile("data/layernorm_ref.bin")

x_gelu.tofile("data/gelu_in.bin")
ref_gelu.tofile("data/gelu_ref.bin")

Q.tofile("data/attn_Q.bin")
K.tofile("data/attn_K.bin")
V.tofile("data/attn_V.bin")
ref_attn.astype(np.float32).tofile("data/attn_ref.bin")

with open("data/attn_meta.txt", "w") as f:
    f.write(f"seq_len={seq_len}\n")
    f.write(f"d_k={d_k}\n")
    f.write(f"d_v={d_v}\n")

print("Done! All files saved to data/")
