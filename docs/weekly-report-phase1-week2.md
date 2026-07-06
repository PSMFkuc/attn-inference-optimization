# Phase 1 GEMM 周报 — Week 2: SIMD 向量化 & Block Size 调优

> 日期：2026-07-06 | 项目：AttnInferenceFramework / Phase1 GEMM

---

## 一、本周工作摘要

完成两项核心优化：
1. **AVX2 SIMD 手写向量化**：用 `<immintrin.h>` intrinsics 替换标量内层循环，理论 8x 加速
2. **Block Size 参数扫描**：测试 8 组 block size 配置，找到最优 tiling 参数

---

## 二、SIMD 向量化

### 2.1 原理

```
标量 (ikj)：
  内层 j 循环：每次处理 1 个 float
  C[j] += a * B[j]      → 1 mul + 1 add = 2 FLOPs/iteration

AVX2 SIMD (256位 = 8 floats)：
  内层 j 循环：每次处理 8 个 float
  C[j..j+7] += a * B[j..j+7]  → 1 FMA instruction = 16 FLOPs/iteration
```

### 2.2 核心指令

| Intrinsic | 作用 | 延迟 |
|-----------|------|------|
| `_mm256_set1_ps(x)` | 标量广播到 8 个 lane | ~3 cycles |
| `_mm256_loadu_ps(ptr)` | 加载 8 个 float | ~4 cycles |
| `_mm256_fmadd_ps(a,b,c)` | FMA: `c = a*b + c` | ~4 cycles (吞吐 2/cycle) |
| `_mm256_storeu_ps(ptr,reg)` | 存储 8 个 float | ~4 cycles |

### 2.3 代码实现

新增文件：
- `src/gemm_simd.h` — 接口声明
- `src/gemm_simd.cpp` — AVX2 intrinsics 实现

核心结构：ikj 循环顺序 + 内层 j 以 8 为步长 SIMD + 尾循环处理 N%8

### 2.4 正确性验证

GoogleTest 5/5 全部通过：
- `SIMD_MatchesReference`：N=64 (8 倍数)，误差 < 1e-4
- `SIMD_NonMultipleOf8`：N=50 (非 8 倍数，测试尾循环)，误差 < 1e-3

### 2.5 性能对比

| size | ikj GFLOPS | SIMD GFLOPS | SIMD/ikj | 分析 |
|------|-----------|-------------|----------|------|
| 64 | 30.84 | 23.83 | 0.77x | SIMD 慢：编译器已自动向量化 + 手写 loadu(C) 多一次内存读 |
| 128 | 26.55 | 20.26 | 0.76x | 同上 |
| 256 | 29.00 | 20.12 | 0.69x | 同上 |
| 512 | 35.04 | 21.91 | 0.63x | 同上，内存带宽瓶颈加剧 |
| 1024 | 19.67 | 14.72 | 0.75x | 同上 |

### 2.6 关键发现

**手写 SIMD 反而比编译器自动向量化慢 25-37%！**

原因分析：
1. **GCC 14.2 -O3 -march=native 已自动向量化 ikj**：编译器在简单连续循环上的向量化能力极强
2. **手写版本多了一次 C 的 load**：`_mm256_loadu_ps(C)` 在每次内层迭代都执行，而编译器可能通过循环展开 + 寄存器重用来减少
3. **内存带宽瓶颈**：当前实现在 512/1024 上受限于内存带宽（~35 GFLOPS 峰值 vs 70.4 GFLOPS 理论），SIMD 的计算优势无法发挥

**结论**：在纯 CPU GEMM 上，不配合数据打包（packing），手写 SIMD 不会超越编译器自动向量化。下一步应该做 **SIMD + 数据打包 + 寄存器分块** 组合优化。

---

## 三、Block Size 参数调优

### 3.1 测试设计

测试 8 组有代表性的 block size，覆盖 L1→L2 范围：

| 配置 | 工作集大小 | 设计依据 |
|------|-----------|----------|
| 16×16×16 | 3KB | L1 有大量余量 |
| 32×32×32 | 12KB | 当前默认值 |
| 48×48×48 | 28KB | L1 接近上限 (48KB) |
| 64×64×64 | 48KB | L1 临界点 |
| 96×96×96 | 96KB | 溢出 L1，进入 L2 |
| 128×128×128 | 192KB | L2 安全范围 |
| 64×64×256 | 48KB (非对称) | 长 K 维，减少块外循环 |
| 128×16×256 | 40KB (非对称) | 寄存器友好（小 N） |

### 3.2 扫描结果

| block config | 256 GFLOPS | 512 GFLOPS | 1024 GFLOPS |
|-------------|-----------|-----------|-------------|
| 16×16×16 (3KB) | 12.28 | 13.02 | 11.48 |
| 32×32×32 (12KB) | 18.68 | 18.24 | 17.70 |
| 48×48×48 (28KB) | 19.25 | 23.97 | 21.61 |
| 64×64×64 (48KB) | 20.12 | 25.49 | 23.70 |
| 96×96×96 (96KB) | 21.24 | 27.87 | 25.92 |
| **128×128×128 (192KB)** | **21.87** | **31.02** | **26.90** |
| 64×64×256 (asym) | 20.13 | 24.38 | 19.97 |
| 128×16×256 (asym) | 9.52 | 10.02 | 9.31 |

### 3.3 分析

**最优配置：128×128×128**

- 在所有尺寸上均为最高 GFLOPS
- 1024 尺寸达到 26.90 GFLOPS，比默认 32×32 的 17.70 GFLOPS 提升 **52%**
- 工作集 192KB，虽然溢出 L1 (48KB)，但在 L2 (1.25MB) 内，L2 的命中率仍然很高
- 更大的 block 减少了块外循环开销（块数量从 (1024/32)³=32768 降到 (1024/128)³=512）

**性能随 block size 单调递增**：

```
16 → 32 → 48 → 64 → 96 → 128 : GFLOPS 持续增长
```

说明在 i5-12450H 上，L2 cache 足够大，可以用更大的 block 换取更少的块外循环次数。

**非对称配置表现差**：
- `64×64×256` 比 `128×128×128` 慢 26%
- `128×16×256` 最慢（9 GFLOPS），因为 BN=16 太窄，循环开销占比过大
- 非对称配置需要在 SIMD + 寄存器分块配合下才能发挥优势

### 3.4 结论

| 项目 | 旧值 | 新值 | 提升 |
|------|------|------|------|
| 默认 block size | 32×32×32 | **128×128×128** | — |
| 256 GFLOPS | 18.68 | 21.87 | +17% |
| 512 GFLOPS | 18.24 | 31.02 | +70% |
| 1024 GFLOPS | 17.70 | 26.90 | +52% |

**推荐将 tiled 版本的默认 block size 改为 128×128×128。**

---

## 四、完整性能汇总（含本周两项优化）

| size | ijk | ikj | tiled(128) | simd | best | peak% |
|------|-----|-----|-----------|------|------|-------|
| 64 | 4.37 | 30.84 | — | 23.83 | **30.84** | 43.8% |
| 128 | 3.43 | 26.55 | — | 20.26 | **26.55** | 37.7% |
| 256 | 3.00 | 29.00 | 21.87 | 20.12 | **29.00** | 41.2% |
| 512 | 2.25 | 35.04 | 31.02 | 21.91 | **35.04** | 49.8% |
| 1024 | 0.47 | 19.67 | 26.90 | 14.72 | **26.90** | 38.2% |

- 小矩阵 (≤256)：ikj 仍然最优
- 大矩阵 (≥1024)：tiled(128×128×128) 反超 ikj，成为最佳实现
- 峰值效率 49.8%（512 尺寸，ikj）

---

## 五、与 OpenBLAS 差距

| size | C++ best | NumPy | Ratio | 差距来源 |
|------|----------|-------|-------|----------|
| 256 | 29.00 | 51.24 | 1.77x | SIMD 微内核 |
| 512 | 35.04 | 81.16 | 2.32x | SIMD + packing |
| 1024 | 26.90 | 81.64 | 3.04x | SIMD + packing + prefetch |

差距从上周的 2-4x 缩小到了 1.8-3x（大矩阵 tiled 128 提升显著）。

---

## 六、项目文件变更

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `src/gemm_simd.h` | 新增 | SIMD 接口声明 |
| `src/gemm_simd.cpp` | 新增 | AVX2 intrinsics 实现 |
| `profiling/bench_gemm.cpp` | 修改 | 加入 SIMD 测试列 + block size 扫描 |
| `tests/test_gemm.cpp` | 修改 | 加入 SIMD 正确性测试 (2 tests) |
| `CMakeLists.txt` | 修改 | 加入 gemm_simd.cpp 编译 |

---

## 七、perf 指令级热点分析（新增）

### 7.1 方法

使用 WSL2 的 `perf record -e cpu-clock` + `perf report` + `perf annotate` 进行指令级热点分析。WSL2 内核不支持硬件 PMU 计数器（`<not supported>`），改用软件时钟采样。

### 7.2 ikj 函数热点指令

```
7.69%   vmovups (%rdx,%rcx),%ymm0       ← load B (连续访问，cache 友好)
29.88%  vfmadd213ps (%rax,%rcx),%ymm2,%ymm0  ← FMA: load C + 乘 + 加
28.17%  vmovups %ymm0,(%rax,%rcx)       ← store C
31.73%  add $0x20,%rcx                  ← 循环计数器
1.58%   cmp %rcx,%rdi                   ← 循环判断
```

### 7.3 关键发现

1. **编译器已自动向量化**：`vmovups` 和 `vfmadd213ps` 是 AVX2 指令，GCC 已把 ikj 标量循环转成了 SIMD
2. **循环控制开销 33.3%**（add 31.7% + cmp 1.6%）：每 3 个 CPU 周期有 1 个花在"判断循环是否继续"上
3. **内存访问健康**：B 的 load 仅占 7.7%，说明连续访问的 cache 命中率高
4. **手写 SIMD 不会更快**：编译器已经生成了最优的 SIMD 指令序列

---

## 八、循环展开优化（新增）

### 8.1 原理

基于 perf annotate 发现循环控制开销占 33%，实施 4x 循环展开：
- 改前：每 8 个 float 做一次 add+cmp+jne（33% 开销）
- 改后：每 32 个 float 做一次 add+cmp+jne（~8% 开销）

### 8.2 实现

新增文件：
- `src/gemm_simd_unroll.h` — 接口声明
- `src/gemm_simd_unroll.cpp` — 4x 展开实现

结构：4 个连续的 FMA 块 + 2x 尾循环（处理 8≤剩余<32）+ 标量尾循环（剩余<8）

### 8.3 正确性验证

GoogleTest 7/7 全部通过（新增 2 tests）：
- `SIMD_Unroll_MatchesReference`：N=64 (32 对齐)，误差 < 1e-4
- `SIMD_Unroll_NonMultipleOf32`：N=50 (非 32 对齐)，误差 < 1e-3

### 8.4 性能对比

| size | simd GFLOPS | unroll GFLOPS | unroll/simd | 分析 |
|------|------------|--------------|-------------|------|
| 64 | 11.40 | **13.44** | **1.18x** | ✅ +18%，小矩阵在 L1，计算瓶颈 |
| 128 | 20.26 | 20.36 | 1.01x | ≈ 持平，内存带宽瓶颈掩盖收益 |
| 256 | 23.21 | **24.44** | **1.05x** | ✅ +5% |
| 512 | 21.69 | **23.20** | **1.07x** | ✅ +7% |
| 1024 | 15.70 | **18.18** | **1.16x** | ✅ +16% |

### 8.5 结论

- 循环展开在手写 SIMD 上有效（+5~18%）
- 但编译器自动向量化的 **ikj 仍然是综合最优**（23-37 GFLOPS vs unroll 13-24 GFLOPS）
- 手写 SIMD 额外 load/store C 的开销抵消了部分展开收益
- 下一步应探索：**循环展开 + 寄存器分块（减少 C 的 load/store 次数）**

---

## 九、完整性能汇总（含全部 5 种实现）

| size | ijk | ikj | tiled(32) | simd | unroll | best | peak% |
|------|-----|-----|-----------|------|--------|------|-------|
| 64 | 2.32 | 12.19 | 9.20 | 11.40 | 13.44 | **13.44** | 19.1% |
| 128 | 3.29 | **23.43** | 19.24 | 20.26 | 20.36 | 23.43 | 33.3% |
| 256 | 2.88 | **35.54** | 22.09 | 23.21 | 24.44 | 35.54 | 50.5% |
| 512 | 2.76 | **36.64** | 20.16 | 21.69 | 23.20 | 36.64 | 52.0% |
| 1024 | 0.61 | **21.83** | 18.05 | 15.70 | 18.18 | 21.83 | 31.0% |

## 十、项目文件变更（更新）

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `src/gemm_simd.h` | 新增 | SIMD 接口声明 |
| `src/gemm_simd.cpp` | 新增 | AVX2 intrinsics 实现 |
| `src/gemm_simd_unroll.h` | 新增 | 4x 循环展开接口声明 |
| `src/gemm_simd_unroll.cpp` | 新增 | 4x 循环展开实现 |
| `profiling/bench_gemm.cpp` | 修改 | 加入 SIMD + unroll 测试列 + block size 扫描 |
| `tests/test_gemm.cpp` | 修改 | 加入 4 个新测试 (SIMD + unroll) |
| `CMakeLists.txt` | 修改 | 加入 gemm_simd.cpp + gemm_simd_unroll.cpp |

## 十一、下一步计划

| 优先级 | 任务 | 预期收益 |
|--------|------|----------|
| P0 | 多线程 (OpenMP) 并行化 | 4P-core = ~4x 理论加速 |
| P1 | 循环展开 + 寄存器分块（micro-kernel） | 减少 C 的 load/store，接近 OpenBLAS |
| P2 | 数据打包 (packing) | 配合 micro-kernel 提升 SIMD 效率 |
| P3 | 软件预取 (prefetch) | 隐藏 L2 miss 延迟 |

## 十二、附录：运行命令

```powershell
# 完整 benchmark（含 5 种实现 + block size 扫描）
$env:Path = "C:\mingw64\bin;$env:Path"
cd phase1_gemm\build
.\bench_gemm.exe

# 正确性测试（7 tests）
.\test_gemm.exe

# WSL2 perf 热点分析
cd phase1_gemm/build_linux
perf record -e cpu-clock ./bench_gemm
perf report --stdio --sort=overhead,symbol | head -20
perf annotate gemm_naive_ikj
```
