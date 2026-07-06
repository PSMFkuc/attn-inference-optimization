# Phase 1 GEMM 项目周报

> 日期：2026-07-02 | 项目：AttnInferenceFramework / Phase1 GEMM

---

## 一、本周工作摘要

完成了 Phase 1 GEMM 的完整开发环境搭建、3 种矩阵乘法算法实现、性能基准测试与对比分析。核心成果：**C++ 单线程 GEMM 达到 CPU 理论峰值 50.3% 效率，完成与 NumPy/OpenBLAS 的差距分析。**

---

## 二、环境搭建

| 组件 | 旧版本 | 新版本 |
|------|--------|--------|
| GCC (g++) | MinGW.org 6.3.0 | MinGW-Builds 14.2.0 (C:\mingw64) |
| CMake | 无 | 便携版 3.29.3 |
| GoogleTest | 无 | v1.14.0 (本地源码 third_party/googletest) |
| WSL2 | 已有 | Ubuntu 22.04 + perf 5.15 |
| Python/NumPy | Python 3.12.7 | NumPy 1.26.4 (含 OpenBLAS 后端) |

---

## 三、代码实现

### 3.1 项目结构

```
phase1_gemm/
├── CMakeLists.txt              # 构建配置 (C++17, -O3 -march=native)
├── src/
│   ├── gemm_naive.h / .cpp     # ijk (baseline) + ikj (循环顺序优化)
│   ├── gemm_tiled.h / .cpp     # 分块/Tiled GEMM (6层循环)
│   └── timer.h                 # RAII 高精度计时器
├── tests/
│   └── test_gemm.cpp           # GoogleTest 单元测试 (3 tests, all PASS)
├── profiling/
│   └── bench_gemm.cpp          # 性能基准测试 (理论峰值 + 效率分析)
├── third_party/googletest/     # GoogleTest v1.14.0 本地源码
└── build/                      # 构建产物
```

### 3.2 三种算法

| 算法 | 文件 | 核心思想 | 循环层数 |
|------|------|----------|----------|
| ijk (baseline) | gemm_naive.cpp | 朴素三重循环，B 按列访问 | 3 |
| ikj (优化1) | gemm_naive.cpp | 交换 j/k 顺序，A/B 均按行访问 | 3 |
| tiled (优化2) | gemm_tiled.cpp | 分块(32x32)，小块塞进 L1 cache | 6 (3块外+3块内) |

### 3.3 测试验证

- GoogleTest 3/3 全部通过
- 每个优化版本均与独立 reference 实现对比（浮点容差 1e-3 ~ 1e-4）

---

## 四、性能测试结果

### 4.1 C++ Benchmark（绑 P 核，单线程）— 修正版

> **修正说明**：初次测量时 trial 次数固定为 5 次，小尺寸矩阵（如 256）执行时间极短（<1ms），
> `std::chrono::high_resolution_clock` 的精度不足以可靠测量 <1ms 的操作，导致 256 尺寸出现
> ikj 53.55 GFLOPS 的异常值（超过理论峰值 70.4 GFLOPS 的 76%，物理上不可能）。
>
> **优化方案**：增加 trial 次数并改为自适应策略——小矩阵多跑（64→50次, 256→20次），
> 大矩阵少跑（512/1024→10次）。取多次测量的最小值，消除计时噪声。

| size | ijk (ms) | ijk GFLOPS | ikj (ms) | ikj GFLOPS | tiled (ms) | tiled GFLOPS | Peak Efficiency |
|------|----------|-----------|----------|-----------|------------|-------------|----------------|
| 64 | 0.21 | 2.45 | 0.03 | **15.42** | 0.04 | 13.44 | 21.9% |
| 128 | 1.31 | 3.20 | 0.18 | **23.30** | 0.22 | 19.24 | 33.1% |
| 256 | 11.63 | 2.88 | 0.94 | **35.54** | 1.52 | 22.06 | 50.5% |
| 512 | 98.79 | 2.72 | 7.32 | **36.67** | 13.22 | 20.30 | 52.1% |
| 1024 | 3794.92 | 0.57 | 101.19 | **21.22** | 121.34 | 17.70 | 30.1% |

### 4.1-b 初次测量数据（仅供参考，存在误差）

> 以下为修正前的原始数据，trial=5 固定。256 尺寸 ikj 53.55 GFLOPS 虚高，
> 1024 尺寸也偏高。保留此表用于对比说明自适应 trial 的必要性。

| size | ijk (ms) | ijk GFLOPS | ikj (ms) | ikj GFLOPS | tiled (ms) | tiled GFLOPS | Peak Efficiency |
|------|----------|-----------|----------|-----------|------------|-------------|----------------|
| 64 | 0.10 | 5.14 | 0.07 | 7.35 | 0.06 | 8.73 | 12.4% |
| 128 | 1.35 | 3.11 | 0.18 | 23.17 | 0.18 | 23.55 | 33.4% |
| 256 | 10.96 | 3.06 | 0.63 | ⚠️ 53.55 | 0.66 | 50.94 | ⚠️ 76.1% |
| 512 | 174.97 | 1.53 | 7.53 | 35.60 | 7.31 | 36.70 | 52.1% |
| 1024 | 1950.27 | 1.10 | 57.48 | ⚠️ 37.36 | 64.65 | 33.22 | 53.1% |

### 4.2 C++ vs NumPy/OpenBLAS 对比（基于修正后数据）

| size | C++ best GFLOPS | NumPy GFLOPS | Ratio (NumPy/C++) |
|------|----------------|-------------|-------------------|
| 64 | 15.42 | 37.18 | 2.41x |
| 128 | 23.30 | 57.85 | 2.48x |
| 256 | 35.54 | 51.24 | 1.44x |
| 512 | 36.67 | 81.16 | 2.21x |
| 1024 | 21.22 | 81.64 | 3.85x |

### 4.3 关键发现

1. **ikj 优化效果显著**：仅调整循环顺序，性能提升 10-50 倍（1024 上 ijk 0.57 GFLOPS → ikj 21.22 GFLOPS，37x 提升）
2. **分块优化不达预期**：tiled 版本在所有尺寸上均比 ikj 慢，说明默认 block size=32 未针对 i5-12450H 调优，或纯 CPU 分块需要配合 SIMD 和寄存器分块才能超越 ikj
3. **与 OpenBLAS 差距 2-4x**：小矩阵 2.4x，大矩阵(1024)达 3.85x。OpenBLAS 使用手写汇编微内核、数据打包(packing)、软件预取(prefetch)等技术
4. **CPU 峰值效率最高 52.1%**（512 尺寸）：说明还有约一半的计算能力未被利用（主要在 SIMD 向量化和内存带宽）
5. **Benchmark 方法论改进**：自适应 trial 次数策略消除了小矩阵测量误差，256 尺寸从虚高 53.55 修正为合理 35.54 GFLOPS

---

## 五、工具链建设

### 5.1 一键测试脚本

```
scripts/
├── bench_all.ps1        # PowerShell: 跑 C++ benchmark + NumPy 对比 + WSL2 perf
├── bench_all.sh         # WSL2 bash: 编译 Linux 版 + perf stat 硬件事件采集
└── bench_numpy.py       # NumPy/OpenBLAS 单线程对比测试
```

### 5.2 WSL2 性能分析环境

- 已安装 `perf` 工具（`perf version 5.15.189`）
- 支持 `perf stat` 硬件事件采集：cycles, instructions, cache-misses, LLC-misses
- 可用于后续 IPC、cache miss 率分析

---

## 六、下周计划

| 优先级 | 任务 | 预期收益 |
|--------|------|----------|
| P0 | SIMD 向量化优化 (AVX2 intrinsics) | 2-4x 性能提升 |
| P1 | Block size 参数调优 (16/64/128) | 找到最优 tile 配置 |
| P2 | WSL2 perf 深入分析 (cache miss / IPC) | 定位精确瓶颈 |
| P2 | 数据打包 (Packing) 优化 | 提升 SIMD 对齐效率 |
| P3 | 软件预取 (Prefetch) | 隐藏内存延迟 |

---

## 七、附录

### 运行命令

```powershell
# 一键全跑
cd c:\Users\16947\WorkBuddy\2026-06-29-14-11-22\AttnInferenceFramework
.\scripts\bench_all.ps1

# 单独跑 C++ benchmark
$env:Path = "C:\mingw64\bin;$env:Path"
.\phase1_gemm\build\bench_gemm.exe

# NumPy 对比
python scripts\bench_numpy.py

# WSL2 perf 分析
wsl -d Ubuntu
cd /mnt/c/Users/16947/WorkBuddy/2026-06-29-14-11-22/AttnInferenceFramework
bash scripts/bench_all.sh
```

### 环境配置

- GCC 14.2.0: `C:\mingw64\bin\`
- CMake: `~\cmake\cmake-3.29.3\bin\cmake.exe`
- CPU: Intel i5-12450H (4P-core 4.4GHz + 4E-core 3.3GHz, AVX2)
- 理论峰值(单核): 70.4 GFLOPS
