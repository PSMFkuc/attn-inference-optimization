# Phase 4 CPU Inference Acceleration

## Positioning

This project has no CUDA-capable GPU in the current environment, so Phase 4 follows the CPU inference acceleration path: OpenMP operator parallelism, graph-level parallel scheduling, and INT8 weight quantization. This is the llama.cpp-style route rather than the TensorRT/CUDA route.

## Deliverables

Phase 4 adds an independent acceleration layer under `phase4_gpu/`:

- `Backend`: a small execution-provider interface for one graph node.
- `CpuParallelBackend`: OpenMP-backed MatMul, Softmax, LayerNorm, GELU, and Attention.
- `ParallelGraphExecutor`: runs the Phase 3 DAG either serially or by execution levels, so independent nodes like `q_proj`, `k_proj`, and `v_proj` can run concurrently.
- `quantization`: symmetric INT8 weight quantization plus FP32 activation x INT8 weight GEMM.
- `test_phase4_cpu`: correctness checks for levelization, graph execution, row-wise LayerNorm, and INT8 error.
- `bench_phase4_cpu`: benchmark for naive GEMM, parallel GEMM, INT8 weight GEMM, and graph scheduling.

## Design Decisions

Phase 4 intentionally does not rewrite the Phase 3 graph engine. It reuses `ComputeGraph`, `Node`, and `Tensor`, then adds a backend boundary on top. That keeps Phase 3 as the model representation and scheduling source while Phase 4 owns accelerated execution.

The graph-level parallel executor first snapshots each level's input tensors, launches independent node work, and writes outputs back after the level completes. This avoids data races with the current `ComputeGraph` tensor store, whose `unordered_map` is not thread-safe.

LayerNorm is implemented row-wise in Phase 4 because transformer inference normalizes the hidden dimension per token. This is more faithful to inference workloads than normalizing the entire flattened tensor.

INT8 is implemented as weight-only quantization:

```text
W_fp32[K, N] -> W_int8[K, N] + scale[N]
Y_fp32[M, N] = X_fp32[M, K] x dequant(W_int8[K, N])
```

This path demonstrates quantization error and memory reduction without depending on CPU-specific AVX2 intrinsics. A later optimization can replace the scalar inner loop with AVX2/VNNI kernels.

## Build and Run

```powershell
cd phase4_gpu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --output-on-failure
.\build\test_phase4_cpu.exe
.\build\bench_phase4_cpu.exe
```

The current machine does not expose `cmake` on PATH. The implementation was
also verified with the bundled MinGW toolchain:

```powershell
cd phase4_gpu
..\mingw64\bin\g++.exe -std=c++17 -O2 -fopenmp -Iinclude -I..\phase1_gemm\src -I..\phase2_operators\include -I..\phase3_graph\include tests\test_phase4_cpu.cpp src\cpu_parallel_backend.cpp src\parallel_graph_executor.cpp src\quantization.cpp ..\phase1_gemm\src\gemm_naive.cpp ..\phase1_gemm\src\gemm_tiled.cpp ..\phase1_gemm\src\gemm_parallel.cpp ..\phase2_operators\src\operators.cpp ..\phase3_graph\src\graph.cpp ..\phase3_graph\src\json_parser.cpp ..\phase3_graph\src\memory_pool.cpp -o build_manual\test_phase4_cpu.exe
```

## Current Verification

`test_phase4_cpu` passes:

```text
Results: 27 passed, 0 failed
```

Representative benchmark data from this Windows/MinGW environment:

```text
GEMM Paths
size     naive(ms)    parallel(ms) int8w(ms)    int8_err
128      1.978        2.269        1.059        0.01516
256      8.290        3.939        4.355        0.02296
512      66.228       15.985       24.736       0.03274

Graph Schedule
shape            serial(ms)   level(ms)    levels
64x64            1.004        6.435        3
128x64           2.280        10.127       3
```

The graph-level parallel path is correct but slower in this first version. The
reason is structural: it snapshots tensors to avoid races with the Phase 3
`unordered_map` tensor store, and it can oversubscribe threads because each
MatMul already uses OpenMP internally. This is a useful measured result, not a
failure: the next optimization is choosing either graph-level parallelism or
operator-level OpenMP per workload, instead of enabling both blindly.

## Expected Analysis Points

- Small matrices may not benefit from thread-level parallelism because launch and scheduling overhead dominate.
- Larger GEMMs should shift the bottleneck toward memory bandwidth and cache reuse.
- Graph-level parallelism helps only where the DAG has independent ready nodes.
- INT8 weight-only GEMM primarily reduces weight bandwidth and memory footprint; scalar implementation may not beat optimized FP32 until vectorized.
- This Phase 4 path naturally leads into Phase 5 CPU Flash Attention or future CUDA backend work.
