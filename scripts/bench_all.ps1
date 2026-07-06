# Phase 1 一键性能分析（PowerShell 版）

$ErrorActionPreference = "Continue"
$PHASE1 = "c:\Users\16947\WorkBuddy\2026-06-29-14-11-22\AttnInferenceFramework\phase1_gemm"
$PROJECT = "c:\Users\16947\WorkBuddy\2026-06-29-14-11-22\AttnInferenceFramework"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Phase 1 一键性能分析 (Windows + WSL2)" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

# ===========================================================================
# Part 1: Windows C++ Benchmark（绑 P 核 + 理论峰值）
# ===========================================================================
Write-Host "`n[1/3] Windows C++ Benchmark（绑 P 核 + 理论峰值）" -ForegroundColor Cyan
$env:Path = "C:\mingw64\bin;$env:Path"

if (Test-Path "$PHASE1\build\bench_gemm.exe") {
    Write-Host "  运行 bench_gemm.exe ..." -ForegroundColor Yellow
    & "$PHASE1\build\bench_gemm.exe"
} else {
    Write-Host "  bench_gemm.exe 不存在，正在编译..." -ForegroundColor Red
    $cmakeBin = "$env:USERPROFILE\cmake\cmake-3.29.3-windows-x86_64\bin\cmake.exe"
    Push-Location $PHASE1
    if (Test-Path build) { Remove-Item -Recurse -Force build }
    mkdir build | Out-Null
    cd build
    & $cmakeBin .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
    mingw32-make -j4 bench_gemm
    Pop-Location
    Write-Host "  编译完成，重新运行..." -ForegroundColor Yellow
    & "$PHASE1\build\bench_gemm.exe"
}

# ===========================================================================
# Part 2: NumPy/OpenBLAS 对比
# ===========================================================================
Write-Host "`n[2/3] NumPy/OpenBLAS 对比" -ForegroundColor Cyan
Write-Host "  运行 scripts/bench_numpy.py ..." -ForegroundColor Yellow
python "$PROJECT\scripts\bench_numpy.py"

# ===========================================================================
# Part 3: WSL2 perf 分析
# ===========================================================================
Write-Host "`n[3/3] WSL2 perf 硬件事件分析" -ForegroundColor Cyan
Write-Host "  调用 WSL 中的 bench_all.sh ..." -ForegroundColor Yellow

# 把 Windows 路径转成 WSL 路径
$wslProject = "/mnt/c/Users/16947/WorkBuddy/2026-06-29-14-11-22/AttnInferenceFramework"
$wslScript = "$wslProject/scripts/bench_all.sh"

wsl -d Ubuntu -- bash -c "cd $wslProject && bash $wslScript 2>&1"

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "  全部测试完成！" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  结果汇总:"
Write-Host "    1. C++ benchmark  → 看上面的 GFLOPS 表 + 效率百分比"
Write-Host "    2. NumPy 对比     → 看差距倍数（NumPy/C++）"
Write-Host "    3. WSL2 perf      → 看 IPC + cache miss 率"
Write-Host ""
Write-Host "  下一步: 把结果记录到 docs/performance-log.md" -ForegroundColor Yellow
