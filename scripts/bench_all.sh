#!/bin/bash
# ===========================================================================
# Phase 1 一键性能分析脚本
# 用法：在 WSL2 bash 中执行
#   cd /mnt/c/Users/16947/WorkBuddy/2026-06-29-14-11-22/AttnInferenceFramework
#   bash scripts/bench_all.sh
#
# 功能：
#   1. 编译 Linux 版 bench_gemm
#   2. perf stat 采集硬件事件（cache miss, IPC）
#   3. 跑 Windows 版 bench_gemm（对比）
#   4. 跑 NumPy 对比
# ===========================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PHASE1_DIR="$PROJECT_DIR/phase1_gemm"
WIN_EXE="$PHASE1_DIR/build/bench_gemm.exe"
LINUX_BUILD_DIR="$PHASE1_DIR/build_linux"

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}==========================================================${NC}"
echo -e "${GREEN}  Phase 1 一键性能分析${NC}"
echo -e "${GREEN}==========================================================${NC}"

# --- Step 1: 安装依赖 ---
echo -e "\n${CYAN}[1/4] 检查 WSL 编译环境...${NC}"
MISSING=""
command -v cmake >/dev/null 2>&1 || MISSING="$MISSING cmake"
command -v g++ >/dev/null 2>&1 || MISSING="$MISSING g++"
command -v make >/dev/null 2>&1 || MISSING="$MISSING make"
command -v perf >/dev/null 2>&1 || MISSING="$MISSING perf"

if [ -n "$MISSING" ]; then
    echo -e "${YELLOW}  缺少: $MISSING ，正在安装...${NC}"
    sudo apt update -qq
    sudo apt install -y -qq cmake g++ make linux-tools-common linux-tools-$(uname -r) 2>/dev/null || true
fi
echo -e "${GREEN}  编译环境就绪${NC}"

# --- Step 2: 编译 Linux 版本 ---
echo -e "\n${CYAN}[2/4] 编译 Linux 版 bench_gemm...${NC}"
mkdir -p "$LINUX_BUILD_DIR"
cd "$LINUX_BUILD_DIR"

cmake "$PHASE1_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native" \
    -DBUILD_TESTS=OFF \
    > /dev/null 2>&1
echo -e "  cmake 配置完成"

make bench_gemm -j$(nproc) > /dev/null 2>&1
echo -e "${GREEN}  编译完成 → $LINUX_BUILD_DIR/bench_gemm${NC}"

# --- Step 3: perf stat 硬件事件分析 ---
echo -e "\n${CYAN}[3/4] perf stat 硬件事件分析...${NC}"
echo -e "  采集指标: cycles, IPC, cache-miss, LLC-miss\n"

perf stat \
    -e cycles,instructions,cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,\
branches,branch-misses \
    ./bench_gemm 2>&1 | tee /tmp/perf_output.txt

# 提取关键指标
echo -e "\n${CYAN}--- 关键指标解读 ---${NC}"
IPC=$(grep "instructions" /tmp/perf_output.txt | grep -oP '[\d.]+(?=\s+insn per cycle)' || echo "N/A")
L1_MISS=$(grep "L1-dcache-load-misses" /tmp/perf_output.txt | grep -oP '[\d.]+%' || echo "N/A")
LLC_MISS=$(grep "LLC-load-misses" /tmp/perf_output.txt | grep -oP '[\d.]+%' || echo "N/A")

echo -e "  IPC (指令/周期):    ${GREEN}${IPC}${NC}  (>2=好, <0.5=等内存)"
echo -e "  L1 cache miss 率:   ${YELLOW}${L1_MISS}${NC}  (<5%=好, >20%=差)"
echo -e "  LLC (L3) miss 率:   ${RED}${LLC_MISS}${NC}  (越低越好，miss=必须访主存)"

# --- Step 4: Windows 版 + NumPy 提示 ---
echo -e "\n${CYAN}[4/4] 其他测试（手动执行）${NC}"
echo -e "  以下命令请在 PowerShell 中运行:"
echo -e ""
echo -e "  # Windows 版 benchmark（绑 P 核 + 理论峰值）"
echo -e "  cd ${PHASE1_DIR//\//\\}\\build"
echo -e "  \$env:Path = \"C:\\mingw64\\bin;\$env:Path\""
echo -e "  .\\bench_gemm.exe"
echo -e ""
echo -e "  # NumPy/OpenBLAS 对比"
echo -e "  cd ${PROJECT_DIR//\//\\}"
echo -e "  python scripts\\bench_numpy.py"
echo -e ""

echo -e "${GREEN}==========================================================${NC}"
echo -e "${GREEN}  一键脚本完成！${NC}"
echo -e "${GREEN}==========================================================${NC}"
