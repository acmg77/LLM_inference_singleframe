#!/usr/bin/env bash
# KuiperLLama 一键编译脚本（在容器内运行）
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_TYPE="${1:-qwen2}"   # 可选: llama3 | qwen2 | qwen3

echo "======================================"
echo " KuiperLLama 编译脚本"
echo " 目标模型类型: $MODEL_TYPE"
echo "======================================"

mkdir -p build && cd build

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DUSE_CPM=ON"

case "$MODEL_TYPE" in
  llama3)
    CMAKE_ARGS="$CMAKE_ARGS -DLLAMA3_SUPPORT=ON"
    ;;
  qwen2)
    CMAKE_ARGS="$CMAKE_ARGS -DQWEN2_SUPPORT=ON"
    ;;
  qwen3)
    CMAKE_ARGS="$CMAKE_ARGS -DQWEN3_SUPPORT=ON"
    ;;
  *)
    echo "未知模型类型: $MODEL_TYPE，使用默认（无特殊支持）"
    ;;
esac

echo "CMake 参数: $CMAKE_ARGS"
cmake $CMAKE_ARGS ..
make -j$(nproc)

echo ""
echo "编译完成！可执行文件位于 build/demo/ 目录下。"
