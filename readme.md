# KuiperLLama

一个使用 C++ / CUDA 实现的大模型推理项目，当前包含 Llama 系列和 Qwen 系列模型的部分推理与导出流程示例。

## 项目说明

这个仓库主要包含以下内容：

- 基于 CMake 的 C++ / CUDA 工程结构
- 常见推理算子的 CPU / CUDA 实现
- Llama / Qwen 部分模型的推理示例
- 模型导出相关脚本
- 单元测试与简单验证代码

## 目录结构

- `kuiper/`：核心推理框架代码
- `demo/`：推理示例入口
- `test/`：测试代码
- `tools/`：模型导出与辅助脚本
- `hf_infer/`：Hugging Face 侧参考推理脚本
- `cmake/`：CMake 辅助脚本
- `models/`：本地模型目录示例

## 依赖

项目依赖以下组件：

1. [glog](https://github.com/google/glog)
2. [googletest](https://github.com/google/googletest)
3. [sentencepiece](https://github.com/google/sentencepiece)
4. [armadillo](https://arma.sourceforge.net/download.html)
5. CUDA Toolkit

如果本地未安装完整依赖，可以在配置 CMake 时启用 `USE_CPM=ON` 自动拉取部分依赖。

## 编译

```shell
mkdir build
cd build
cmake ..
make -j16
```

如果希望通过 CPM 自动下载部分依赖：

```shell
mkdir build
cd build
cmake -DUSE_CPM=ON ..
make -j16
```

## 基本运行

### Llama 示例

```shell
./build/demo/llama_infer llama2_7b.bin tokenizer.model
```

### Qwen 示例

```shell
./build/demo/qwen_infer Qwen2.5-0.5B.bin Qwen/Qwen2.5-0.5B/tokenizer.json
```

## 模型导出

### Llama 3.2

以 `meta-llama/Llama-3.2-1B` 为例：

1. 下载模型：

```shell
export HF_ENDPOINT=https://hf-mirror.com
pip3 install huggingface-cli
huggingface-cli download --resume-download meta-llama/Llama-3.2-1B --local-dir meta-llama/Llama-3.2-1B --local-dir-use-symlinks False
```

2. 导出二进制模型：

```shell
python3 tools/export.py Llama-3.2-1B.bin --hf=meta-llama/Llama-3.2-1B
```

3. 编译：

```shell
mkdir build
cd build
cmake -DUSE_CPM=ON -DLLAMA3_SUPPORT=ON ..
make -j16
```

4. 运行：

```shell
./build/demo/llama_infer Llama-3.2-1B.bin meta-llama/Llama-3.2-1B/tokenizer.json
python3 hf_infer/llama3_infer.py
```

### Qwen 2.5

以 `Qwen/Qwen2.5-0.5B` 为例：

1. 下载模型：

```shell
export HF_ENDPOINT=https://hf-mirror.com
pip3 install huggingface-cli
huggingface-cli download --resume-download Qwen/Qwen2.5-0.5B --local-dir Qwen/Qwen2.5-0.5B --local-dir-use-symlinks False
```

2. 导出二进制模型：

```shell
python3 tools/export_qwen2.py Qwen2.5-0.5B.bin --hf=Qwen/Qwen2.5-0.5B
```

3. 编译：

```shell
mkdir build
cd build
cmake -DUSE_CPM=ON -DQWEN2_SUPPORT=ON ..
make -j16
```

4. 运行：

```shell
./build/demo/qwen_infer Qwen2.5-0.5B.bin Qwen/Qwen2.5-0.5B/tokenizer.json
python3 hf_infer/qwen2_infer.py
```

### Qwen 3

Qwen 3 的导出流程在 `tools/export_qwen3/` 目录下，通常分为两步：

1. 在 `tools/export_qwen3/load.py` 中配置输入模型名与输出路径，导出为 `pth`
2. 使用同目录下的 `write_bin.py` 将中间结果导出为可用于推理的二进制文件

编译时需要启用：

```shell
cmake -DQWEN3_SUPPORT=ON ..
```

## 说明

- `models/` 目录下的模型文件通常较大，适合本地保存，不建议直接提交到仓库。
- profiling 结果、构建目录和导出的二进制文件建议通过 `.gitignore` 排除。
- 具体实现细节可以从 `kuiper/source/` 和 `kuiper/include/` 开始阅读。
