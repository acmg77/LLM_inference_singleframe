// 引入张量（Tensor）类的头文件，管理数据、维度、设备
#include <tensor/tensor.h>
// 引入NVIDIA CUB库：CUDA高性能并行计算库，这里用于【块级归约求和】
#include <cub/block/block_reduce.cuh>
// 引入核函数对外接口的声明
#include "../kernels_interface.h"
// 本核函数的声明头文件
#include "matmul_kernel.cuh"

// 所有CUDA核函数都放在 kernel 命名空间
namespace kernel {

// ====================== 核函数1：FP32全精度矩阵乘法 ======================
// 模板参数：THREAD_PER_BLOCK=每个线程块的线程数，ROW_PER_BLOCK=每个块计算的输出行数
// __global__：CUDA核函数修饰符，由CPU调用，在GPU上执行
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(
    const float* input,    // 输入向量：1×M （FP32）
    const float* weight,   // 权重矩阵：K×M （FP32）
    float* output,         // 输出向量：1×K （FP32）
    int M,                 // 权重矩阵列数 / 输入向量长度
    int K) {               // 权重矩阵行数 / 输出向量长度

  // 共享内存：GPU块内线程共享的高速内存，存储每个线程的局部求和结果
  // 大小 = 每个块的线程数，类型float
  __shared__ float sdata[THREAD_PER_BLOCK];
  // 获取当前线程在【线程块内】的索引 (0 ~ THREAD_PER_BLOCK-1)
  unsigned int tid = threadIdx.x;

  // 计算当前线程块负责的输出行范围
  // blockIdx.x = 当前线程块在网格中的索引（每个块处理1行输出）
  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  // 越界判断：如果起始行超过总K行，直接返回（无效线程）
  if (start_row >= K) {
    return;
  }

  // ====================== 优化：float4向量内存加载 ======================
  // CUDA优化技巧：一次加载4个float，提升内存访问带宽（减少访存次数）
  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;   // 完整的4float组数量
  const int pack_off = pack_size * pack_num; // 完整组的结束偏移量

  // 循环：当前线程块负责的输出行（这里ROW_PER_BLOCK=1，即每个块算1行）
#pragma unroll  // 编译指令：手动展开循环，提升执行速度
  for (int p = start_row; p < end_row; ++p) {
    // 初始化当前线程的局部和为0
    sdata[tid] = 0;
    // 权重矩阵第p行的起始偏移量：p*M
    int row_offset = p * M;

    // 把输入/权重指针强转为float4指针，实现一次读取4个float
    float4* input_float4_ptr = (float4*)input;
    float4* weight_float4_ptr = (float4*)(weight + row_offset);

    // ====================== 第一步：向量并行计算（4个float一组） ======================
#pragma unroll
    for (int i = tid; i < pack_num; i += blockDim.x) {
      // 读取4个float的输入/权重数据
      float4 input_float4 = *(input_float4_ptr + i);
      float4 weight_float4 = *(weight_float4_ptr + i);
      // 计算4个元素的乘加和：x*y + x*y + x*y + x*y
      float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                       input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
      // 累加到当前线程的共享内存中
      sdata[tid] += part_sum;
    }

    // ====================== 第二步：处理剩余不足4个的元素 ======================
    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      sdata[tid] += input[i] * weight[row_offset + i];
    }

    // 块内线程同步：等待所有线程完成计算，再执行归约
    __syncthreads();

    // ====================== CUB块级归约：把所有线程的和求和 ======================
    // 定义CUB块归约类型：对THREAD_PER_BLOCK个float求和
    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    // CUB需要的临时共享内存
    __shared__ typename BlockReduce::TempStorage temp;
    // 执行归约：将所有线程的sdata[tid]求和，得到最终结果（仅0号线程拿到总和）
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    // 同步：等待归约完成
    __syncthreads();

    // 0号线程：将最终求和结果写入输出张量
    if (tid == 0) {
      output[p] = part_sum;
    }
    // 同步：确保写入完成，再进入下一次循环
    __syncthreads();
  }
}

// ====================== 核函数2：FP32×INT8量化矩阵乘法 ======================
// 功能：输入FP32 + 权重INT8（量化）+ 缩放因子scale → 输出FP32
// 量化优势：INT8权重占用内存更小，推理更快
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(
    const float* input,    // 输入向量：1×M （FP32）
    const int8_t* weight,  // 权重矩阵：K×M （INT8量化）
    const float* scales,   // 量化缩放因子：反量化回FP32
    const int32_t group_size, // 量化分组大小
    float* output,         // 输出向量：1×K
    int M,                 // 输入长度/权重列数
    int K) {               // 输出长度/权重行数

  // 共享内存：存储线程局部和
  __shared__ float sdata[THREAD_PER_BLOCK];
  // 线程块内索引
  unsigned int tid = threadIdx.x;

  // 计算当前块负责的输出行
  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

  // 每个块计算1行输出
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    // 线程并行遍历元素：i从tid开始，步长=块线程数
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      // 权重索引：p行i列
      const int weight_idx = p * M + i;
      // 分组量化：根据索引找到对应的缩放因子
      const int group_idx = weight_idx / group_size;
      // 核心计算：输入(FP32) × 缩放因子 × 权重(INT8强转FP32)
      sdata[tid] += input[i] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
    }

    // 同步：等待所有线程计算完成
    __syncthreads();

    // ====================== CUB块归约求和 ======================
    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    // 0号线程写结果
    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

// ====================== 对外接口1：调用FP32全精度核函数 ======================
// CPU端函数：参数校验 + 启动GPU核函数
void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  // 检查：输入非空、维度≤2
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  // 检查：数据在GPU上
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  // 权重参数校验：非空、2维矩阵、GPU上
  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);

  // 矩阵维度定义：weight(K行, M列)
  const int32_t K = weight.get_dim(0);
  const int32_t M = weight.get_dim(1);

  // 检查：输入长度 = 权重列数
  CHECK_EQ(M, input.get_dim(0));

  // ====================== 启动CUDA核函数 ======================
  // 执行配置：<<<网格大小, 线程块大小, 共享内存大小, CUDA流>>>
  // 网格大小=K（每个块算1个输出元素），线程块大小=128
  if (config && config->stream) {
    // 使用指定CUDA流（异步执行）
    matmul_kernel_cu_fp32<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    // 默认流执行
    matmul_kernel_cu_fp32<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<float>(),
                                              const_cast<float*>(output.ptr<float>()), M, K);
  }
}

// ====================== 对外接口2：调用INT8量化核函数 ======================
void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config) {
  // 参数校验
  CHECK(config != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);

  // 维度定义
  const int32_t K = weight.get_dim(0);
  const int32_t M = weight.get_dim(1);
  CHECK_EQ(M % 4, 0);
  CHECK_EQ(M, input.get_dim(0));

  // 启动核函数：网格K个块，每个块128线程
  if (config->stream) {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<int8_t>(),
                                                  scale.ptr<float>(), group_size,
                                                  const_cast<float*>(output.ptr<float>()), M, K);
  }
}

}  // namespace kernel