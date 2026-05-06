// 引入核函数接口声明
#include "../kernels_interface.h"
// 本核函数的声明头文件
#include "argmax_kernel.cuh"
// 张量（Tensor）相关头文件
#include "tensor/tensor.h"

namespace kernel {

/**
 * @brief Warp 级别的归约求最大值下标
 * @param val 输入：当前线程的最大值；输出：warp内的最大值
 * @param ptr 输入：当前线程的最大值下标；输出：warp内的最大值下标
 * @note Warp 是 GPU 最小执行单元，一个 warp = 32 个线程
 */
__forceinline__ __device__ void warp_reduce_argmax(float& val, size_t& ptr) {
  float tmp_val;       // 临时存储从其他线程 shuffle 过来的值
  size_t tmp_ptr;      // 临时存储从其他线程 shuffle 过来的下标
  // 让 warp 内所有线程同步，生成线程掩码
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);

  // 二分法归约：从 16 → 8 → 4 → 2 → 1，不断比较相邻线程
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1) {
    // 从下方 k 个线程的位置，把值和下标 shuffle 过来
    tmp_val = __shfl_down_sync(mask, val, k, warpSize);
    tmp_ptr = __shfl_down_sync(mask, ptr, k, warpSize);

    // 如果下标无效，跳过
    if (ptr == SIZE_MAX || tmp_ptr == SIZE_MAX) continue;

    // 核心逻辑：谁大就保留谁；值相等时保留下标更小的
    if (tmp_val > val) {
      val = tmp_val;
      ptr = tmp_ptr;
    } else if (tmp_val == val && tmp_ptr < ptr) {
      ptr = tmp_ptr;
    }
  }
}

/**
 * @brief 线程块级别的归约求最大值下标
 * @param val 线程局部最大值
 * @param ptr 线程局部最大值下标
 * @param shared_value 共享内存：存储每个warp的最大值
 * @param shared_ptr 共享内存：存储每个warp的最大值下标
 */
__forceinline__ __device__ void block_reduce_argmax(float& val, size_t& ptr, float* shared_value,
                                                    size_t* shared_ptr) {
  // 计算当前线程在 warp 内的编号（0~31）
  int lane_id = threadIdx.x % warpSize;
  // 计算当前线程属于第几个 warp
  int warp_id = threadIdx.x / warpSize;

  // 第一步：先在自己的 warp 内部做归约，得到每个 warp 的最大值
  warp_reduce_argmax(val, ptr);

  // 同步所有线程，确保 warp 归约完成
  __syncthreads();

  // 每个 warp 的 0 号线程，把本 warp 的最大值写入共享内存
  if (lane_id == 0) {
    shared_value[warp_id] = val;
    shared_ptr[warp_id] = ptr;
  }

  // 同步，确保共享内存写入完成
  __syncthreads();

  // 让每个 warp 的 0 号线程，重新加载共享内存里的 warp 最大值
  if (threadIdx.x < blockDim.x / warpSize) {
    val = shared_value[lane_id];
    ptr = shared_ptr[lane_id];
  } else {
    // 非 0 号线程赋无效值，不参与最终归约
    val = 0;
    ptr = SIZE_MAX;
  }

  // 第二步：对所有 warp 的最大值，再做一次 warp 归约，得到整个块的最大值
  if (warp_id == 0) {
    warp_reduce_argmax(val, ptr);
  }
}

/**
 * @brief CUDA 核函数：float32 类型求 argmax（找最大值下标）
 * @param input_ptr 输入数组（GPU上）
 * @param size 输入数组长度
 * @param output_idx 输出：最大值下标（GPU上）
 */
__global__ void argmax_kernel_fp32(const float* input_ptr, size_t size, size_t* output_idx) {
  // 共享内存：最多存 32 个 warp 的最大值（512线程 = 16个warp，32足够）
  __shared__ size_t shared_max_ptr[32];
  __shared__ float shared_max_value[32];

  // 当前线程在线程块内的 ID
  uint32_t tid = threadIdx.x;

  // 线程 ID 超过数组长度，直接返回（不处理）
  if (tid >= size) {
    return;
  }

  // ===================== 第一步：每个线程先找自己负责的元素的最大值 =====================
  // 初始化：当前线程的最大值下标 = 自己的 tid
  size_t max_index = threadIdx.x;
  // 初始化：当前线程的最大值 = 数组对应位置的值
  float max_value = input_ptr[max_index];

  // 线程并行遍历：步长 = 块内总线程数（512）
  // 比如 tid=0 → 0,512,1024...；tid=1 →1,513,1025...
  for (size_t i = tid; i < size; i += blockDim.x) {
    // 如果当前元素更大，更新最大值和下标
    if (input_ptr[i] > max_value) {
      max_index = i;
      max_value = input_ptr[i];
    }
  }

  // ===================== 第二步：块内归约，把所有线程的最大值合并成一个 =====================
  block_reduce_argmax(max_value, max_index, shared_max_value, shared_max_ptr);
  __syncthreads();

  // ===================== 第三步：0 号线程把最终结果写回输出 =====================
  if (threadIdx.x == 0) {
    *output_idx = max_index;
  }
}

/**
 * @brief CPU 端调用接口：启动 argmax 核函数
 * @param input_ptr 输入数据指针（GPU上）
 * @param size 数据长度
 * @param stream CUDA 流（可以为空）
 * @return 最大值下标（CPU端）
 */
size_t argmax_kernel_cu(const float* input_ptr, size_t size, void* stream) {
  // 获取 CUDA 内存分配器
  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();

  // 在 GPU 上申请 1 个 size_t 大小的内存，存结果下标
  size_t* index = static_cast<size_t*>(alloc_cu->allocate(sizeof(size_t)));
  // CPU 端接收结果的变量
  size_t output_index = 0;

  // 如果没有传入 CUDA 流，使用默认流
  if (!stream) {
    // 启动核函数：1 个线程块，每个块 512 线程
    argmax_kernel_fp32<<<1, 512>>>(input_ptr, size, index);
    // 把结果从 GPU 拷贝到 CPU
    cudaMemcpy(&output_index, index, sizeof(size_t), cudaMemcpyDeviceToHost);
  } else {
    // 使用传入的 CUDA 流
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    // 异步启动核函数
    argmax_kernel_fp32<<<1, 512, 0, stream_>>>(input_ptr, size, index);
    // 异步拷贝结果到 CPU
    cudaMemcpyAsync(&output_index, index, sizeof(size_t), cudaMemcpyDeviceToHost, stream_);
  }

  // 释放 GPU 内存
  alloc_cu->release(index);
  // 返回最大值下标
  return output_index;
}

}  // namespace kernel