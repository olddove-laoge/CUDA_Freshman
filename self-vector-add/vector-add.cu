#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

// 1. 内核函数（GPU端执行）：每个线程负责一个元素的加法
// __global__ 标识这是一个可被Host调用、在Device上执行的内核函数
__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    // 计算当前线程的全局唯一索引（关键！线程通过索引确定自己要处理的数据）
    // threadIdx.x：线程在块内的编号（0~blockDim.x-1）
    // blockIdx.x：线程块在网格内的编号（0~gridDim.x-1）
    // blockDim.x：每个线程块的线程数
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // 防止线程索引超出向量长度（当n不是blockDim.x的整数倍时）
    if (tid < n) {
        C[tid] = A[tid] + B[tid];
        // 打印当前线程的信息（仅用于演示，实际并行计算中应避免大量打印）
        printf("线程块编号: %d, 块内线程编号: %d, 全局索引: %d, 计算: %f + %f = %f\n",
               blockIdx.x, threadIdx.x, tid, A[tid], B[tid], C[tid]);
    }
}

// 2. Host端代码（CPU执行）：负责控制流程、准备数据、启动内核
int main() {
    // 定义向量长度（为简化演示，使用较小的数值）
    const int n = 16;
    size_t size = n * sizeof(float);

    // Host端内存（CPU可直接访问）
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // 初始化Host端数据
    for (int i = 0; i < n; i++) {
        h_A[i] = static_cast<float>(i);       // A = [0,1,2,...,15]
        h_B[i] = static_cast<float>(i * 2);   // B = [0,2,4,...,30]
    }

    // Device端内存（GPU可直接访问，Host需通过API操作）
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Host→Device：将数据从CPU内存复制到GPU内存
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 3. 配置线程结构并启动内核（Host→GPU的关键交互）
    int blockSize = 8;                  // 每个线程块包含8个线程
    int gridSize = (n + blockSize - 1) / blockSize;  // 计算需要的线程块数（向上取整）
    printf("\nHost启动内核：网格大小=%d（线程块数），块大小=%d（每块线程数），总线程数=%d\n\n",
           gridSize, blockSize, gridSize * blockSize);

    // 启动内核：<<<网格大小, 块大小>>> 是执行配置，指定线程组织方式
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();  // Host等待GPU完成内核执行（同步操作）

    // Device→Host：将计算结果从GPU内存复制回CPU内存
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 验证结果（Host端操作）
    printf("\n最终结果验证：\n");
    for (int i = 0; i < n; i++) {
        printf("C[%d] = %f (预期: %f)\n", i, h_C[i], h_A[i] + h_B[i]);
    }

    // 释放内存
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
