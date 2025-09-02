#include <stdio.h>
#include <cuda_runtime.h>

// 全局变量，用于线程间通信
__device__ int flag = 0;
__device__ int data = 0;

// 没有同步的核函数
__global__ void unsynchronizedKernel() {
    int tid = threadIdx.x;
    
    if (tid == 0) {
        // 线程0: 先更新数据，再设置标志
        data = 42;      // 写操作1
        flag = 1;       // 写操作2
    } else if (tid == 1) {
        // 线程1: 等待标志被设置，然后读取数据
        while (flag == 0);  // 等待
        printf("线程1读取到的数据: %d (期望42)\n", data);
    }
}

// 有同步的核函数
__global__ void synchronizedKernel() {
    int tid = threadIdx.x;
    
    if (tid == 0) {
        // 线程0: 先更新数据，再设置标志
        data = 42;      // 写操作1
        __threadfence(); // 内存栅栏，确保data的修改对其他线程可见
        flag = 1;       // 写操作2
    } else if (tid == 1) {
        // 线程1: 等待标志被设置，然后读取数据
        while (flag == 0);  // 等待
        __threadfence(); // 内存栅栏，确保读取到最新的数据
        printf("线程1读取到的数据: %d (期望42)\n", data);
    }
}

int main() {
    printf("=== 没有同步的情况 ===\n");
    unsynchronizedKernel<<<1, 2>>>();
    cudaDeviceSynchronize();
    
    // 重置全局变量（使用新的API替代cudaMemsetToSymbol）
    int zero = 0;
    cudaMemcpyToSymbol(flag, &zero, sizeof(int));
    cudaMemcpyToSymbol(data, &zero, sizeof(int));
    
    printf("\n=== 有同步的情况 ===\n");
    synchronizedKernel<<<1, 2>>>();
    cudaDeviceSynchronize();
    
    return 0;
}
    