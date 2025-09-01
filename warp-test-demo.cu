#include <stdio.h>
#include <cuda_runtime.h>

// 核函数：不同线程束执行不同任务
__global__ void warpTaskKernel(int* input, int* output, int n) {
    // 计算全局线程ID
    int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalTid >= n) return;

    // 计算线程束ID（32线程/束）
    int warpId = globalTid / 32;

    // 根据线程束ID分配不同任务
    switch (warpId % 3) {  // 模3让任务循环分配
        case 0:  // 线程束0：执行加法
            output[globalTid] = input[globalTid] + 10;
            break;
        case 1:  // 线程束1：执行乘法
            output[globalTid] = input[globalTid] * 2;
            break;
        default:  // 其他线程束：执行减法
            output[globalTid] = input[globalTid] - 5;
            break;
    }
}

int main() {
    const int n = 128;  // 数据量（刚好4个线程束：128/32=4）
    int h_input[n], h_output[n];
    int *d_input, *d_output;

    // 初始化输入数据
    for (int i = 0; i < n; i++) {
        h_input[i] = i;  // 输入：0,1,2,...,127
    }

    // 分配设备内存
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    // 启动核函数（1个线程块，128线程，刚好4个线程束）
    warpTaskKernel<<<1, 128>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    // 拷贝结果到主机
    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印部分结果（验证不同线程束的任务）
    printf("线程束0（线程0-31）：加法+10\n");
    printf("线程0结果：%d（期望0+10=10）\n", h_output[0]);
    printf("线程31结果：%d（期望31+10=41）\n", h_output[31]);

    printf("\n线程束1（线程32-63）：乘法×2\n");
    printf("线程32结果：%d（期望32×2=64）\n", h_output[32]);
    printf("线程63结果：%d（期望63×2=126）\n", h_output[63]);

    printf("\n线程束2（线程64-95）：减法-5\n");
    printf("线程64结果：%d（期望64-5=59）\n", h_output[64]);
    printf("线程95结果：%d（期望95-5=90）\n", h_output[95]);

    // 释放内存
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
