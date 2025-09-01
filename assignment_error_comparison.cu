#include <stdio.h>
#include <cuda_runtime.h>

// 预期的正确任务分配（注释）vs 实际错误的分配（代码）
__global__ void taskAssignmentDemo(int* input, int* output, int n) {
    int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalTid >= n) return;

    // 预期分配（正确）：
    // 0-31线程 → 加10（任务A）
    // 32-63线程 → 乘2（任务B）
    // 64-95线程 → 减5（任务C）
    
    // 实际错误分配（范围边界错误）：
    if (globalTid <= 32) {       // 错误：应该是 <32，写成了 <=32
        output[globalTid] = input[globalTid] + 10;  // 任务A
    } else if (globalTid <= 63) { // 正确：33-63（实际少了线程32）
        output[globalTid] = input[globalTid] * 2;   // 任务B
    } else if (globalTid <= 97) { // 错误：应该是 <=95，写成了 <=97
        output[globalTid] = input[globalTid] - 5;   // 任务C
    } else {
        output[globalTid] = input[globalTid];
    }
}

int main() {
    const int n = 128;
    int h_input[n], h_output[n];
    int *d_input, *d_output;

    // 初始化输入数据（0-127）
    for (int i = 0; i < n; i++) {
        h_input[i] = i;
    }

    // 分配设备内存并拷贝数据
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    // 启动核函数（1个线程块，128线程）
    taskAssignmentDemo<<<1, 128>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    // 拷贝结果到主机
    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印预期与实际结果对比
    printf("=== 关键线程的结果对比 ===\n");
    
    // 线程32：本应属于任务B（32×2=64），却被错误分配到任务A（32+10=42）
    printf("线程32:\n");
    printf("预期结果: %d (32×2)\n", 32 * 2);
    printf("实际结果: %d (32+10)\n\n", h_output[32]);
    
    // 线程63：属于任务B，结果正确
    printf("线程63:\n");
    printf("预期结果: %d (63×2)\n", 63 * 2);
    printf("实际结果: %d\n\n", h_output[63]);
    
    // 线程95：属于任务C，结果正确
    printf("线程95:\n");
    printf("预期结果: %d (95-5)\n", 95 - 5);
    printf("实际结果: %d\n\n", h_output[95]);
    
    // 线程96：本应不执行任务C（应从96开始不执行），却错误执行了任务C
    printf("线程96:\n");
    printf("预期结果: %d (不执行任务C)\n", 96);
    printf("实际结果: %d (96-5)\n", h_output[96]);

    // 释放内存
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
