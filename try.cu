#include <stdio.h>
#include <cuda_runtime.h>

// -------------------------- 1. 核函数（在GPU上执行）--------------------------
// 每个Thread执行1次，负责计算1个向量元素的加法
__global__ void vectorAddKernel(const float* d_a, const float* d_b, float* d_c, int N) {
    // （1）计算当前Thread在全局所有Thread中的唯一编号（全局索引）
    // blockIdx.x：当前Block在Grid中的编号（x方向，因本例是1D任务，仅用x维度）
    // blockDim.x：每个Block包含的Thread数量（即 ThreadPerBlock）
    // threadIdx.x：当前Thread在所属Block中的编号（x方向）
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    // （2）避免Thread编号超过向量长度（防止越界访问）
    if (globalThreadId < N) {
        d_c[globalThreadId] = d_a[globalThreadId] + d_b[globalThreadId];
    }
}

// -------------------------- 2. 主机端函数（在CPU上执行，负责调度）--------------------------
void vectorAdd(const float* h_a, const float* h_b, float* h_c, int N) {
    int size = N * sizeof(float); // 向量数据总字节数
    float *d_a, *d_b, *d_c;       // 设备端（GPU）内存指针

    // -------------------------- 步骤1：分配GPU内存（cudaMalloc）
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // -------------------------- 步骤2：CPU数据拷贝到GPU（cudaMemcpyHostToDevice）
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // -------------------------- 步骤3：定义Block和Grid（软件层任务划分）
    // 1. ThreadPerBlock：每个Block包含的Thread数量（通常取32的倍数，匹配Warp粒度）
    const int ThreadPerBlock = 256; 
    // 2. BlockPerGrid：Grid中包含的Block数量（向上取整，确保覆盖所有元素）
    //    例：若N=1000，ThreadPerBlock=256 → BlockPerGrid=4（4×256=1024 ≥ 1000）
    const int BlockPerGrid = (N + ThreadPerBlock - 1) / ThreadPerBlock;

    printf("软件层任务划分：\n");
    printf("Block数量（BlockPerGrid）：%d\n", BlockPerGrid);
    printf("每个Block的Thread数量（ThreadPerBlock）：%d\n", ThreadPerBlock);
    printf("总Thread数量：%d（覆盖向量长度N=%d）\n", BlockPerGrid * ThreadPerBlock, N);

    // -------------------------- 步骤4：启动核函数（<<<BlockPerGrid, ThreadPerBlock>>> 是关键）
    // 作用：将Grid中的Block分配到GPU的SM上，每个Block的Thread被拆分为Warp执行
    vectorAddKernel<<<BlockPerGrid, ThreadPerBlock>>>(d_a, d_b, d_c, N);

    // -------------------------- 步骤5：GPU结果拷贝回CPU（cudaMemcpyDeviceToHost）
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // -------------------------- 步骤6：释放GPU内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// -------------------------- 3. 主函数（测试入口）--------------------------
int main() {
    const int N = 1000; // 向量长度
    float h_a[N], h_b[N], h_c[N]; // 主机端（CPU）内存数组

    // 初始化CPU端输入数据
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;    // h_a = [0, 1, 2, ..., 999]
        h_b[i] = (float)i * 2;// h_b = [0, 2, 4, ..., 1998]
    }

    // 调用向量加法（内部包含GPU调度）
    vectorAdd(h_a, h_b, h_c, N);

    // 验证结果（打印前5个和最后5个元素，确保正确）
    printf("\n验证结果（d_c[i] = d_a[i] + d_b[i]）：\n");
    for (int i = 0; i < 5; i++) {
        printf("h_c[%d] = %.0f + %.0f = %.0f\n", i, h_a[i], h_b[i], h_c[i]);
    }
    printf("...\n");
    for (int i = N-5; i < N; i++) {
        printf("h_c[%d] = %.0f + %.0f = %.0f\n", i, h_a[i], h_b[i], h_c[i]);
    }

    return 0;
}