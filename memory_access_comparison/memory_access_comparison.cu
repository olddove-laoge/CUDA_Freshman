#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// 行主序访问: smem[threadIdx.y][threadIdx.x]
__global__ void rowMajorAccess(const float* input, float* output, int size) {
    __shared__ float smem[32][32];  // 32x32的共享内存数组
    
    // 计算全局索引
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int globalIdy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (globalIdx < size && globalIdy < size) {
        int index = globalIdy * size + globalIdx;
        
        // 加载数据到共享内存 (行主序)
        smem[threadIdx.y][threadIdx.x] = input[index];
        __syncthreads();  // 等待所有线程加载完成
        
        // 增加计算复杂度，使内存访问成为瓶颈
        float temp = 0.0f;
        for (int i = 0; i < 10; i++) {
            temp += smem[threadIdx.y][threadIdx.x] * (i + 1);
        }
        
        output[index] = temp;
    }
}

// 列主序访问: smem[threadIdx.x][threadIdx.y]
__global__ void columnMajorAccess(const float* input, float* output, int size) {
    __shared__ float smem[32][32];  // 32x32的共享内存数组
    
    // 计算全局索引
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int globalIdy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (globalIdx < size && globalIdy < size) {
        int index = globalIdy * size + globalIdx;
        
        // 加载数据到共享内存 (列主序)
        smem[threadIdx.x][threadIdx.y] = input[index];
        __syncthreads();  // 等待所有线程加载完成
        
        // 增加计算复杂度，使内存访问成为瓶颈
        float temp = 0.0f;
        for (int i = 0; i < 10; i++) {
            temp += smem[threadIdx.x][threadIdx.y] * (i + 1);
        }
        
        output[index] = temp;
    }
}

// 初始化数据
void initializeData(float* data, int size) {
    for (int i = 0; i < size * size; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
}

// 检查结果是否正确
bool checkResults(const float* input, const float* output, int size) {
    for (int i = 0; i < size * size; i++) {
        float expected = 0.0f;
        for (int j = 0; j < 10; j++) {
            expected += input[i] * (j + 1);
        }
        if (fabs(output[i] - expected) > 1e-5f) {
            return false;
        }
    }
    return true;
}

// 错误检查宏
#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "错误: %s 在文件 %s, 行号 %d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    const int size = 2048;  // 增大数组规模
    const int dataSize = size * size * sizeof(float);
    printf("测试数组大小: %dx%d\n", size, size);
    
    // 分配主机内存
    float* h_input = (float*)malloc(dataSize);
    float* h_output_row = (float*)malloc(dataSize);
    float* h_output_col = (float*)malloc(dataSize);
    
    if (!h_input || !h_output_row || !h_output_col) {
        fprintf(stderr, "主机内存分配失败\n");
        exit(EXIT_FAILURE);
    }
    
    // 初始化输入数据
    initializeData(h_input, size);
    
    // 分配设备内存
    float* d_input, *d_output_row, *d_output_col;
    CHECK(cudaMalloc(&d_input, dataSize));
    CHECK(cudaMalloc(&d_output_row, dataSize));
    CHECK(cudaMalloc(&d_output_col, dataSize));
    
    // 复制数据到设备
    CHECK(cudaMemcpy(d_input, h_input, dataSize, cudaMemcpyHostToDevice));
    
    // 配置线程块和网格大小
    dim3 blockDim(32, 32);  // 32x32的线程块
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x, 
                 (size + blockDim.y - 1) / blockDim.y);
    printf("线程块大小: %dx%d\n", blockDim.x, blockDim.y);
    printf("网格大小: %dx%d\n", gridDim.x, gridDim.y);
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    
    // 预热运行 - 确保准确测量
    rowMajorAccess<<<gridDim, blockDim>>>(d_input, d_output_row, size);
    columnMajorAccess<<<gridDim, blockDim>>>(d_input, d_output_col, size);
    CHECK(cudaDeviceSynchronize());
    
    // 测试行主序访问性能
    CHECK(cudaEventRecord(start));
    rowMajorAccess<<<gridDim, blockDim>>>(d_input, d_output_row, size);
    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));
    
    float rowTime;
    CHECK(cudaEventElapsedTime(&rowTime, start, end));  // 单位为毫秒
    
    // 测试列主序访问性能
    CHECK(cudaEventRecord(start));
    columnMajorAccess<<<gridDim, blockDim>>>(d_input, d_output_col, size);
    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));
    
    float colTime;
    CHECK(cudaEventElapsedTime(&colTime, start, end));  // 单位为毫秒
    
    // 复制结果回主机
    CHECK(cudaMemcpy(h_output_row, d_output_row, dataSize, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_output_col, d_output_col, dataSize, cudaMemcpyDeviceToHost));
    
    // 验证结果
    bool rowCorrect = checkResults(h_input, h_output_row, size);
    bool colCorrect = checkResults(h_input, h_output_col, size);
    
    // 输出结果
    printf("\n行主序访问测试: %s\n", rowCorrect ? "成功" : "失败");
    printf("列主序访问测试: %s\n", colCorrect ? "成功" : "失败");
    printf("行主序访问时间: %.6f 毫秒\n", rowTime);
    printf("列主序访问时间: %.6f 毫秒\n", colTime);
    printf("性能差异倍数: %.2f 倍\n", colTime / rowTime);
    
    // 清理资源
    free(h_input);
    free(h_output_row);
    free(h_output_col);
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output_row));
    CHECK(cudaFree(d_output_col));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
    
    return 0;
}
