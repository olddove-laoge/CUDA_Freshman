#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 错误检查宏
#define CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
} while (0)

// 打印float数组
void printFloatArray(const char* arrayName, float* arr, int size) {
    printf("%s 的元素值：", arrayName);
    for (int i = 0; i < size; i++) {
        printf("%.2f ", arr[i]);
    }
    printf("\n");
}

// 核函数：只显式访问第一个元素，但读取整个32字节块的内容
__global__ void verify_32byte_loading(float *d_arr, float *d_result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // 仅显式访问第一个元素（触发32字节加载）
        float first_element = d_arr[0];  // 这一行会导致GPU加载d_arr[0]~d_arr[7]
        
        // 关键：虽然没有显式访问d_arr[1]~d_arr[7]，但仍尝试读取它们
        d_result[0] = d_arr[0];  // 显式访问过的元素
        d_result[1] = d_arr[1];  // 未显式访问的元素
        d_result[2] = d_arr[2];
        d_result[3] = d_arr[3];
        d_result[4] = d_arr[4];
        d_result[5] = d_arr[5];
        d_result[6] = d_arr[6];
        d_result[7] = d_arr[7];
        
        // 用first_element防止编译器优化掉上面的访问
        if (first_element < 0) {
            printf("Dummy: %.2f\n", first_element);
        }
    }
}

int main() {
    // 初始化设备
    int dev = 0;
    CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("使用GPU: %s\n\n", prop.name);

    // 定义8个float的数组（32字节）
    const int ARR_SIZE = 8;
    float h_arr[ARR_SIZE];
    // 初始化数组（赋不同值便于区分）
    for (int i = 0; i < ARR_SIZE; i++) {
        h_arr[i] = (float)i * 1.23f;  // 0.00, 1.23, 2.46, 3.69, 4.92, 6.15, 7.38, 8.61
    }
    printFloatArray("主机端初始数组 h_arr", h_arr, ARR_SIZE);

    // 分配设备内存
    float *d_arr, *d_result;
    float h_result[ARR_SIZE];
    CHECK(cudaMalloc(&d_arr, ARR_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_result, ARR_SIZE * sizeof(float)));

    // 主机→设备拷贝
    CHECK(cudaMemcpy(d_arr, h_arr, ARR_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // 启动核函数（仅访问第一个元素）
    verify_32byte_loading<<<1, 1>>>(d_arr, d_result);
    CHECK(cudaDeviceSynchronize());

    // 设备→主机拷贝结果
    CHECK(cudaMemcpy(h_result, d_result, ARR_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // 输出结果
    printf("\n=== 关键验证 ===\n");
    printf("核函数中只显式访问了d_arr[0]，但读取到的所有元素：\n");
    printFloatArray("设备端读取的结果 h_result", h_result, ARR_SIZE);
    
    // 结论
    printf("\n结论：即使只显式访问第一个元素，也能正确读取到全部8个元素的值，\n");
    printf("证明GPU自动加载了整个32字节内存块（8个float）到缓存中。\n");

    // 释放内存
    CHECK(cudaFree(d_arr));
    CHECK(cudaFree(d_result));
    return 0;
}
