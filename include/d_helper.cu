#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#define BLOCK_SIZE 32
#define ELE_IDX(x, y, col) (x * col + y)

struct d_Matrix { 
    int32_t height;
    int32_t width;
    float* data;

    d_Matrix(int32_t row, int32_t col) : height(row), width(col) {
        cudaError_t err = cudaMalloc(&data, height * width * sizeof(float));
        if (err != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        }
    }

    ~d_Matrix() {
        cudaFree(data);
    }
};
//device function
__device__ float d_GetMatrixElement(const d_Matrix& A, int32_t row, int32_t col) {
    return A.data[ELE_IDX(row, col, A.width)];
}

__device__ void d_SetMatrixElement(const d_Matrix& A, int32_t row, int32_t col, float val) {
    printf("val:%f", val);
    A.data[ELE_IDX(row, col, A.width)] = val;
    printf(", C.data:%f\n", A.data[ELE_IDX(row, col, A.width)]);
    return;
}