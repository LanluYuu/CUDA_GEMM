#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#define BLOCK_SIZE 4096
#define ELE_IDX(x, y, col) (x * col + y)
#define FLOAT4(arr) reinterpret_cast<float4*>(&arr)[0]
#define HALF4(arr) (reinterpret_cast<half4*>(&arr)[0])
#define HALF8(arr) (reinterpret_cast<half8*>(&arr)[0])
#define WARMUPT 10
#define RUNTIMES 100

struct half4 {
    half x, y, z, w;
    // __device__ __host__ half4() = deflaut;
    __device__ __host__ half4(half _x, half _y, half _z, half _w) :
        x(_x), y(_y), z(_z), w(_w) {};
};

struct half8 {
    half x, y, z, w, a, b, c, d;
    // __device__ __host__ half4() = deflaut;
    __device__ __host__ half8(half _x, half _y, half _z, half _w, half _a, half _b, half _c, half _d) :
        x(_x), y(_y), z(_z), w(_w), a(_a), b(_b), c(_c), d(_d) {};
};

// struct d_Matrix { 
//     int32_t height;
//     int32_t width;
//     float* data;
//     d_Matrix(int32_t row, int32_t col) : height(row), width(col) {
//         cudaError_t err = cudaMalloc(&data, height * width * sizeof(float));
//         if (err != cudaSuccess || data == nullptr) {
//             printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
//         }
//         printf("new matrix addr:0x%lX", &data);
//     }

//     ~d_Matrix() {
//         cudaFree(data);
//     }
// };
// //device function
// __device__ float d_GetMatrixElement(const d_Matrix& A, int32_t row, int32_t col) {
//     return A.data[ELE_IDX(row, col, A.width)];
// }

// __device__ void d_SetMatrixElement(d_Matrix& A, int32_t row, int32_t col, float val) {
//     //printf("val:%f", val);
//     A.data[ELE_IDX(row, col, A.width)] = val;
//     printf("\nkernel C_d matrix data addr:0x%lX", &A.data);
//     //printf(", C.data:%f\n", A.data[ELE_IDX(row, col, A.width)]);
//     return;
// }

// debug
        //         if (bkx == 10 && bky == 0 && thx == 64 && k_stride == 0)    
        // {printf("shared mem A:\n");
        //     for (int32_t i = 0; i < bM; ++i) {
        //         for (int32_t j = 0; j < bK; ++j) {
        //             printf("%f,", __half2float(shm_A[i][j]));
        //         }
        //         printf("\n");
        //     }
        //     printf("shared mem B:\n");
        //     for (int32_t i = 0; i < bK; ++i) {
        //         for (int32_t j = 0; j < bN; ++j) {
        //             printf("%f,", __half2float(shm_B[i][j]));
        //         }
        //         printf("\n");
        //     }}