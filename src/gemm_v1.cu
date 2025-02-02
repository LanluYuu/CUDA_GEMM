#include <cuda_runtime.h>
#include "d_helper.cu"

#define MAX_SHM_SIZE 32

__global__ void gemm_v1(float* A, float* B, float* C, int32_t m, int32_t k, int32_t n) {
    __shared__ float shm_A[MAX_SHM_SIZE * MAX_SHM_SIZE];
    __shared__ float shm_B[MAX_SHM_SIZE * MAX_SHM_SIZE];
    int32_t bkx = blockIdx.x;
    int32_t bky = blockIdx.y;
    int32_t thx = threadIdx.x;
    int32_t thy = threadIdx.y;

    float res = 0.0f;
    int32_t start_row = bky * MAX_SHM_SIZE;
    int32_t start_col = bkx * MAX_SHM_SIZE;
    #pragma unroll
    for(int32_t stride = 0; stride < k; stride += MAX_SHM_SIZE) {
        shm_A[ELE_IDX(thy, thx, MAX_SHM_SIZE)] = A[ELE_IDX((start_row + thy), (stride + thx), k)];
        shm_B[ELE_IDX(thy, thx, MAX_SHM_SIZE)] = B[ELE_IDX((stride + thy), (start_col + thx), n)];     
        __syncthreads();
        #pragma unroll
        for (int32_t z = 0; z < MAX_SHM_SIZE; ++z) {
            res += shm_A[ELE_IDX(thy, z, MAX_SHM_SIZE)] * shm_B[ELE_IDX(z, thx, MAX_SHM_SIZE)];
        }
        __syncthreads();
    }

    C[ELE_IDX((start_row + thy), (start_col + thx), n)] = res;
}