#include <cuda_runtime.h>
#include "d_helper.cu"

__global__ void gemm_v1(d_Matrix A, d_Matrix B, d_Matrix C) {
    constexpr int32_t BlkProcessSize = BLOCK_SIZE / 2;
    __shared__ float shm_A[BlkProcessSize][BlkProcessSize]; // 4blocks, each block read form global to shared and compute twice.
    __shared__ float shm_B[BlkProcessSize][BlkProcessSize]; 
    __shared__ float res[BlkProcessSize][BlkProcessSize];
    int32_t bkx = blockIdx.x;
    int32_t bky = blockIdx.y;
    int32_t thx = threadIdx.x; // 8x8threads in one block
    int32_t thy = threadIdx.y;
    int32_t idx = thy * blockDim.x + thx;

    int32_t bk_start_row  = bkx * BlkProcessSize; // the first row of A which processed by this block
    int32_t bk_start_col  = bky * BlkProcessSize; // the first col of B which processed by this block
    int32_t thd_r_size = BlkProcessSize / blockDim.x;
    for (int32_t i = 0; i < 2; ++i) {
        //read from global to shared mem
        //each thread read 4 data 
        int32_t shmA_start_col = i * BlkProcessSize; int32_t shmB_start_row = i * BlkProcessSize;
        int32_t thd_start_row = bk_start_row + thx * thd_r_size;
        int32_t thd_start_col = bk_start_col + thy * thd_r_size;
        for (int32_t x = 0; x < thd_r_size; ++x) {
            for (int32_t y = 0; y < thd_r_size; ++y) {
                shm_A[thx * thd_r_size + x][thy * thd_r_size + y] = A.data[ELE_IDX(thd_start_row + thx * thd_r_size + x, shmA_start_col + thy * thd_r_size + y, A.width)];
                shm_B[thx * thd_r_size + x][thy * thd_r_size + y] = B.data[ELE_IDX(shmB_start_row + thx * thd_r_size + x, thd_start_col + thy * thd_r_size + y, B.width)];
            }
        }
        __syncthreads();
        //calculate 
        if (idx < BlkProcessSize) { // the threads(idx from 0-15) calculate
            for (int32_t x = 0; x < BlkProcessSize; ++x) {
                for (int32_t y = 0; y < BlkProcessSize; ++y) {
                    res[x][y] += shm_A[x][idx] * shm_B[idx][y];
                }
            }
        }
        __syncthreads();
    }
    // store res to C
    for (int32_t x = 0; x < 2; ++x) {
        for (int32_t y = 0; y < 2; ++y) {
            C.data[ELE_IDX(bk_start_row + thx * thd_r_size + x, bk_start_col + thy * thd_r_size + y, C.width)] = res[thx * thd_r_size + x][thy * thd_r_size + y];
        }
    }
}