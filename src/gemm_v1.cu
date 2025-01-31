#include <cuda_runtime.h>
#include "d_helper.cu"

#define MAX_SHM_SIZE 64

__global__ void gemm_v1(float* A, float* B, float* C, int32_t m, int32_t k, int32_t n) {
    constexpr int32_t read_per_thread = 2; //each thread read 16 data
    const int32_t loop_times = k / MAX_SHM_SIZE; //each block read and calculate 4096/64=64 times
    __shared__ float shm_A[MAX_SHM_SIZE][MAX_SHM_SIZE]; 
    __shared__ float shm_B[MAX_SHM_SIZE][MAX_SHM_SIZE]; 
    int32_t bkx = blockIdx.x;
    int32_t bky = blockIdx.y;
    int32_t thx = threadIdx.x; 
    int32_t thy = threadIdx.y;

    const int32_t shm_A_len = MAX_SHM_SIZE, shm_B_len = MAX_SHM_SIZE;
    int32_t bk_start_row  = bky * shm_A_len; // the first row of A for this block
    int32_t bk_start_col  = bkx * shm_B_len; // the first col of B for this block

    float res[read_per_thread][read_per_thread] = {0}; 

    #pragma unroll
    for (int32_t i = 0; i < loop_times; ++i) {
        //read from global to shared mem
        int32_t start_col = i * shm_A_len; int32_t start_row = i * shm_B_len;
        #pragma unroll
        for (int32_t y = 0; y < read_per_thread; ++y) {
            for (int32_t x = 0; x < read_per_thread; ++x) {
                    int32_t A_row = bk_start_row + thy * read_per_thread + y;
                    int32_t A_col = start_col + thx * read_per_thread + x;
                    int32_t B_row = start_row + thy * read_per_thread + y;
                    int32_t B_col = bk_start_col + thx * read_per_thread + x;
                    shm_A[thy * read_per_thread + y][thx * read_per_thread + x] = A[ELE_IDX(A_row, A_col, k)];
                    shm_B[thy * read_per_thread + y][thx * read_per_thread + x] = B[ELE_IDX(B_row, B_col, n)];
            }
        }
        __syncthreads();
        //calculate 
        #pragma unroll
        for (int32_t a = 0; a < read_per_thread; ++a) {
            for (int32_t b = 0; b < read_per_thread; ++b) {
                for (int32_t z = 0; z < shm_A_len; ++z) {
                    res[a][b] += shm_A[thy * read_per_thread + a][z]
                                * shm_B[z][thx * read_per_thread + b];
                }
            }
        }
        __syncthreads();
    }
    // store res to C
    #pragma unroll
    for (int32_t x = 0; x < read_per_thread; ++x) {
        for (int32_t y = 0; y < read_per_thread; ++y) {
            int32_t C_row = bk_start_row + thy * read_per_thread + x;
            int32_t C_col = bk_start_col + thx * read_per_thread + y;
            C[ELE_IDX(C_row, C_col, n)] = res[x][y];
        }
    }
}