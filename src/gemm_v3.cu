#include <cuda_runtime.h>
#include "d_helper.cu"

#define MAX_SHM_SIZE 32

/*__global__ void gemm_v3(float* A, float* B, float* C, int32_t m, int32_t k, int32_t n) {
    constexpr int32_t eleNumPerThread = 4; // each thread read 4 data from Gmemory
    const int32_t times = k / MAX_SHM_SIZE;
    __shared__ float shm_A[2][MAX_SHM_SIZE * (MAX_SHM_SIZE + 1)]; // double buffer
    __shared__ float shm_B[2][MAX_SHM_SIZE * (MAX_SHM_SIZE + 1)];

    int32_t bkx = blockIdx.x;
    int32_t bky = blockIdx.y;
    int32_t thx = threadIdx.x;
    int32_t thy = threadIdx.y;
    int32_t thread_id = thy * blockDim.x + thx;

    double res[2][2] = {{0}};

    int32_t bk_start_row = bky * MAX_SHM_SIZE;
    int32_t bk_start_col = bkx * MAX_SHM_SIZE;
    // firstly read from Gmemory, fill the stage0 shared memory
    int32_t stage = 0;
    int32_t shm_row = (thread_id * eleNumPerThread) / 32;
    int32_t shm_col = (thread_id * eleNumPerThread) % 32; 
    int32_t A_row = bk_start_row + shm_row;
    int32_t A_col = 0 + shm_col; // because of first time read
    int32_t B_row = 0 + shm_row;
    int32_t B_col = bk_start_col + shm_col;

    #pragma unroll
    for (int32_t i = 0; i < eleNumPerThread; ++i) {
        if (A_row < m && A_col + i < k) {
            shm_A[stage][ELE_IDX(shm_row, shm_col + i, (MAX_SHM_SIZE + 1))] = A[ELE_IDX(A_row, A_col + i, k)];
        }
        if (B_row < k && B_col + i < n) 
            shm_B[stage][ELE_IDX(shm_row, shm_col + i, (MAX_SHM_SIZE + 1))] = B[ELE_IDX(B_row, B_col + i, n)];
    }
    __syncthreads();
    // loop: read-->calculate
    #pragma unroll
    for (int32_t i = 0; i < times; ++i) { // last time no read only calculate
        int32_t next_stage = (stage + 1) % 2;
        if (i < times - 1) {
            A_col = (i + 1) * MAX_SHM_SIZE + shm_col;
            B_row = (i + 1) * MAX_SHM_SIZE + shm_row;
            #pragma unroll
            for (int32_t j = 0; j < eleNumPerThread; ++j) {
                if (A_row < m && A_col + j < k) 
                    shm_A[next_stage][ELE_IDX(shm_row, shm_col + j, (MAX_SHM_SIZE + 1))] = A[ELE_IDX(A_row, A_col + j, k)];
                if (B_row < k && B_col + j < n) 
                    shm_B[next_stage][ELE_IDX(shm_row, shm_col + j, (MAX_SHM_SIZE + 1))] = B[ELE_IDX(B_row, B_col + j, n)];
            }
        }
        __syncthreads();
        // read to next stage done, now calculate
        #pragma unroll
        for (int32_t x = 0; x < 2; ++x) {
            #pragma unroll
            for (int32_t y = 0; y < 2; ++y) {
                #pragma unroll
                for (int32_t z = 0; z < MAX_SHM_SIZE; ++z) {
                    res[x][y] += shm_A[stage][ELE_IDX((2 * thy + x), z, (MAX_SHM_SIZE + 1))] * 
                                shm_B[stage][ELE_IDX(z, (2 * thx + y), (MAX_SHM_SIZE + 1))];
                }
            }
        }
        stage = next_stage;
    }
    
    // store to G_memory
    #pragma unroll
    for (int32_t x = 0; x < 2; ++x) {
        #pragma unroll
        for (int32_t y = 0; y < 2; ++y) {
            int32_t C_row = bk_start_row + 2 * thy + x;
            int32_t C_col = bk_start_col + 2 * thx + y;
            if (C_row < m && C_col < n)
                C[ELE_IDX(C_row, C_col, n)] = res[x][y];
        }
    }
    __syncthreads();
}*/

__global__ void gemm_v3_1(float* A, float* B, float* C, int32_t m, int32_t k, int32_t n) { // inherit form gemm_v2_1
// each thread read 4 * 2 data from G_mem
    constexpr int32_t BM  = 128; 
    constexpr int32_t BN  = 128;
    constexpr int32_t BK  = 8;
    constexpr int32_t Trs = 4; // thread read 4 data 
    constexpr int32_t Tcs = 8; // thread calculte 8x8 
    const int32_t bkx = blockIdx.x;
    const int32_t bky = blockIdx.y;
    const int32_t thx = threadIdx.x;
    const int32_t thy = threadIdx.y;
    const int32_t tid = thy * blockDim.x + thx;
    
        //printf("bkx:%d, bky:%d", bkx, bky);
    
    const int32_t start_row = bky * BM;
    const int32_t start_col = bkx * BN;
    __shared__ float shm_A[BM][BK];
    __shared__ float shm_B[BK][BN];

    float reg_A[Tcs]    = {0.0f};
    float reg_B[Tcs]    = {0.0f};
    float res[Tcs][Tcs] = {{0.0f}};

    #pragma unroll
    for (int32_t stride = 0; stride < k; stride += BK) {
        #pragma unroll
        for (int32_t i = 0; i < Trs; ++i) {
            shm_A[(tid / 8) * 4 + i][(tid % 8)]     = A[ELE_IDX((start_row + (tid / 8) * 4 + i), (stride + (tid % 8)), k)];
        }
        #pragma unroll 
        for (int32_t i = 0; i < Trs; ++i) {
            shm_B[(tid / 128) * 4 + i][(tid % 128)] = B[ELE_IDX((stride + (tid / 128) * 4 + i), (start_col + (tid % 128)), n)];
        }
        __syncthreads();
        /*if (bkx < 3 && bky < 3 && thx < 3 && thy < 3) {
            printf("\nshmA:\n");
            for (int32_t i = 0; i < BM; ++i) {
                for (int32_t j = 0; j < BK; ++j) {
                    printf("%f,", shm_A[i][j]);                    
                }
                printf("\n");
            }
        }*/
        #pragma unroll
        for(int32_t dotIdx = 0; dotIdx < BK; ++dotIdx) {
            #pragma unroll
            for (int32_t i = 0; i < Tcs; ++i) {
                reg_A[i] = shm_A[Tcs * thy + i][dotIdx];
            }
            #pragma unroll 
            for (int32_t i = 0; i < Tcs; ++i) {
                reg_B[i] = shm_B[dotIdx][Tcs * thx + i];
            }
            #pragma unroll
            for (int32_t i = 0; i < Tcs; ++i) {
                #pragma unroll
                for (int32_t j = 0; j < Tcs; ++j) {
                    res[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int32_t i = 0; i < Tcs; ++i) {
        #pragma unroll
        for (int32_t j = 0; j < Tcs; ++j) {
            C[ELE_IDX((start_row + thy * Tcs + i), (start_col + thx * Tcs + j), n)] = res[i][j];
        }
    }
}