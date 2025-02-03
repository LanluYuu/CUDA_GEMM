#include <cuda_runtime.h>
#include "d_helper.cu"

#define FLOAT4(arr) reinterpret_cast<float4*>(&arr)[0]

__global__ void gemm_v4(float* A, float* B, float* C, int32_t m, int32_t k, int32_t n) { 
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

    __shared__ float shm_A[BK][BN];
    __shared__ float shm_B[BK][BM];
    float reg_A[Tcs] = {0.0f};
    float reg_B[Tcs] = {0.0f};
    float res[Tcs][Tcs] = {{0.0f}};
    const int32_t start_row = bky * BM;
    const int32_t start_col = bkx * BN;
    #pragma unroll
    for (int32_t stride = 0; stride < k; stride += BK) {
        float4 tmp = FLOAT4(A[ELE_IDX((start_row + (tid * Trs / BK)), (stride + (tid * Trs % BK)), k)]);
        shm_A[(tid * Trs % BK) + 0][tid * Trs / BK] = tmp.x;
        shm_A[(tid * Trs % BK) + 1][tid * Trs / BK] = tmp.y;
        shm_A[(tid * Trs % BK) + 2][tid * Trs / BK] = tmp.z;
        shm_A[(tid * Trs % BK) + 3][tid * Trs / BK] = tmp.w;
        
        FLOAT4(shm_B[tid * 4 / BN][tid * 4 % BN]) = FLOAT4(B[ELE_IDX((stride + tid * 4 / BN), (start_col + tid * 4 % BN), n)]);
        __syncthreads();
        /*if(bkx == 0 && bky == 0 && thx == 1 && thy == 0) {
            printf("shmB:\n");
            for (int32_t i = 0; i < BK; ++i) {
                for (int32_t j = 0; j < BM; ++j) {
                    printf("%f, ", shm_B[i][j]);
                }        
                printf("\n");
            }
            printf("shmA:\n");
            for (int32_t i = 0; i < BK; ++i) {
                for (int32_t j = 0; j < BM; ++j) {
                    printf("%f, ", shm_A[i][j]);
                }        
                printf("\n");
            }
        }*/
        #pragma unroll
        for (int32_t dotIdx = 0; dotIdx < BK; ++dotIdx) {
            #pragma unroll
            for (int32_t i = 0; i < Tcs; ++i) {
                reg_A[i] = shm_A[dotIdx][Tcs * thy + i];
            }
            #pragma unroll
            for (int32_t i = 0; i < Tcs; ++i) {
                reg_B[i] = shm_B[dotIdx][Tcs * thx + i];
            }
            #pragma unroll
            for (int32_t x = 0; x < Tcs; ++x) {
                #pragma unroll
                for (int32_t y = 0; y < Tcs; ++y) {
                    res[x][y] += reg_A[x] * reg_B[y];
                }
            }
        }
        __syncthreads();
        //if (bkx == 0 && bky == 0 && thx == 1 && thy == 0) {
            /*printf("\nregA:\n");
            for (int32_t i = 0; i < Tcs; ++i) {
                printf("%f,", reg_A[i]);
            }
            printf("\nregB:\n");
            for (int32_t i = 0; i < Tcs; ++i) {
                printf("%f,", reg_B[i]);
            }*/
        /*    printf("\nres:\n");
            for (int32_t i = 0; i < Tcs; ++i) {
                for (int32_t j = 0; j < Tcs; ++j) {
                    printf("%f,", res[i][j]);
                }
                printf("\n");
            }
        }*/
    }
    #pragma unroll
    for (int32_t x = 0; x < Tcs; ++x) {
        #pragma unroll
        for(int32_t y = 0; y < Tcs / 4; ++y) {
            FLOAT4(C[ELE_IDX((start_row + thy * Tcs + x), (start_col + thx * Tcs + y * 4), n)]) = FLOAT4(res[x][y * 4]);
        }
    }
}