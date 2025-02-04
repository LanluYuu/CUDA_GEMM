#include <cuda_runtime.h>
#include "d_helper.cu"

constexpr int32_t threadNums = 128;
constexpr int32_t warpSize = 32;
constexpr int32_t BM = 64;
constexpr int32_t BN = 128;
constexpr int32_t BK = 8;
constexpr int32_t Trs = 4;
constexpr int32_t wM = 32;
constexpr int32_t wN = 64;
constexpr int32_t wIterX = 2;
constexpr int32_t wIterY = 2;
constexpr int32_t TM = 4;
constexpr int32_t TN = 4;

__global__ void gemm_v5(float* A, float* B, float* C, int32_t m, int32_t k, int32_t n) {
    const int32_t bkx = blockIdx.x;
    const int32_t bky = blockIdx.y;
    const int32_t thx = threadIdx.x;
    const int32_t start_row = BM * bky;
    const int32_t start_col = BN * bkx;
    const int32_t warpId = thx / warpSize;
    const int32_t wIdX = warpId / (BN / wN);
    const int32_t wIdY = warpId % (BN / wN);
    const int32_t wSubM = wM / wIterY; // 16
    const int32_t wSubN = wN / wIterX; // 32
    const int32_t thYinWarp = thx % warpSize / (wSubN / TN);
    const int32_t thXinWarp = thx % warpSize % (wSubN / TN);
    const int32_t BstrideY = 4;
    __shared__ float shm_A[BK][BM];
    __shared__ float shm_B[BK][BN];
    float reg_A[wIterY * TM] = {0.0f};
    float reg_B[wIterX * TN] = {0.0f};
    float res[wIterY * TM][wIterX * TN] = {{0.0f}};
    // read from G_mem    
    #pragma unroll 
    for (int32_t stride = 0; stride < k; stride += BK) {
  
        float4 tmp = FLOAT4(A[ELE_IDX((start_row + thx * Trs / BK), (stride + thx * Trs % BK), k)]);
        shm_A[thx * Trs % BK + 0][thx * Trs / BK] = tmp.x;
        shm_A[thx * Trs % BK + 1][thx * Trs / BK] = tmp.y;
        shm_A[thx * Trs % BK + 2][thx * Trs / BK] = tmp.z;
        shm_A[thx * Trs % BK + 3][thx * Trs / BK] = tmp.w;

        #pragma unroll 
        for (int32_t offset = 0; offset < BK; offset += BstrideY) {
            FLOAT4(shm_B[thx * Trs / BN + offset][thx * Trs % BN]) = FLOAT4(B[ELE_IDX((stride + thx * Trs / BN + offset), (start_col + thx * Trs % BN), n)]);
        }

        __syncthreads();

        #pragma unroll 
        for (int32_t dotIdx = 0; dotIdx < BK; ++dotIdx) {
            #pragma unroll 
            for (int32_t i = 0; i < wIterY; ++i) {
                #pragma unroll 
                for (int32_t j = 0; j < TM; ++j) {
                    reg_A[i * TM + j] = shm_A[dotIdx][wIdY * wM + i * wSubM + TM * thYinWarp + j];
                }
            }
            #pragma unroll 
            for (int32_t i = 0; i < wIterX; ++i) {
                #pragma unroll 
                for (int32_t j = 0; j < TN; ++j) {
                    reg_B[i * TN + j] = shm_B[dotIdx][wIdX * wN + i * wSubN + TN * thXinWarp + j];
                }
            }
            #pragma unroll 
            for (int32_t x = 0; x < wIterX; ++x) {
                #pragma unroll 
                for (int32_t y = 0; y < wIterY; ++y) {
                    #pragma unroll 
                    for (int32_t i = 0; i < TM; ++i) {
                        #pragma unroll 
                        for (int32_t j = 0; j < TN; ++j) {
                            res[x * TM + i][y * TN + j] += reg_A[x * TM + i] * reg_B[y * TN + j];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    // store to G_mem
    int32_t C_row; int32_t C_col;
    #pragma unroll 
    for (int32_t x = 0; x < wIterX; ++x) {
        #pragma unroll 
        for (int32_t y = 0; y < wIterY; ++y) {
            #pragma unroll 
            for (int32_t i = 0; i < TM; ++i) {
                C_row = start_row + wIdY * wM + y * wSubM + thYinWarp * TM + i;
                #pragma unroll 
                for (int32_t j = 0; j < TN / 4; ++j) {
                    C_col = start_col + wIdX *wN + x * wSubN + thXinWarp * TN + 4 * j;
                    FLOAT4(C[ELE_IDX(C_row, C_col, n)]) = FLOAT4(res[y * TM + i][x * TN + 4 * j]);
                }
            }
        }
    }
}  