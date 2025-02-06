// double buffer
#include <cuda_runtime.h>
#include "d_helper.cu"

__global__ void gemm_v6(float* A, float* B, float* C, int32_t m, int32_t k, int32_t n) { 
    constexpr int32_t threadNums = 128;
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
    constexpr int32_t wSubM = wM / wIterY; // 16
    constexpr int32_t wSubN = wN / wIterX; // 32
    constexpr int32_t BstrideY = 4;

    const int32_t bkx = blockIdx.x;
    const int32_t bky = blockIdx.y;
    const int32_t thx = threadIdx.x;
    const int32_t start_row = BM * bky;
    const int32_t start_col = BN * bkx;
    const int32_t warpId = thx / warpSize;
    //printf("warpId:%d, ", warpId);
    const int32_t wIdX = warpId / (BN / wN);
    const int32_t wIdY = warpId % (BN / wN);
    const int32_t thYinWarp = thx % warpSize / (wSubN / TN);
    const int32_t thXinWarp = thx % warpSize % (wSubN / TN);
    __shared__ float shm_A[2][BK][BM];
    __shared__ float shm_B[2][BK][BN];
    float reg_A[2][wIterY * TM] = {{0.0f}};
    float reg_B[2][wIterX * TN] = {{0.0f}};
    float res[wIterY * TM][wIterX * TN] = {{0.0f}};
    int32_t shm_stage = 0;
    int32_t reg_stage = 0;
    // prefetch from G_mem and S_mem
    float4 tmp = FLOAT4(A[ELE_IDX((start_row + thx * Trs / BK), (thx * Trs % BK), k)]);
    shm_A[shm_stage][thx * Trs % BK + 0][thx * Trs / BK] = tmp.x;
    shm_A[shm_stage][thx * Trs % BK + 1][thx * Trs / BK] = tmp.y;
    shm_A[shm_stage][thx * Trs % BK + 2][thx * Trs / BK] = tmp.z;
    shm_A[shm_stage][thx * Trs % BK + 3][thx * Trs / BK] = tmp.w;
    for (int32_t offset = 0; offset < BK; offset += BstrideY) {
        FLOAT4(shm_B[shm_stage][thx * Trs / BN + offset][thx * Trs % BN]) = FLOAT4(B[ELE_IDX((thx * Trs / BN + offset), (start_col + thx * Trs % BN), n)]);
    }
    __syncthreads();
    
    // read from G_mem    
    #pragma unroll 
    for (int32_t stride = BK; stride < k + 1; stride += BK) {
        // load next stage data from G_mem to S_mem
        int32_t preShmStage = shm_stage;
        shm_stage = (shm_stage + 1) % 2;
        if (stride < k) {
            tmp = FLOAT4(A[ELE_IDX((start_row + thx * Trs / BK), (stride + thx * Trs % BK), k)]);
            shm_A[shm_stage][thx * Trs % BK + 0][thx * Trs / BK] = tmp.x;
            shm_A[shm_stage][thx * Trs % BK + 1][thx * Trs / BK] = tmp.y;
            shm_A[shm_stage][thx * Trs % BK + 2][thx * Trs / BK] = tmp.z;
            shm_A[shm_stage][thx * Trs % BK + 3][thx * Trs / BK] = tmp.w;

            #pragma unroll 
            for (int32_t offset = 0; offset < BK; offset += BstrideY) {
                FLOAT4(shm_B[shm_stage][thx * Trs / BN + offset][thx * Trs % BN]) = FLOAT4(B[ELE_IDX((stride + thx * Trs / BN + offset), (start_col + thx * Trs % BN), n)]);
            }
        }

        // read dotIdx=0 to register
        #pragma unroll
        for (int32_t i = 0; i < wIterX; ++i) {
            FLOAT4(reg_B[reg_stage][i * TN]) = FLOAT4(shm_B[preShmStage][0][wIdX * wN + i * wSubN + thXinWarp * TN]);
            // #pragma unroll 
            // for (int32_t j = 0; j < TN; ++j) {
            //     reg_B[reg_stage][i * TN + j] = shm_B[preShmStage][0][wIdX * wN + i * wSubN + TN * thXinWarp + j];
            // }
        }
        #pragma unroll
        for (int32_t i = 0; i < wIterY; ++i) {
            FLOAT4(reg_A[reg_stage][i * TM]) = FLOAT4(shm_A[preShmStage][0][wIdY * wM + i * wSubM + thYinWarp * TM]);
            // #pragma unroll 
            // for (int32_t j = 0; j < TM; ++j) {
            //     reg_A[reg_stage][i * TM + j] = shm_A[preShmStage][0][wIdY * wM + i * wSubM + TM * thYinWarp + j];
            // }
        }
        __syncthreads();

        #pragma unroll 
        for (int32_t dotIdx = 1;  dotIdx < BK + 1; ++dotIdx) {
            int32_t preRegStage = reg_stage;
            reg_stage = (reg_stage + 1) % 2;
            if (dotIdx < BK) {
                #pragma unroll 
                for (int32_t i = 0; i < wIterY; ++i) {
                    FLOAT4(reg_A[reg_stage][i * TM]) = FLOAT4(shm_A[preShmStage][dotIdx][wIdY * wM + i * wSubM + thYinWarp * TM]);
                    // #pragma unroll 
                    // for (int32_t j = 0; j < TM; ++j) {
                    //     reg_A[reg_stage][i * TM + j] = shm_A[preShmStage][dotIdx][wIdY * wM + i * wSubM + TM * thYinWarp + j];
                    // }
                }
                #pragma unroll 
                for (int32_t i = 0; i < wIterX; ++i) {
                    FLOAT4(reg_B[reg_stage][i * TN]) = FLOAT4(shm_B[preShmStage][dotIdx][wIdX * wN + i * wSubN + thXinWarp * TN]);
                    // #pragma unroll 
                    // for (int32_t j = 0; j < TN; ++j) {
                    //     reg_B[reg_stage][i * TN + j] = shm_B[preShmStage][dotIdx][wIdX * wN + i * wSubN + TN * thXinWarp + j];
                    // }
                }
            }

            #pragma unroll 
            for (int32_t y = 0; y < wIterY; ++y) {
                #pragma unroll 
                for (int32_t x = 0; x < wIterX; ++x) {
                    #pragma unroll 
                    for (int32_t i = 0; i < TM; ++i) {
                        #pragma unroll 
                        for (int32_t j = 0; j < TN; ++j) {
                            res[y * TM + i][x * TN + j] += reg_A[preRegStage][y * TM + i] * reg_B[preRegStage][x * TN + j];
                        }
                    }
                }
            }
            __syncthreads();
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

__global__ void gemm_v6_1(float* A, float* B, float* C, int32_t m, int32_t k, int32_t n) { 
    // for this config : 14.5601TFLOPS
    constexpr int32_t threadNums = 128;
    constexpr int32_t BM = 32;
    constexpr int32_t BN = 64;
    constexpr int32_t BK = 16;
    constexpr int32_t Trs = 4;
    constexpr int32_t wM = 16;
    constexpr int32_t wN = 32;
    constexpr int32_t TM = 4;
    constexpr int32_t TN = 4;
    constexpr int32_t BstrideY = 8;

    const int32_t bkx = blockIdx.x;
    const int32_t bky = blockIdx.y;
    const int32_t thx = threadIdx.x;
    const int32_t start_row = BM * bky;
    const int32_t start_col = BN * bkx;
    const int32_t warpId = thx / warpSize;
    //printf("warpId:%d, ", warpId);
    const int32_t wIdX = warpId / (BN / wN);
    const int32_t wIdY = warpId % (BN / wN);
    const int32_t thYinWarp = thx % warpSize / (wN / TN);
    const int32_t thXinWarp = thx % warpSize % (wN / TN);
    __shared__ float shm_A[2][BK][BM];
    __shared__ float shm_B[2][BK][BN];
    float reg_A[2][TM] = {{0.0f}};
    float reg_B[2][TN] = {{0.0f}};
    float res[TM][TN] = {{0.0f}};
    int32_t shm_stage = 0;
    int32_t reg_stage = 0;
    // prefetch from G_mem and S_mem
    float4 tmp = FLOAT4(A[ELE_IDX((start_row + thx * Trs / BK), (thx * Trs % BK), k)]);
    shm_A[shm_stage][thx * Trs % BK + 0][thx * Trs / BK] = tmp.x;
    shm_A[shm_stage][thx * Trs % BK + 1][thx * Trs / BK] = tmp.y;
    shm_A[shm_stage][thx * Trs % BK + 2][thx * Trs / BK] = tmp.z;
    shm_A[shm_stage][thx * Trs % BK + 3][thx * Trs / BK] = tmp.w;
    for (int32_t offset = 0; offset < BK; offset += BstrideY) {
        FLOAT4(shm_B[shm_stage][thx * Trs / BN + offset][thx * Trs % BN]) = FLOAT4(B[ELE_IDX((thx * Trs / BN + offset), (start_col + thx * Trs % BN), n)]);
    }
    __syncthreads();
    
    // read from G_mem    
    #pragma unroll 
    for (int32_t stride = BK; stride < k + 1; stride += BK) {
        // load next stage data from G_mem to S_mem
        int32_t preShmStage = shm_stage;
        shm_stage = (shm_stage + 1) % 2;
        if (stride < k) {
            tmp = FLOAT4(A[ELE_IDX((start_row + thx * Trs / BK), (stride + thx * Trs % BK), k)]);
            shm_A[shm_stage][thx * Trs % BK + 0][thx * Trs / BK] = tmp.x;
            shm_A[shm_stage][thx * Trs % BK + 1][thx * Trs / BK] = tmp.y;
            shm_A[shm_stage][thx * Trs % BK + 2][thx * Trs / BK] = tmp.z;
            shm_A[shm_stage][thx * Trs % BK + 3][thx * Trs / BK] = tmp.w;

            #pragma unroll 
            for (int32_t offset = 0; offset < BK; offset += BstrideY) {
                FLOAT4(shm_B[shm_stage][thx * Trs / BN + offset][thx * Trs % BN]) = FLOAT4(B[ELE_IDX((stride + thx * Trs / BN + offset), (start_col + thx * Trs % BN), n)]);
            }
        }

        // read dotIdx=0 to register
        //FLOAT4(reg_B[reg_stage][i * TN]) = FLOAT4(shm_B[shm_stage][0][wIdX * wN + i * wSubN + thXinWarp * TN]);
        #pragma unroll 
        for (int32_t j = 0; j < TN; ++j) {
            reg_B[reg_stage][j] = shm_B[preShmStage][0][wIdX * wN + TN * thXinWarp + j];
        }
        
        //FLOAT4(reg_A[reg_stage][i * TM]) = FLOAT4(shm_A[shm_stage][0][wIdY * wM + i * wSubM + thYinWarp * TM]);
        #pragma unroll 
        for (int32_t j = 0; j < TM; ++j) {
            reg_A[reg_stage][j] = shm_A[preShmStage][0][wIdY * wM + TM * thYinWarp + j];
        }
        __syncthreads();

        #pragma unroll 
        for (int32_t dotIdx = 1;  dotIdx < BK + 1; ++dotIdx) {
            int32_t preRegStage = reg_stage;
            reg_stage = (reg_stage + 1) % 2;
            if (dotIdx < BK) {
                #pragma unroll 
                for (int32_t j = 0; j < TM; ++j) {
                    reg_A[reg_stage][j] = shm_A[preShmStage][dotIdx][wIdY * wM + TM * thYinWarp + j];
                }
                
                #pragma unroll 
                for (int32_t j = 0; j < TN; ++j) {
                    reg_B[reg_stage][j] = shm_B[preShmStage][dotIdx][wIdX * wN + TN * thXinWarp + j];
                }
            }

            #pragma unroll 
            for (int32_t i = 0; i < TM; ++i) {
                #pragma unroll 
                for (int32_t j = 0; j < TN; ++j) {
                    res[i][j] += reg_A[preRegStage][i] * reg_B[preRegStage][j];
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }

    // store to G_mem
    int32_t C_row; int32_t C_col;
    #pragma unroll 
    for (int32_t i = 0; i < TM; ++i) {
        C_row = start_row + wIdY * wM + thYinWarp * TM + i;
        #pragma unroll 
        for (int32_t j = 0; j < TN / 4; ++j) {
            C_col = start_col + wIdX *wN + thXinWarp * TN + 4 * j;
            FLOAT4(C[ELE_IDX(C_row, C_col, n)]) = FLOAT4(res[i][4 * j]);
        }
    }
        
    
}  