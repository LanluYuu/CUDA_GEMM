#include <mma.h>
#include "d_helper.cu"

using namespace nvcuda;

__global__ void gemm_v4(half *A, half *B, half *C, int32_t m, int32_t k, int32_t n) {
    // double buffer prefetching from G_mem to S_mem : 133.338TFLOPS
    constexpr int32_t warpSize = 32;
    constexpr int32_t fragSize = 16;
    constexpr int32_t wM = 64; // each warp calculate 64x64 
    constexpr int32_t wN = 64;
    constexpr int32_t fragRows = wN / fragSize; // 64/16=4
    constexpr int32_t fragCols = wM / fragSize; // 64/16=4
    constexpr int32_t bM = 256;
    constexpr int32_t bN = 128;
    constexpr int32_t bK = 32;
    constexpr int32_t Padding = 8; 
    // constexpr int32_t shmStride = 40; // 32 + 8
    constexpr const int32_t shm_A_idx = 0;
    constexpr const int32_t shm_A_idx_offset = bM * (bK + Padding);
    constexpr const int32_t shm_B_idx = shm_A_idx + 2 * shm_A_idx_offset;
    constexpr const int32_t shm_B_idx_offset = bK * (bN + Padding);
    constexpr int32_t warpNumX = bN / wN;
    // constexpr int32_t warpNumY = bM / wM;
    constexpr int32_t w_kStride = 16;
    const int32_t A_rd_times = bM * bK / (8 * blockDim.x); // 4
    const int32_t B_rd_times = bK * bN / (8 * blockDim.x); // 2
    const int32_t B_rd_rowsPerTime  = (8 * blockDim.x) / bN;
    const int32_t thx = threadIdx.x;
    const int32_t bkx = blockIdx.x;
    const int32_t bky = blockIdx.y;
    const int32_t warpId = thx / warpSize;
    const int32_t warpIdx = warpId % warpNumX;
    const int32_t warpIdy = warpId / warpNumX;
    const int32_t c_start_row = warpIdy * wM;
    const int32_t c_start_col = warpIdx * wN;
    const int32_t st_start_row = bky * bM + c_start_row;
    const int32_t st_start_col = bkx * bN + c_start_col;
    extern __shared__ half shm[]; // A:2x256x(32+8) + B:2x32x(128+8)
    int32_t LDGstage = 0;
    // int32_t LDSstage = 0;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[2][fragRows];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[2][fragCols];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_acc[fragRows][fragCols];
    #pragma unroll 
    for (int32_t row = 0; row < fragRows; ++row) {
        for (int32_t col = 0; col < fragCols; ++col) {
            wmma::fill_fragment(frag_acc[row][col], 0.0f);
        }
    }

    // async from G_mem at stage0
    #pragma unroll
    for (int32_t A_rd_time = 0; A_rd_time < A_rd_times; ++A_rd_time) {
        size_t shm_A_Dst   = __cvta_generic_to_shared(&shm[shm_A_idx + ELE_IDX(thx, (A_rd_time * 8), (bK + Padding))]);
        int32_t *glb_A_Src = (int32_t*)(&A[ELE_IDX((bky * bM + thx), (0 + A_rd_time * 8), k)]);
        cp_async_global_to_shared(shm_A_Dst, glb_A_Src); 
    }
    #pragma unroll 
    for (int32_t B_rd_time = 0; B_rd_time < B_rd_times; ++B_rd_time) {
        // int32_t glb_col    = (thx % 8) * 16 + B_rd_time * 8;
        size_t shm_B_Dst   = __cvta_generic_to_shared(&shm[shm_B_idx + ELE_IDX((B_rd_rowsPerTime * B_rd_time + thx * 8 / bN), ((thx * 8) % bN), (bN + Padding))]);
        int32_t *glb_B_Src = (int32_t*)(&B[ELE_IDX((0 + B_rd_rowsPerTime * B_rd_time + thx * 8 / bN), (bkx * bN + (thx * 8) % bN), n)]);
        cp_async_global_to_shared(shm_B_Dst, glb_B_Src);
    }
    cp_async_commit_group();
    cp_async_wait_group0();
    __syncthreads();
        //             if (bkx == 10 && bky == 0 && thx == 64)    
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
    int32_t next_LDGstage = LDGstage;
    for (int32_t k_stride = bK; k_stride <= k; k_stride += bK) {
        if (k_stride < k) {
            // read next area to S_mem
            next_LDGstage = (LDGstage + 1) % 2;
            #pragma unroll
            for (int32_t A_rd_time = 0; A_rd_time < A_rd_times; ++A_rd_time) {
                size_t shm_A_Dst   = __cvta_generic_to_shared(&shm[shm_A_idx + next_LDGstage * shm_A_idx_offset + ELE_IDX(thx, (A_rd_time * 8), (bK + Padding))]);
                int32_t *glb_A_Src = (int32_t*)(&A[ELE_IDX((bky * bM + thx), (k_stride + A_rd_time * 8), k)]);
                cp_async_global_to_shared(shm_A_Dst, glb_A_Src); 
            }
            #pragma unroll 
            for (int32_t B_rd_time = 0; B_rd_time < B_rd_times; ++B_rd_time) {
                // int32_t glb_col    = (thx % 8) * 16 + B_rd_time * 8;
                size_t shm_B_Dst   = __cvta_generic_to_shared(&shm[shm_B_idx + next_LDGstage * shm_B_idx_offset + ELE_IDX((B_rd_rowsPerTime * B_rd_time + thx * 8 / bN), ((thx * 8) % bN), (bN + Padding))]);
                int32_t *glb_B_Src = (int32_t*)(&B[ELE_IDX((k_stride + B_rd_rowsPerTime * B_rd_time + thx * 8 / bN), (bkx * bN + (thx * 8) % bN), n)]);
                cp_async_global_to_shared(shm_B_Dst, glb_B_Src);
            }
        }

        // calculate cur area
        #pragma unroll
        for (int32_t fragRow = 0; fragRow < fragRows; ++fragRow) {
            load_matrix_sync(frag_A[0][fragRow], &shm[shm_A_idx + LDGstage * shm_A_idx_offset + ELE_IDX((c_start_row + fragRow * fragSize), 0, (bK + Padding))], bK + Padding);
            load_matrix_sync(frag_A[1][fragRow], &shm[shm_A_idx + LDGstage * shm_A_idx_offset + ELE_IDX((c_start_row + fragRow * fragSize), w_kStride, (bK + Padding))], bK + Padding);
        }
        #pragma unroll
        for (int32_t fragCol = 0; fragCol < fragCols; ++fragCol) {
            load_matrix_sync(frag_B[0][fragCol], &shm[shm_B_idx + LDGstage * shm_B_idx_offset + ELE_IDX(0, (c_start_col + fragCol * fragSize), (bN + Padding))], bN + Padding);
            load_matrix_sync(frag_B[1][fragCol], &shm[shm_B_idx + LDGstage * shm_B_idx_offset + ELE_IDX(w_kStride, (c_start_col + fragCol * fragSize), (bN + Padding))], bN + Padding);
        }
        // printf("xxx");
        // read second frag
        // int32_t next_LDSstage = (LDSstage + 1) % 2;
        // #pragma unroll
        // for (int32_t fragRow = 0; fragRow < fragRows; ++fragRow) {
        //     load_matrix_sync(frag_A[1][fragRow], &shm[shm_A_idx + LDGstage * shm_A_idx_offset + c_start_row + fragRow * fragSize][fragSize], shmStride);
        // }
        // #pragma unroll
        // for (int32_t fragCol = 0; fragCol < fragCols; ++fragCol) {
        //     load_matrix_sync(frag_B[1][fragCol], &shm[shm_B_idx + LDGstage * shm_B_idx_offset + c_start_col + fragSize + (fragCol / 2) * 32][fragCol % 2 * fragSize], shmStride);
        // }
        // calculate cur frag
        #pragma unroll
        for (int32_t row = 0; row < fragRows; ++row) {
            #pragma unroll
            for (int32_t col = 0; col < fragCols; ++col) {
                mma_sync(frag_acc[row][col], frag_A[0][row], frag_B[0][col], frag_acc[row][col]);
                mma_sync(frag_acc[row][col], frag_A[1][row], frag_B[1][col], frag_acc[row][col]);
            }
        }
        // LDSstage = next_LDSstage;
        // calculate next frag
        // #pragma unroll
        // for (int32_t row = 0; row < fragRows; ++row) {
        //     #pragma unroll
        //     for (int32_t col = 0; col < fragCols; ++col) {
        //         mma_sync(frag_acc[row][col], frag_A[1][row], frag_B[1][col], frag_acc[row][col]);
        //     }
        // }
        cp_async_commit_group();
        cp_async_wait_group0();
        LDGstage = next_LDGstage;
        __syncthreads();
    }

    // store to G_mem
    #pragma unroll
    for (int32_t row = 0; row < fragRows; ++row) {
        #pragma unroll
        for (int32_t col = 0; col < fragCols; ++col) {
            store_matrix_sync(&C[ELE_IDX((st_start_row + row * fragSize), (st_start_col + col * fragSize), n)], frag_acc[row][col], n, wmma::mem_row_major);
        }
    }
}