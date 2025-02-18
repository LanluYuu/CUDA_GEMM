#include <mma.h>
#include "d_helper.cu"

using namespace nvcuda;

__global__ void gemm_v3(half *A, half *B, half *C, int32_t m, int32_t k, int32_t n) {
    // async + padding : 113.675TFLOPS
    constexpr int32_t warpSize = 32;
    constexpr int32_t fragSize = 16;
    constexpr int32_t wM = 64; // each warp calculate 64x64 
    constexpr int32_t wN = 64;
    constexpr int32_t fragRows = wN / fragSize;
    constexpr int32_t fragCols = wM / fragSize;
    constexpr int32_t bM = 256;
    constexpr int32_t bN = 128;
    constexpr int32_t bK = 32;
    constexpr int32_t Padding = 8; 
    constexpr int32_t warpNumX = bN / wN;
    constexpr int32_t warpNumY = bM / wM;
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
    __shared__ half shm_A[bM][bK + Padding]; // 32x(256+16)
    __shared__ half shm_B[bK][bN + Padding]; // 32x(128+16)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[fragRows];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[fragCols];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_acc[fragRows][fragCols];

    #pragma unroll
    for (int32_t fragRow = 0; fragRow < fragRows; ++fragRow) {
        #pragma unroll
        for (int32_t fragCol = 0; fragCol < fragCols; ++fragCol) {
            wmma::fill_fragment(frag_acc[fragRow][fragCol], 0.0f);
        }
    }

    #pragma unroll
    for (int32_t k_stride = 0; k_stride < k; k_stride += bK) {
        // 16B(8half) move per async
        #pragma unroll
        for (int32_t A_rd_time = 0; A_rd_time < A_rd_times; ++A_rd_time) {
            size_t shm_A_Dst   = __cvta_generic_to_shared(&shm_A[thx][A_rd_time * 8]);
            int32_t* glb_A_Src = (int32_t*)(&A[ELE_IDX((bky * bM + thx), (k_stride + A_rd_time * 8), k)]);
            cp_async_global_to_shared(shm_A_Dst, glb_A_Src);
        }
        #pragma unroll
        for (int32_t B_rd_time = 0; B_rd_time < B_rd_times; ++B_rd_time) {
            size_t shm_B_Dst   = __cvta_generic_to_shared(&shm_B[B_rd_rowsPerTime * B_rd_time + thx * 8 / bN][(thx * 8) % bN]);
            int32_t* glb_B_Src = (int32_t*)(&B[ELE_IDX((k_stride + B_rd_rowsPerTime * B_rd_time + thx * 8 / bN), (bkx * bN + (thx * 8) % bN), n)]);
            cp_async_global_to_shared(shm_B_Dst, glb_B_Src);
        }
        cp_async_commit_group();
        cp_async_wait_group0();
        __syncthreads();
        #pragma unroll
        for (int32_t w_start_k = 0; w_start_k < bK; w_start_k += w_kStride) {
            #pragma unroll
            for (int32_t fragRow = 0; fragRow < fragRows; ++fragRow) {
                load_matrix_sync(frag_A[fragRow], &shm_A[c_start_row + fragRow * fragSize][w_start_k], bK + Padding);
            }
            #pragma unroll
            for (int32_t fragCol = 0; fragCol < fragCols; ++fragCol) {
                load_matrix_sync(frag_B[fragCol], &shm_B[w_start_k][c_start_col + fragCol * fragSize], bN + Padding);
            }
            #pragma unroll
            for (int32_t fragRow = 0; fragRow < fragRows; ++fragRow) {
                #pragma unroll
                for (int32_t fragCol = 0; fragCol < fragCols; ++fragCol) {
                    mma_sync(frag_acc[fragRow][fragCol], frag_A[fragRow], frag_B[fragCol], frag_acc[fragRow][fragCol]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int32_t fragRow = 0; fragRow < fragRows; ++fragRow) {
        #pragma unroll
        for (int32_t fragCol = 0; fragCol < fragCols; ++fragCol) {
            store_matrix_sync(&C[ELE_IDX((st_start_row + fragRow * fragSize), (st_start_col + fragCol * fragSize), n)], frag_acc[fragRow][fragCol],
                n, wmma::mem_row_major);
        }
    }
}

__global__ void gemm_v3_1(half *A, half *B, half *C, int32_t m, int32_t k, int32_t n) {
    // async + padding : 97.7679TFLOPS
    constexpr int32_t warpSize = 32;
    constexpr int32_t fragSize = 16;
    constexpr int32_t wM = 64; // each warp calculate 64x64 
    constexpr int32_t wN = 64;
    constexpr int32_t fragRows = wN / fragSize;
    constexpr int32_t fragCols = wM / fragSize;
    constexpr int32_t bM = 256;
    constexpr int32_t bN = 128;
    constexpr int32_t bK = 32;
    constexpr int32_t Padding = 8; 
    constexpr int32_t shmStride = bK + Padding;
    constexpr int32_t warpNumX = bN / wN;
    constexpr int32_t warpNumY = bM / wM;
    constexpr int32_t w_kStride = 16;
    constexpr const int32_t shm_A_idx = 0;
    constexpr const int32_t shm_A_idx_offset = bM * (bK + Padding);
    constexpr const int32_t shm_B_idx = shm_A_idx + shm_A_idx_offset;
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
    extern __shared__ half shm[];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_A[fragRows];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_B[fragCols];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_acc[fragRows][fragCols];

    #pragma unroll
    for (int32_t fragRow = 0; fragRow < fragRows; ++fragRow) {
        #pragma unroll
        for (int32_t fragCol = 0; fragCol < fragCols; ++fragCol) {
            wmma::fill_fragment(frag_acc[fragRow][fragCol], 0.0f);
        }
    }

    #pragma unroll
    for (int32_t k_stride = 0; k_stride < k; k_stride += bK) {
        // 16B(8half) move per async
        #pragma unroll
        for (int32_t A_rd_time = 0; A_rd_time < A_rd_times; ++A_rd_time) {
            size_t shm_A_Dst   = __cvta_generic_to_shared(&shm[shm_A_idx + ELE_IDX(thx, A_rd_time * 8, (bK + Padding))]);
            int32_t* glb_A_Src = (int32_t*)(&A[ELE_IDX((bky * bM + thx), (k_stride + A_rd_time * 8), k)]);
            cp_async_global_to_shared(shm_A_Dst, glb_A_Src);
        }
        #pragma unroll
        for (int32_t B_rd_time = 0; B_rd_time < B_rd_times; ++B_rd_time) {
            // int32_t glb_col    = (thx % 8) * 16 + B_rd_time * 8;
            size_t shm_B_Dst   = __cvta_generic_to_shared(&shm[shm_B_idx + ELE_IDX((B_rd_rowsPerTime * B_rd_time + thx * 8 / bN), ((thx * 8) % bN), (bN + Padding))]);
            int32_t* glb_B_Src = (int32_t*)(&B[ELE_IDX((k_stride + B_rd_rowsPerTime * B_rd_time + thx * 8 / bN), (bkx * bN + (thx * 8) % bN), n)]);
            cp_async_global_to_shared(shm_B_Dst, glb_B_Src);
        }
        cp_async_commit_group();
        cp_async_wait_group0();
        __syncthreads();
        #pragma unroll
        for (int32_t w_start_k = 0; w_start_k < bK; w_start_k += w_kStride) {
            #pragma unroll
            for (int32_t fragRow = 0; fragRow < fragRows; ++fragRow) {
                load_matrix_sync(frag_A[fragRow], &shm[shm_A_idx + ELE_IDX((c_start_row + fragRow * fragSize), w_start_k, (bK + Padding))], bK + Padding);
            }
            #pragma unroll
            for (int32_t fragCol = 0; fragCol < fragCols; ++fragCol) {
                load_matrix_sync(frag_B[fragCol], &shm[shm_B_idx + ELE_IDX(w_start_k, (c_start_col + fragCol * fragSize), (bN + Padding))], bN + Padding);
            }
            #pragma unroll
            for (int32_t fragRow = 0; fragRow < fragRows; ++fragRow) {
                #pragma unroll
                for (int32_t fragCol = 0; fragCol < fragCols; ++fragCol) {
                    mma_sync(frag_acc[fragRow][fragCol], frag_A[fragRow], frag_B[fragCol], frag_acc[fragRow][fragCol]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int32_t fragRow = 0; fragRow < fragRows; ++fragRow) {
        #pragma unroll
        for (int32_t fragCol = 0; fragCol < fragCols; ++fragCol) {
            store_matrix_sync(&C[ELE_IDX((st_start_row + fragRow * fragSize), (st_start_col + fragCol * fragSize), n)], frag_acc[fragRow][fragCol],
                n, wmma::mem_row_major);
        }
    }
}
