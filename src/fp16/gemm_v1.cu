#include <mma.h>
#include "d_helper.cu"

using namespace nvcuda;

__global__ void gemm_v1(half *A, half *B, half *C, int32_t m, int32_t k, int32_t n) {
    // 27.6488TFLOPS
    constexpr int32_t warpSize = 32;
    constexpr int32_t fragSize = 16;
    constexpr int32_t wM = 64; // each warp calculate 64x64 
    constexpr int32_t wN = 64;
    constexpr int32_t fragRows = wN / fragSize;
    constexpr int32_t fragCols = wM / fragSize;
    // constexpr int32_t wK = 16;
    constexpr int32_t bM = 256;
    constexpr int32_t bN = 128;
    constexpr int32_t bK = 16;
    constexpr int32_t warpNumX = bN / wN;
    constexpr int32_t warpNumY = bM / wM;
    constexpr int32_t rNumPerThread = 8;

    const int32_t thx = threadIdx.x;
    const int32_t bkx = blockIdx.x;
    const int32_t bky = blockIdx.y;
    const int32_t warpId = thx / warpSize;
    const int32_t warpIdx = warpId % warpNumX;
    const int32_t warpIdy = warpId / warpNumX;

    const int32_t c_start_row = warpIdy * wM;
    const int32_t c_start_col = warpIdx * wN;
    // const int32_t r_A_times = bM * bK / (rNumPerThread * blockDim.x);
    // const int32_t r_B_times = bK * bN / (rNumPerThread * blockDim.x);
    const int32_t st_start_row = bky * bM + c_start_row;
    const int32_t st_start_col = bkx * bN + c_start_col;
    __shared__ half shm_A[bM][bK]; // 256x16
    __shared__ half shm_B[bK][bN]; // 16x128

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
        // read A to S_mem
        // #pragma unroll
        // for (int32_t i = 0; i < rANumPerThread; ++i) {
        //     shm_A[rANumPerThread * thx / bK][rANumPerThread * thx % bK + i] = 
        //         A[ELE_IDX((bM * bky + rANumPerThread * thx / bK), (k_stride + rANumPerThread * thx % bK + i), k)];
        // }
        for (int32_t ArowOffset = 0; ArowOffset < bM; ArowOffset += rNumPerThread * blockDim.x / bK) {
            HALF8(shm_A[rNumPerThread * thx / bK + ArowOffset][rNumPerThread * thx % bK]) = HALF8(A[ELE_IDX((bM * bky + rNumPerThread * thx / bK + ArowOffset), (k_stride + rNumPerThread * thx % bK), k)]);
        }
        // read B to S_mem
        // #pragma unroll
        // for (int32_t i = 0; i < rBNumPerThread; ++i) {
        //     shm_B[rBNumPerThread * thx / bN][rBNumPerThread * thx % bN + i] = 
        //         B[ELE_IDX((k_stride + rBNumPerThread * thx / bN), (bN * bkx + rBNumPerThread * thx % bN + i), n)];
        // }
        HALF8(shm_B[rNumPerThread * thx / bN][rNumPerThread * thx % bN]) = HALF8(B[ELE_IDX((k_stride + rNumPerThread * thx / bN), (bN * bkx + rNumPerThread * thx % bN), n)]);
        __syncthreads();
        #pragma unroll
        for (int32_t fragRow = 0; fragRow < fragRows; ++fragRow) {
            load_matrix_sync(frag_A[fragRow], &shm_A[c_start_row + fragRow * fragSize][0], bK);
        }
        #pragma unroll
        for (int32_t fragCol = 0; fragCol < fragCols; ++fragCol) {
            load_matrix_sync(frag_B[fragCol], &shm_B[0][c_start_col + fragCol * fragSize], bN);
        }
        #pragma unroll
        for (int32_t fragRow = 0; fragRow < fragRows; ++fragRow) {
            #pragma unroll
            for (int32_t fragCol = 0; fragCol < fragCols; ++fragCol) {
                mma_sync(frag_acc[fragRow][fragCol], frag_A[fragRow], frag_B[fragCol], frag_acc[fragRow][fragCol]);
            }
        }
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