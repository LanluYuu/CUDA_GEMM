#include <mma.h>
#include "d_helper.cu"

using namespace nvcuda;

// #define ELE_IDX(x, y, col) (x * col + y)

// inline __device__ void cp_async_global_to_shared(void *dst, const void *src, int size) {
//     asm volatile (
//         "cp.async.ca.shared.global [%0], [%1], %2;\n"
//         :: "l"(dst), "l"(src), "r"(size)
//     );
// };

// inline __device__ void cp_async_commit_group() {
//     asm volatile (
//         "cp.async.commit_group;\n"
//     );
// };

// inline __device__ void cp_async_wait_group(int N) {
//     asm volatile (
//         "cp.async.wait_group %0;\n" 
//         :: "r"(N)
//     );
// }

__global__ void gemm_v2(half *A, half *B, half *C, int32_t m, int32_t k, int32_t n) {
    // 54.1578TFLOPS
    constexpr int32_t warpSize = 32;
    constexpr int32_t fragSize = 16;
    constexpr int32_t wM = 64; // each warp calculate 64x64 
    constexpr int32_t wN = 64;
    constexpr int32_t fragRows = wN / fragSize;
    constexpr int32_t fragCols = wM / fragSize;
    // constexpr int32_t wK = 16;
    constexpr int32_t bM = 256;
    constexpr int32_t bN = 128;
    constexpr int32_t bK = 32;
    constexpr int32_t warpNumX = bN / wN;
    constexpr int32_t warpNumY = bM / wM;
    // constexpr int32_t rBNumPerThread = 16;
    // constexpr int32_t rANumPerThread = 32;
    // constexpr int32_t rNumPerThread = 8;
    constexpr int32_t w_kStride = 16;
    int32_t A_rd_times = bM * bK / (8 * blockDim.x);
    int32_t B_rd_times = bK * bN / (8 * blockDim.x);
    int32_t B_rd_rowsPerTime  = (8 * blockDim.x) / bN;
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
    // const int32_t r_A_rowStride = rNumPerThread * blockDim.x / bK;
    // const int32_t r_B_rowStride = rNumPerThread * blockDim.x / bN;
    const int32_t st_start_row = bky * bM + c_start_row;
    const int32_t st_start_col = bkx * bN + c_start_col;
    __shared__ half shm_A[bM][bK]; // 256x32
    __shared__ half shm_B[bK][bN]; // 32x128

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

        for (int32_t B_rd_time = 0; B_rd_time < B_rd_times; ++B_rd_time) {
            size_t shm_B_Dst   = __cvta_generic_to_shared(&shm_B[B_rd_rowsPerTime * B_rd_time + thx * 8 / bN][(thx * 8) % bN]);
            int32_t* glb_B_Src = (int32_t*)(&B[ELE_IDX((k_stride + B_rd_rowsPerTime * B_rd_time + thx * 8 / bN), (bkx * bN + (thx * 8) % bN), n)]);
            cp_async_global_to_shared(shm_B_Dst, glb_B_Src);
        }
        
        // half *shm_A_Dst = &shm_A[0][0];
        // half *glb_A_Src = &A[ELE_IDX((bky * bM), (k_stride), k)];
        // int32_t mv_A_size = bM * bK * sizeof(half);
        // cp_async_global_to_shared(shm_A_Dst, glb_A_Src, mv_A_size);
        // half *shm_B_Dst = &shm_B[0][0];
        // half *glb_B_Src = &B[ELE_IDX((k_stride), (bkx * bN), n)];
        // int32_t mv_B_size = bK * bN * sizeof(half);
        // cp_async_global_to_shared(shm_B_Dst, glb_B_Src, mv_B_size);

        cp_async_commit_group();
        cp_async_wait_group0();
        __syncthreads();
        #pragma unroll
        for (int32_t w_start_k = 0; w_start_k < bK; w_start_k += w_kStride) {
            #pragma unroll
            for (int32_t fragRow = 0; fragRow < fragRows; ++fragRow) {
                load_matrix_sync(frag_A[fragRow], &shm_A[c_start_row + fragRow * fragSize][w_start_k], bK);
            }
            #pragma unroll
            for (int32_t fragCol = 0; fragCol < fragCols; ++fragCol) {
                load_matrix_sync(frag_B[fragCol], &shm_B[w_start_k][c_start_col + fragCol * fragSize], bN);
            }
            #pragma unroll
            for (int32_t fragRow = 0; fragRow < fragRows; ++fragRow) {
                #pragma unroll
                for (int32_t fragCol = 0; fragCol < fragCols; ++fragCol) {
                    mma_sync(frag_acc[fragRow][fragCol], frag_A[fragRow], frag_B[fragCol], frag_acc[fragRow][fragCol]);
                }
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