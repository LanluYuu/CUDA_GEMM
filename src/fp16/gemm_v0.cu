#include <mma.h>
#include "d_helper.cu"

using namespace nvcuda;

__global__ void gemm_v0(half *A, half *B, half *C, int32_t m, int32_t k, int32_t n) {
    constexpr int32_t warpNumX = 2;
    constexpr int32_t warpNumY = 2;
    constexpr int32_t wM = 16;
    constexpr int32_t wN = 16;
    constexpr int32_t wK = 16;
    constexpr int32_t bM = wM * warpNumY;
    constexpr int32_t bN = wN * warpNumX;
    const int32_t bkx = blockIdx.x;
    const int32_t bky = blockIdx.y;
    const int32_t thx = threadIdx.x;
    const int32_t warpId = thx / 32; 
    const int32_t warpIdx = warpId % warpNumX;
    const int32_t warpIdy = warpId / warpNumX;
    
    wmma::fragment<wmma::matrix_a, wM, wN, wK, half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, wM, wN, wK, half, wmma::row_major> frag_B;
    wmma::fragment<wmma::accumulator, wM, wN, wK, half> frag_acc;

    wmma::fill_fragment(frag_acc, 0.0f);
    int32_t start_row = bky * bM + warpIdy * wM;
    int32_t start_col = bkx * bN + warpIdx * wN;
    for (int32_t k_stride = 0; k_stride < k; k_stride += wK) {
        wmma::load_matrix_sync(frag_A, &A[ELE_IDX(start_row, k_stride, k)], k);
        wmma::load_matrix_sync(frag_B, &B[ELE_IDX(k_stride, start_col, n)], n);

        wmma::mma_sync(frag_acc, frag_A, frag_B, frag_acc);
    }
    // for (int32_t A_col_stride = 0; A_col_stride < k; A_col_stride += wK) {
    //     for (int32_t B_row_stride = 0; B_row_stride < k; B_row_stride += wK) {
    //         wmma::load_matrix_sync(frag_A, &A[ELE_IDX(start_row, A_col_stride, k)], k);
    //         wmma::load_matrix_sync(frag_B, &B[ELE_IDX(B_row_stride, start_col, n)], n);

    //         wmma::mma_sync(frag_acc, frag_A, frag_B, frag_acc);
    //     }
    // }

    wmma::store_matrix_sync(&C[ELE_IDX(start_row, start_col, n)], frag_acc, n, wmma::mem_row_major);
}

// bool genRand(half *A, int32_t m, int32_t n) {
//     if (!A) {
//         return false;
//     }

//     std::random_device rd; //random seed
//     std::mt19937 gen(rd()); //Mersenne Twister random generator
//     std::uniform_real_distribution<float> dis(0, 1);

//     for (int32_t idx = 0; idx < m * n; ++idx) {
//         float tmp = dis(gen);
//         A[idx] = __float2half(tmp);
//     }

//     return true;
// }

// int main() {
//     int32_t blockSize = 32;
//     half *A_h = new half[blockSize * blockSize];
//     half *B_h = new half[blockSize * blockSize];
//     half *C_h = new half[blockSize * blockSize];
//     float *C_ref = new float[blockSize * blockSize];

//     if (genRand(A_h, blockSize, blockSize) && genRand(B_h, blockSize, blockSize)) {
//         std::cout << "gen random done\n";
//     } else {
//         return -1;
//     }

//     for (int32_t row = 0; row < blockSize; ++row) {
//         for (int32_t col = 0; col < blockSize; ++col) {
//             for (int32_t k = 0; k < blockSize; ++k) {
//                 C_ref[row * blockSize + col] += __half2float(A_h[row * blockSize + k]) * __half2float(B_h[k * blockSize + col]);
//             }
//         }
//     }

//     half *A_d; half *B_d; half *C_d;
//     size_t size = blockSize * blockSize * sizeof(half);
//     cudaMalloc(&A_d, size);
//     cudaMalloc(&B_d, size);
//     cudaMalloc(&C_d, size);
//     cudaError_t err = cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
//     if (err != cudaSuccess) {
//         std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
//     }
//     err = cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
//     if (err != cudaSuccess) {
//         std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
//     }

//     dim3 dimBlock(128);
//     dim3 dimGrid(1);
//     gemm_v0<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, blockSize, blockSize, blockSize);
//     cudaDeviceSynchronize();

//     err = cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
//     if (err != cudaSuccess) {
//         std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
//     }

//     for (int32_t idx = 0; idx < blockSize * blockSize; ++idx) {
//         if (abs(__half2float(C_h[idx]) - C_ref[idx]) > 1e-2) {
//             std::cout << "idx:" << idx << ", C_d:" << __half2float(C_h[idx]) << ", C_ref:" << C_ref[idx] << std::endl;
//         }
//     }
// }