#include "d_helper.cu"

__global__ void gemm_v0(float* A, float* B, float* C, int32_t m, int32_t k, int32_t n) {
    // 3.06806TFLOPS
    int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    float res = 0.0f;
    for (int32_t i = 0; i < k; ++i) {
        res += A[ELE_IDX(row, i, k)] * B[ELE_IDX(i, col, n)]; 
        //printf("row:%d, col:%d, A:%f, B:%f\n",row ,col ,d_GetMatrixElement(A, row, i), d_GetMatrixElement(B, i, col));
    }
    //d_SetMatrixElement(C, row, col, res);
    C[ELE_IDX(row, col, n)] = res;
    //printf("row:%d, col:%d, res:%f\n", row, col, res);
    //__syncthreads();
}