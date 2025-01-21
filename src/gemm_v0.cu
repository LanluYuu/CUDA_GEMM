#include "d_helper.cu"

__global__ void gemm_v0(float* A, float* B, float* C) {
    int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    float res = 0.0f;
    for (int32_t i = 0; i < BLOCK_SIZE; ++i) {
        res += A[ELE_IDX(row, i, BLOCK_SIZE)] * B[ELE_IDX(i, col, BLOCK_SIZE)]; 
        //printf("row:%d, col:%d, A:%f, B:%f\n",row ,col ,d_GetMatrixElement(A, row, i), d_GetMatrixElement(B, i, col));
    }
    //d_SetMatrixElement(C, row, col, res);
    C[ELE_IDX(row, col, BLOCK_SIZE)] = res;
    //printf("row:%d, col:%d, res:%f\n", row, col, res);
    //__syncthreads();
}