#include "d_helper.cu"

__global__ void gemm_v0(d_Matrix A, d_Matrix B, d_Matrix C) {
    int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float res = 0.0f;
    for (int32_t i = 0; i < A.width; ++i) {
        res += d_GetMatrixElement(A, row, i) * d_GetMatrixElement(B, i, col); 
        printf("A:%f, B:%f\n", d_GetMatrixElement(A, row, i), d_GetMatrixElement(B, i, col));
    }

    d_SetMatrixElement(C, row, col, res);
}