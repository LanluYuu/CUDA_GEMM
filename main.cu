#include "gemm_v0.cu"
#include "gemm_v1.cu"
#include "helper.h"

//host code
void MatMul(int32_t x, int32_t y, int32_t z) { //Matirx A(x, z) * B(z, y)
    //init Matrix
    Matrix A(x, z);
    Matrix B(z, y);
    Matrix C(x, y);
    Matrix C_ref(x, y);
    //generate random value for A matrix
    GenRdVal4Mat(A);
    GenRdVal4Mat(B);
    //compute golden by CPU
    ComputeGolden(A, B, C_ref);
    //init Matrix on device
    d_Matrix A_d(x, z);
    d_Matrix B_d(z, y);
    d_Matrix C_d(x, y);
    //memcpy to device
    size_t A_size = A.height * A.width * sizeof(float);
    size_t B_size = B.height * B.width * sizeof(float);
    size_t C_size = C.height * C.width * sizeof(float);
    cudaMemset(C_d.data, 1, C_size);
    cudaMemcpy(A_d.data, A.data, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d.data, B.data, B_size, cudaMemcpyHostToDevice);
    //invoke kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // =====v0=====
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((x - 1) / BLOCK_SIZE + 1, (y - 1) / BLOCK_SIZE + 1);
    // gemm_v0<<<dimGrid, dimBlock>>> (A_d, B_d, C_d);
    // =====v0===== 
    // =====v1=====
    //dim3 dimBlock(BLOCK_SIZE / 2, BLOCK_SIZE / 2);
    //dim3 dimGrid((x - 1) / BLOCK_SIZE + 1, (y - 1) / BLOCK_SIZE + 1);
    // =====v1=====
    cudaEventRecord(start, 0);
    gemm_v0<<<dimGrid, dimBlock>>> (A_d, B_d, C_d);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    //memcpy to host
    cudaError_t err = cudaMemcpy(A_d.data, C.data, C_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
    }
    printf("C[0][0]:%f", C.data[0]);
    //compare
    if (CompareMat(C, C_ref)) {
        printf("\nresult pass!\n");
    } else {
        printf("\ncompare fail!\n");
    }
    std::cout << "\nexecute time:" << milliseconds << "ms" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    MatMul(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
}