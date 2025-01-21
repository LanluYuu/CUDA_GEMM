#include <cuda_runtime.h>
#include <iostream>
__global__ void gemm_v0(float* A, float* B) {
    int32_t idx = threadIdx.x;
    
    B[idx] = A[idx];
}

void MatMul() { //Matirx A(x, z) * B(z, y)
    float A[] = {1, 1, 1, 1};
    float B[] = {2, 2, 2, 2};
    cudaSetDevice(0);
    float* A_d; float* B_d;
    cudaMalloc(&A_d, 4 * sizeof(float));
    cudaMalloc(&B_d, 4 * sizeof(float));
    cudaError_t err = cudaMemcpy(A_d, A, 4 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "\nInit device failed!\n" << cudaGetErrorString(err) << std::endl;
    }
    dim3 dimGrid(1);
    dim3 dimBlock(4);
    gemm_v0<<<dimGrid, dimBlock>>> (A_d, B_d);
    err = cudaMemcpy(B, B_d, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "\nInit device failed!\n" << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();
    cudaFree(A_d);
    cudaFree(B_d);

    for (int32_t i = 0; i < 4; ++i) {
        printf("%f,", B[i]);
    }
}

int main() {
    MatMul();
}