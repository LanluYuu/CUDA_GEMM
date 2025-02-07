#define CUBLASAPI
#include <cublas_v2.h>
#include "d_helper.cu"

__host__ bool computeGoldenBlas(float* A, float* B, float* C, float* C_ref, int32_t m, int32_t k, int32_t n) {
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("\ncublas handle create fail!\n");
        return false;
    }
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    float alpha = 1.0f;
    float beta  = 0.0f;

    //warmUp 10 times
    for (int32_t i = 0; i < 10; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                n, m, k,
                &alpha,
                B, n,
                A, k, 
                &beta, 
                C, n);
    }

    const int32_t run_times = 100;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    #pragma unroll
    for (int32_t times = 0; times < run_times; ++times) {    

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    n, m, k,
                    &alpha,
                    B, n,
                    A, k, 
                    &beta, 
                    C, n);
    }
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float ms_sum = 0.0f;
    cudaEventElapsedTime(&ms_sum, start, end);
    float avg_ms = ms_sum / run_times;
    printf("\nCublas Execute time:%fms\n", avg_ms);
    double flopsPerMairixMul = 2.0 * k * m * n;
    printf("Cublas Throuphput:%lfTFLOPS\n", (flopsPerMairixMul * 1.0e-12f) / (avg_ms * 1.0e-3f));
    size_t size = m * n * sizeof(float);
    cudaMemcpy(C_ref, C, size, cudaMemcpyDeviceToHost);
    
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    
    cublasDestroy(handle);
    
    return true;
}