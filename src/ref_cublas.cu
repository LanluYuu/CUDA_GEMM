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
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                n, m, k,
                &alpha,
                B, n,
                A, k, 
                &beta, 
                C, n);
    
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, end);
    printf("\nExecute time:%fms\n", milliseconds);
    double flopsPerMairixMul = 2.0 * k * m * n;
    printf("Throuphput:%lfTFLOPS\n", (flopsPerMairixMul * 1.0e-12f) / (milliseconds * 1.0e-3f));
    size_t size = m * n * sizeof(float);
    cudaMemcpy(C_ref, C, size, cudaMemcpyDeviceToHost);
    
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    
    cublasDestroy(handle);
    
    return true;
}