#include <cutlass/gemm/device/gemm.h>
#include "d_helper.cu"

__host__ bool computeGoldenCutlass(float* A, float* B, float* C, float* C_ref, int32_t m, int32_t k, int32_t n) {
    using layout = cutlass::layout::RowMajor;
    using Gemm = cutlass::gemm::device::Gemm<float, 
                                            layout,
                                            float,
                                            layout,
                                            float,
                                            layout>;
    Gemm gemm_op;
    float alpha = 1.0f;
    float beta = 0.0f;
    Gemm::Arguments args({m, n, k},
                                {A, m},
                                {B, k}, 
                                {C, m},
                                {C, m},
                                {alpha, beta});
    // warm up
    for (int32_t i = 0; i < 10; ++i) {
        cutlass::Status status = gemm_op(args);
        if (status != cutlass::Status::kSuccess) {
            printf("\ncutlass compute fail!\n");
            return false;
        }
    }

    //recode time
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    #pragma unroll
    for (int32_t times = 0; times < RUNTIMES; ++times) {    
        cutlass::Status status = gemm_op(args);
        if (status != cutlass::Status::kSuccess) {
            printf("\ncutlass compute fail!\n");
            return false;
        }
    }
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float ms_sum = 0.0f;
    cudaEventElapsedTime(&ms_sum, start, end);
    float avg_ms = ms_sum / RUNTIMES;
    printf("\nCutlass Execute time:%fms\n", avg_ms);
    double flopsPerMairixMul = 2.0 * k * m * n;
    printf("Cutlass Throuphput:%lfTFLOPS\n", (flopsPerMairixMul * 1.0e-12f) / (avg_ms * 1.0e-3f));
    size_t size = m * n * sizeof(float);
    cudaMemcpy(C_ref, C, size, cudaMemcpyDeviceToHost);
    
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    
    return true;
}