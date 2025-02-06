#include "gemm_v0.cu"
#include "gemm_v1.cu"
#include "gemm_v2.cu"
#include "gemm_v3.cu"
#include "gemm_v4.cu"
#include "gemm_v5.cu"
#include "gemm_v6.cu"
#if defined(USE_CUBLAS)
    #include "ref_cublas.cu"
#elif defined(USE_CUTLASS)
    #include "ref_cutlass.cu"
#endif
#include "helper.h"

//host code
void MatMul(int32_t m, int32_t k, int32_t n) { //Matirx A(m, k) * B(k, n)
    //init Matrix
    Matrix A(m, k);
    Matrix B(k, n);
    Matrix C(m, n);
    Matrix C_ref(m, n);
    printf("init host matirx done\n");
    //generate random value for A matrix
    GenRdVal4Mat(A);
    GenRdVal4Mat(B);

    // debug
    /*    printf("A:\n");
        for (int32_t i = 0; i < 8; ++i) {
            for (int32_t j = 0; j < 128; ++j) {
                printf("%f,", A.data[ELE_IDX(i, j, k)]);
            }
            printf("\n");
        }
        printf("\n");
        printf("B:\n");
        for (int32_t i = 0; i < 8; ++i) {
            for (int32_t j = 0; j < 128; ++j) {
                printf("%f,", B.data[ELE_IDX(i, j, n)]);
            }
            printf("\n");
        }
        printf("\n");   
    */
    cudaError_t err = cudaSetDevice(1);
    if (err != cudaSuccess) {
        std::cerr << "\nInit device failed!\n" << cudaGetErrorString(err) << std::endl;
    }
    //set device 
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "device name:" << prop.name << std::endl;

    //init Matrix on device
    float *A_d; float *B_d; float *C_d; float *C_d_ref;
    size_t A_size = m * k * sizeof(float);
    size_t B_size = k * n * sizeof(float);
    size_t C_size = m * n * sizeof(float);
    cudaMalloc(&A_d, A_size);
    cudaMalloc(&B_d, B_size);
    cudaMalloc(&C_d, C_size);
    cudaMalloc(&C_d_ref, C_size);
    //memcpy to device
    err = cudaMemcpy(A_d, A.data, A_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMemcpy(B_d, B.data, B_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
    }
    //compute golden using cublas or cutlass
    //std::cout << "main:::" << REF_TYPE << std::endl;E
#if defined(USE_CUBLAS)
    computeGoldenBlas(A_d, B_d, C_d_ref, C_ref.data, m, k, n);
#elif defined(USE_CUTLASS)
    computeGoldenCutlass(A_d, B_d, C_d_ref, C_ref.data, m, k, n);
#else 
    std::cerr << "which reference not defined!" << std::endl;
#endif
    //invoke kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // =====v0=====
#if K_VERSION == 0
    dim3 dimBlock(32, 32); //set threads per block
    dim3 dimGrid((m - 1) / dimBlock.x + 1, (n - 1) / dimBlock.y + 1);
#elif K_VERSION == 1
    // =====v1=====
    dim3 dimBlock(32, 32); //set threads per block
    dim3 dimGrid((m - 1) / dimBlock.x + 1, ((n - 1) / dimBlock.y + 1) / 8);
#elif K_VERSION == 2
    // =====v2.1=====
    dim3 dimBlock(64, 8); //set threads per block
    dim3 dimGrid((m - 1) / dimBlock.x + 1, ((n - 1) / dimBlock.y + 1) / 8);
#elif K_VERSION == 3
    // =====v3=====
    dim3 dimBlock(16, 16);
    dim3 dimGrid(((m - 1) / dimBlock.x + 1) / 4, ((n - 1) / dimBlock.y + 1) / 4);
    //dim3 dimGrid(1, 1);
#elif K_VERSION == 4
    // =====v4====
    dim3 dimBlock(16, 16);
    dim3 dimGrid(((m - 1) / dimBlock.x + 1) / 8, ((n - 1) / dimBlock.y + 1) / 8);
#elif K_VERSION == 5
    // =====v5=====
    dim3 dimBlock(128);
    dim3 dimGrid(32, 64);
#elif K_VERSION == 6
    // =====v6=====
    dim3 dimBlock(128);
    dim3 dimGrid(32, 64); // for gemm_v6
    //dim3 dimGrid(64, 128); // for gemm_v6_1
#endif
    //warm up for 10times
    for (int32_t i = 0; i < WARMUPT; ++i) {
#if K_VERSION == 0
        gemm_v0<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
#elif K_VERSION == 1
        gemm_v1<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
#elif K_VERSION == 2
        gemm_v2_1<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
#elif K_VERSION == 3
        gemm_v3_1<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
#elif K_VERSION == 4
        gemm_v4<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
#elif K_VERSION == 5
        gemm_v5<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
#elif K_VERSION == 6
        gemm_v6<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
#endif
        cudaDeviceSynchronize();
    }
    // begin record time
    cudaEventRecord(start, 0);
    #pragma unroll
    for (int32_t times = 0; times < RUNTIMES; ++times) {
#if K_VERSION == 0
        gemm_v0<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
#elif K_VERSION == 1
        gemm_v1<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
#elif K_VERSION == 2
        gemm_v2_1<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
#elif K_VERSION == 3
        gemm_v3_1<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
#elif K_VERSION == 4
        gemm_v4<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
#elif K_VERSION == 5
        gemm_v5<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
#elif K_VERSION == 6
        gemm_v6<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
#endif
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    float ms_sum = 0;
    cudaEventElapsedTime(&ms_sum, start, stop);
    float avg_ms = ms_sum / RUNTIMES;
    //memcpy to host
    err = cudaMemcpy(C.data, C_d, C_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
    }
    //compare
    //std::cout << "C data:" << C.data[0] << "," << C.data[1] << "," << C.data[2] << std::endl;
    //std::cout << "C ref data:" << C_ref.data[0] << "," << C_ref.data[1] << "," << C_ref.data[2] << std::endl;
    if (CompareMat(C, C_ref)) {
        printf("\nresult pass!\n");
    } else {
        printf("\ncompare fail!\n");
    }
    std::cout << "\nExecute time:" << avg_ms << "ms" << std::endl;
    double flopsPerMairixMul = 2.0 * k * m * n;
    double tflops = (flopsPerMairixMul * 1e-12) / (avg_ms * 1e-3);
    std::cout << "Throuphput:" << tflops << "TFLOPS\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    MatMul(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
}