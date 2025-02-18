#if defined(USE_CUBLAS)
    #include "ref_cublas.cu"
#elif defined(USE_CUTLASS)
    #include "ref_cutlass.cu"
#endif
#include "helper.h"


__global__ void gemm_v6(float* A, float* B, float* C, int32_t m, int32_t k, int32_t n);
__global__ void gemm_v0(half *A, half *B, half *C, int32_t m, int32_t k, int32_t n);
__global__ void gemm_v1(half *A, half *B, half *C, int32_t m, int32_t k, int32_t n);
__global__ void gemm_v2(half *A, half *B, half *C, int32_t m, int32_t k, int32_t n);
__global__ void gemm_v3(half *A, half *B, half *C, int32_t m, int32_t k, int32_t n);
__global__ void gemm_v4(half *A, half *B, half *C, int32_t m, int32_t k, int32_t n);
__global__ void gemm_v3_1(half *A, half *B, half *C, int32_t m, int32_t k, int32_t n);
__global__ void gemm_v3_2(half *A, half *B, half *C, int32_t m, int32_t k, int32_t n);

//host code
template<typename T> // data type
void MatMul(int32_t m, int32_t k, int32_t n, T val) { //Matirx A(m, k) * B(k, n)
    //init Matrix
    Matrix<T> A(m, k);
    Matrix<T> B(k, n);
    Matrix<T> C(m, n);
    Matrix<T> C_ref(m, n);
    printf("init host matirx done\n");
    //generate random value for A matrix
    GenRdVal4Mat(A);
    GenRdVal4Mat(B);

    cudaError_t err = cudaSetDevice(1);
    if (err != cudaSuccess) {
        std::cerr << "\nInit device failed!\n" << cudaGetErrorString(err) << std::endl;
    }
    //set device 
    cudaSetDevice(1);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "device name:" << prop.name << std::endl;

    //init Matrix on device
    T *A_d; T *B_d; T *C_d; T *C_d_ref;
    size_t A_size = m * k * sizeof(T);
    size_t B_size = k * n * sizeof(T);
    size_t C_size = m * n * sizeof(T);
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
#if defined(USE_CUBLAS)
    #if defined(USE_FP16)
        computeGoldenBlasFp16(A_d, B_d, C_d_ref, C_ref.data, m, k, n);
    #elif defined(USE_FP32)
        computeGoldenBlasFp32(A_d, B_d, C_d_ref, C_ref.data, m, k, n);
    #endif
#elif defined(USE_CUTLASS)
    computeGoldenCutlass(A_d, B_d, C_d_ref, C_ref.data, m, k, n);
#else 
    std::cerr << "reference function not defined!" << std::endl;
#endif
    //invoke kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#if defined(USE_FP32)
        // =====v0=====
    #if K_VERSION == 0
        dim3 dimBlock(32, 32); //set threads per block
        dim3 dimGrid((m - 1) / dimBlock.x + 1, (n - 1) / dimBlock.y + 1);
    #elif K_VERSION == 1
        // =====v1=====
        dim3 dimBlock(32, 32);
        dim3 dimGrid((m - 1) / dimBlock.x + 1, (n - 1) / dimBlock.y + 1);
    #elif K_VERSION == 2
        // =====v2=====
        dim3 dimBlock(32, 32);
        dim3 dimGrid(((m - 1) / dimBlock.x + 1) / 2, ((n - 1) / dimBlock.y + 1) / 2);
        // =====v2.1=====
        // dim3 dimBlock(64, 8); 
        // dim3 dimGrid((m - 1) / dimBlock.x + 1, ((n - 1) / dimBlock.y + 1) / 8);
    #elif K_VERSION == 3
        // =====v3=====
        dim3 dimBlock(16, 16);
        dim3 dimGrid(((m - 1) / dimBlock.x + 1) / 2, ((n - 1) / dimBlock.y + 1) / 2);
        // =====v3_1=====
        // dim3 dimBlock(16, 16);
        // dim3 dimGrid(((m - 1) / dimBlock.x + 1) / 4, ((n - 1) / dimBlock.y + 1) / 4);
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
        dim3 dimGrid(32, 64); 
        // =====v6_1=====
        // dim3 dimBlock(128);
        // dim3 dimGrid(64, 128);
    #endif

    //warm up for 10times
    for (int32_t i = 0; i < WARMUPT; ++i) {
        #if K_VERSION == 0
                gemm_v0<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
        #elif K_VERSION == 1
                gemm_v1<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
        #elif K_VERSION == 2
                gemm_v2<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
        #elif K_VERSION == 3
                gemm_v3<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
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
            gemm_v2<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
        #elif K_VERSION == 3
            gemm_v3<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
        #elif K_VERSION == 4
            gemm_v4<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
        #elif K_VERSION == 5
            gemm_v5<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
        #elif K_VERSION == 6
            gemm_v6<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
        #endif
        }
#elif defined(USE_FP16)
    #if K_VERSION == 0
        dim3 dimBlock(128);
        dim3 dimGrid(n / 32, m / 32);
    #elif K_VERSION == 1
        dim3 dimBlock(256);
        dim3 dimGrid(n / 128, m / 256);
    #elif K_VERSION == 2
        dim3 dimBlock(256);
        dim3 dimGrid(n / 128, m / 256);
    #elif K_VERSION == 3
        dim3 dimBlock(256);
        dim3 dimGrid(n / 128, m / 256);
    #elif K_VERSION == 4
        dim3 dimBlock(256);
        dim3 dimGrid(n / 128, m / 256);
    #endif
    //warm up for 10times
    uint32_t dsmem_v3 = (256 * 40 + 32 * 136) * sizeof(half);
    uint32_t dsmem_v4 = 2 * (256 * 40 + 32 * 136) * sizeof(half);
        for (int32_t i = 0; i < WARMUPT; ++i) {
        #if K_VERSION == 0
            gemm_v0<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
        #elif K_VERSION == 1
            gemm_v1<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
        #elif K_VERSION == 2
            gemm_v2<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
        #elif K_VERSION == 3
            // cudaFuncSetAttribute(gemm_v3_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
            // gemm_v3_2<<<dimGrid, dimBlock, dsmem_v3>>> (A_d, B_d, C_d, m, k, n);
            gemm_v3_1<<<dimGrid, dimBlock, dsmem_v3>>> (A_d, B_d, C_d, m, k, n);
        #elif K_VERSION == 4
            cudaFuncSetAttribute(gemm_v4, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
            gemm_v4<<<dimGrid, dimBlock, dsmem_v4>>> (A_d, B_d, C_d, m, k, n);
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
            gemm_v2<<<dimGrid, dimBlock>>> (A_d, B_d, C_d, m, k, n);
        #elif K_VERSION == 3
            gemm_v3_1<<<dimGrid, dimBlock, dsmem_v3>>> (A_d, B_d, C_d, m, k, n);
            // gemm_v3_2<<<dimGrid, dimBlock, dsmem_v3>>> (A_d, B_d, C_d, m, k, n);
        #elif K_VERSION == 4
            gemm_v4<<<dimGrid, dimBlock, dsmem_v4>>> (A_d, B_d, C_d, m, k, n);
        #endif
        }
#endif
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
#if defined(USE_FP16)
    MatMul(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __float2half(1.0f));
#elif defined(USE_FP32)
    MatMul(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, 1.0f);
#endif
}