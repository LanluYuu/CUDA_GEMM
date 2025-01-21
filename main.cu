#include "gemm_v0.cu"
#include "gemm_v1.cu"
#include "ref_cublas.cu"
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
    printf("A gen random done\n");
    GenRdVal4Mat(B);
    printf("B gen random done\n");
    //compute golden using cublas
    //ComputeGolden(A, B, C_ref);
    printf("compute golden done\n");
    cudaError_t err = cudaSetDevice(1);
    if (err != cudaSuccess) {
        std::cerr << "\nInit device failed!\n" << cudaGetErrorString(err) << std::endl;
    }
    //init Matrix on device
    /*d_Matrix A_d(x, z);
    d_Matrix B_d(z, y);
    d_Matrix C_d(x, y);
    printf("C_d matrix data addr:0x%lX", &C_d.data);*/
    float *A_d; float *B_d; float *C_d; float *C_d_ref;
    size_t size = BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
    cudaMalloc(&A_d, size);
    cudaMalloc(&B_d, size);
    cudaMalloc(&C_d, size);
    cudaMalloc(&C_d_ref, size);
    //memcpy to device
    /*size_t A_size = A.height * A.width * sizeof(float);
    size_t B_size = B.height * B.width * sizeof(float);
    size_t C_size = C * 32 * sizeof(float);*/
    //cudaMemset(C_d.data, 1, C_size);
    //err = cudaMemcpy(A_d.data, A.data, A_size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(A_d, A.data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMemcpy(B_d, B.data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
    }
    computeGoldenBlas(A_d, B_d, C_d_ref, C_ref.data, m, k, n);
    //invoke kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // =====v0=====
    dim3 dimBlock(32, 32); //set threads per block
    dim3 dimGrid((m - 1) / dimBlock.x + 1, (n - 1) / dimBlock.y + 1);
    // gemm_v0<<<dimGrid, dimBlock>>> (A_d, B_d, C_d);
    // =====v0===== 
    // =====v1=====
    //dim3 dimBlock(BLOCK_SIZE / 2, BLOCK_SIZE / 2);
    //dim3 dimGrid((x - 1) / BLOCK_SIZE + 1, (y - 1) / BLOCK_SIZE + 1);
    printf("before warm up\n");
    //warm up for 10times
    for (int32_t i = 0; i < WARMUPT; ++i) {
        gemm_v0<<<dimGrid, dimBlock>>> (A_d, B_d, C_d);
        cudaDeviceSynchronize();
        printf("done");
    }
    // =====v1=====
    cudaEventRecord(start, 0);
    gemm_v0<<<dimGrid, dimBlock>>> (A_d, B_d, C_d);
    printf(" compute gemm done\n");
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    //memcpy to host
    err = cudaMemcpy(C.data, C_d, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
    }
    //compare
    if (CompareMat(C, C_ref)) {
        printf("\nresult pass!\n");
    } else {
        printf("\ncompare fail!\n");
    }
    std::cout << "\nExecute time:" << milliseconds << "ms" << std::endl;
    double flopsPerMairixMul = 2.0 * k * m * n;
    double tflops = (flopsPerMairixMul * 1e-12) / (milliseconds * 1e-3);
    std::cout << "\nThrouphput:" << tflops << "TFLOPS\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    MatMul(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
}