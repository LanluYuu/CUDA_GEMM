cmake_minimum_required(VERSION 3.10)

project(CUDA_GEMM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUTLASS_INCLUDE /root/haojiachen/code/Cuda_Learn/cutlass/include)
set(REF_FILE ref_${REF_TYPE}.cu)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include src ref ${CUTLASS_INCLUDE})
message(STATUS "cuda include dirs:${CUDA_INCLUDE_DIRS}, ${REF_FILE}")

if (${DATA_TYPE} STREQUAL "fp16")
    file(GLOB cudaFiles "src/fp16/${SOURCE_FILE}" "ref/${REF_FILE}" "include/*.cu")
elseif (${DATA_TYPE} STREQUAL "fp32")
    file(GLOB cudaFiles "src/fp32/${SOURCE_FILE}" "ref/${REF_FILE}" "include/*.cu")
else()
    message(FATAL_ERROR "The data type not right! Support fp16 and fp32")
endif()
add_library(cudaMatMul ${cudaFiles})
target_compile_features(cudaMatMul PUBLIC cxx_std_17)
target_compile_options(cudaMatMul PRIVATE -arch=compute_80 -code=sm_80)
target_link_libraries(cudaMatMul cuda)

add_executable(main main.cu)
if (${DATA_TYPE} STREQUAL "fp16")
    target_compile_definitions(main PRIVATE USE_FP16)
elseif (${DATA_TYPE} STREQUAL "fp32")
    target_compile_definitions(main PRIVATE USE_FP32)
endif()

if (${REF_TYPE} STREQUAL "cublas")
    target_compile_definitions(main PRIVATE K_VERSION=${K_VERSION} USE_CUBLAS)
    target_link_libraries(main cudaMatMul cublas)
elseif (${REF_TYPE} STREQUAL "cutlass")
    target_compile_definitions(main PRIVATE K_VERSION=${K_VERSION} USE_CUTLASS)
    target_link_libraries(main cudaMatMul)
else()
    message(FATAL_ERROR "The ref type not right! Support cublas and cutlass")
endif()

