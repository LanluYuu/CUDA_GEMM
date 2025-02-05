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

file(GLOB cudaFiles "src/${SOURCE_FILE}" "ref/${REF_FILE}" "include/*.cu")
add_library(cudaMatMul ${cudaFiles})
target_compile_features(cudaMatMul PUBLIC cxx_std_17)

add_executable(main main.cu)
if (${REF_TYPE} STREQUAL "cublas")
    target_compile_definitions(main PRIVATE K_VERSION=${K_VERSION} USE_CUBLAS)
    target_link_libraries(main cudaMatMul cublas)
elseif (${REF_TYPE} STREQUAL "cutlass")
    target_compile_definitions(main PRIVATE K_VERSION=${K_VERSION} USE_CUTLASS)
endif()
message(STATUS "REF_TYPE:${REF_TYPE}")
