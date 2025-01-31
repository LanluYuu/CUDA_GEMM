cmake_minimum_required(VERSION 3.10)

project(CUDA_GEMM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include src)
message(STATUS "cuda include dirs:${CUDA_INCLUDE_DIRS}")

file(GLOB cudaFiles "src/${SOURCE_FILE}" "src/ref_cublas.cu" "include/*.cu")
add_library(cudaMatMul ${cudaFiles})
target_compile_features(cudaMatMul PUBLIC cxx_std_11)

add_executable(main main.cu)
target_compile_definitions(main PRIVATE K_VERSION=${K_VERSION})
target_link_libraries(main cudaMatMul cublas)