cmake_minimum_required(VERSION 3.10)
project(CudaArgsProfileRuntime)

# 查找CUDA
find_package(CUDA REQUIRED)

add_library(CudaArgsProfileRuntime SHARED profiler.cpp)

target_include_directories(CudaArgsProfileRuntime PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(CudaArgsProfileRuntime ${CUDA_LIBRARIES})
