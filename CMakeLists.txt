project(gpu_least_squares_optimizer LANGUAGES CUDA CXX)
cmake_minimum_required(VERSION 3.21)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

# Libraries
add_library(glso INTERFACE)
target_include_directories(glso INTERFACE include)

# Examples
add_executable(hello examples/hello.cu)
target_link_libraries(hello glso)
# set_property(TARGET hello PROPERTY CUDA_ARCHITECTURES native)

add_executable(optimize examples/optimize.cu)
target_link_libraries(optimize glso)