project(gpu_least_squares_optimizer LANGUAGES CUDA CXX)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
cmake_minimum_required(VERSION 3.21)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

# External dependencies
find_package(Boost 1.74 REQUIRED)

# Check WSL2
message(STATUS "CMake ${CMAKE_VERSION}  kernel: ${CMAKE_HOST_SYSTEM_VERSION}")
if(CMAKE_HOST_SYSTEM_VERSION MATCHES "microsoft-standard-WSL2$")
  add_compile_definitions(__WSL2__)
endif()

# Libraries
find_package(Eigen3 3.1 REQUIRED)
add_library(glso INTERFACE)
target_include_directories(glso INTERFACE include)
target_link_libraries(glso INTERFACE Boost::boost Eigen3::Eigen)

# Examples
add_executable(circle examples/circle.cu)
target_link_libraries(circle glso)

add_executable(bal examples/bal.cu)
target_link_libraries(bal glso)
