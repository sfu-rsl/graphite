project(gpu_least_squares_optimizer LANGUAGES CUDA CXX)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")
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
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 3.1 REQUIRED)
add_library(glso INTERFACE)
target_include_directories(glso INTERFACE include)
target_link_libraries(glso INTERFACE Boost::boost Eigen3::Eigen CUDA::cublas)

# Examples
add_executable(circle examples/circle.cu)
target_link_libraries(circle glso)

add_executable(bal examples/bal.cu)
target_link_libraries(bal glso)

# Formatter
find_program(CLANG_FORMAT_BIN NAMES clang-format)
if(CLANG_FORMAT_BIN)
  file(GLOB_RECURSE FORMAT_FILES
    "include/*.cpp" "include/*.hpp" "include/*.c" "include/*.h" "include/*.cu" "include/*.cuh"
    "examples/*.cpp" "examples/*.hpp" "examples/*.c" "examples/*.h" "examples/*.cu" "examples/*.cuh"
  )
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_BIN}
      -i
      --style=file
      ${FORMAT_FILES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Formatting source files"
  )
else()
  message(WARNING "clang-format not found! 'format' target will not be available.")
endif()
