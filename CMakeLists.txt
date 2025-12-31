cmake_minimum_required(VERSION 3.21)
project(gpu_least_squares_optimizer LANGUAGES CUDA CXX)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 120)
  endif()
  message(STATUS "No CMAKE_CUDA_ARCHITECTURES specified, defaulting to: ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "CMAKE_CUDA_ARCHITECTURES specified as: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Check WSL2
message(STATUS "CMake ${CMAKE_VERSION}  kernel: ${CMAKE_HOST_SYSTEM_VERSION}")
if(CMAKE_HOST_SYSTEM_VERSION MATCHES "microsoft-standard-WSL2$")
  add_compile_definitions(__WSL2__)
endif()

# Libraries
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 3.1 REQUIRED)
find_package(cudss 0.7.0 REQUIRED)

# Setup graphite library
add_library(graphite src/eigen_solver.cpp)
target_include_directories(graphite PUBLIC include)
target_link_libraries(graphite PUBLIC CUDA::cublas Eigen3::Eigen cudss)


# Examples
option(BUILD_EXAMPLES "Build example applications" ON)

if(BUILD_EXAMPLES)
  message(STATUS "Configured to build example applications")

  add_executable(circle examples/circle.cu)
  target_link_libraries(circle graphite)


  add_executable(bal examples/bal.cu)
  target_link_libraries(bal graphite)
endif()


# Formatter
find_program(CLANG_FORMAT_BIN NAMES clang-format)
if(CLANG_FORMAT_BIN)
  file(GLOB_RECURSE FORMAT_FILES
    "include/graphite/*.cpp" "include/graphite/*.hpp" "include/graphite/*.c" "include/graphite/*.h" "include/graphite/*.cu" "include/graphite/*.cuh"
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
