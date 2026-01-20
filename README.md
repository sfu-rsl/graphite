# Graphite

Graphite is a GPU-accelerated nonlinear least squares graph optimization framework based on CUDA. It lets you define an optimization problem in terms of unary, binary, and n-ary constraints (e.g. pose graph optimization, bundle adjustment), using data structures defined in C++, which may be useful for applications in robotics and computer vision such as SLAM. Graphite also supports configurable floating point precisions as well as mixed-precision solving.

⚠️ Graphite is experimental. There may be several bugs, performance issues, and limitations. The interface and implementation may change over time.

For more details, refer to the [paper](https://arxiv.org/abs/2509.26581).

Supported linear solvers:
- Preconditioned Conjugate Gradients
- Eigen LDLT
- cuDSS

Supported algorithms:
- Levenberg-Marquardt

## Building

You need a recent version of the CUDA Toolkit (e.g. >= 12.0), Eigen3, and cuDSS 0.7.0. Graphite can be built using CMake. A Dockerfile for development is also included, which can be used to create a devcontainer for VS Code (requires the NVIDIA Container Toolkit).

You can build Graphite and its examples using the following commands:

```bash
git clone https://github.com/sfu-rsl/graphite.git
cmake -DCMAKE_BUILD_TYPE=Release graphite -B build
cmake --build build
```

To instead use Graphite as a library inside an existing CMake project,
you can add it as a dependency using the [`add_subdirectory(...)`](https://cmake.org/cmake/help/latest/command/add_subdirectory.html) command in your `CMakeLists.txt` file. You can then link `graphite` to your build target. 

You may also have to adjust the following settings near the top, before adding the subdirectory:
```cmake
project(YourProject LANGUAGES CUDA CXX)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda --default-stream per-thread")
set(CMAKE_CUDA_ARCHITECTURES 86) # You need to actually look up the correct CC number for your GPU
```

Graphite code should be called inside a .cu file to avoid compile errors.

## Examples

See the [examples](examples) folder. There are two examples:

- [`circle.cu`](examples/circle.cu) - Optimizes noisy 2D points along the radius of a circle
- [`bal.cu`](examples/bal.cu) - Performs bundle adjustment

## License

Graphite is released under the [MIT License](LICENSE.md).

## Contributing

If you would like to contribute features, bug fixes, code improvements, or tests, please open an issue or discussion first. Questions about usage are better suited for discussion.