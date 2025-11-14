# Graphite

Graphite is a GPU-accelerated graph optimization framework based on CUDA. It lets you define an optimization problem in terms of unary, binary, and n-ary constraints, using types defined in C++, which may be useful for applications in robotics and computer vision such as SLAM. It also supports configurable floating point precisions as well as mixed-precision solving.

⚠️ Graphite is experimental. There may be several bugs, performance issues, and limitations. The interface and implementation may change over time.

For more details, refer to the [paper](https://arxiv.org/abs/2509.26581).

## Building

You need a recent version of the CUDA Toolkit (e.g. >= 12.0), as well as Eigen3 and boost. Graphite can be built using CMake. A Dockerfile for development is also included, which can be used to create a devcontainer for VS Code (requires the NVIDIA Container Toolkit).

## Examples

See the [examples](examples) folder. There are two examples:

- [`circle.cu`](examples/circle.cu) - Optimizes noisy 2D points along the radius of a circle
- [`bal.cu`](examples/bal.cu) - Performs bundle adjustment

## License

Graphite is released under the [MIT License](LICENSE.md).