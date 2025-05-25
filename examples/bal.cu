#include <Eigen/Core>
#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "reprojection_error.cuh"
#include <glso/core.hpp>
#include <glso/solver.hpp>

namespace glso {

template <typename T> class Point {
public:
  Eigen::Matrix<T, 3, 1> p;

  Point() = default;
  Point(T x, T y, T z) : p(x, y, z) {}
};

template <typename T> class Camera {
public:
  Eigen::Matrix<T, 9, 1> params;

  Camera() = default;
  Camera(const std::array<T, 9> &cam) : params(cam.data()) {}
};

template <typename T> struct PointTraits {
  static constexpr size_t dimension = 3;
  using Vertex = Point<T>;

  hd_fn static std::array<T, dimension> parameters(const Vertex &vertex) {
    return vector_to_array<T, dimension>(vertex.p);
  }

  hd_fn static void update(Vertex &vertex, const T *delta) {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> d(delta);
    vertex.p += d;
  }
};

template <typename T> struct CameraTraits {
  static constexpr size_t dimension = 9;
  using State = Camera<T>; // State can be optionally defined
  using Vertex = Camera<T>;

  hd_fn static std::array<T, dimension> parameters(const Vertex &vertex) {
    return vector_to_array<T, dimension>(vertex.params);
  }

  hd_fn static void update(Vertex &vertex, const T *delta) {
    Eigen::Map<const Eigen::Matrix<T, 9, 1>> d(delta);
    vertex.params += d;
  }

  // Defining the state requires custom setters and getters
  hd_fn static State get_state(const Vertex &vertex) { return vertex; }

  hd_fn static void set_state(Vertex &vertex, const State &state) {
    vertex = state;
  }
};

template <typename T>
using PointDescriptor = VertexDescriptor<T, PointTraits<T>>;

template <typename T>
using CameraDescriptor = VertexDescriptor<T, CameraTraits<T>>;

template <typename T> struct ReprojectionErrorTraits {
  static constexpr size_t dimension = 2;
  using VertexDescriptors = std::tuple<CameraDescriptor<T>, PointDescriptor<T>>;
  using Observation = Eigen::Matrix<T, dimension, 1>;
  using Data = unsigned char;
  using Loss = DefaultLoss<T, dimension>;
  using Differentiation = DifferentiationMode::Auto;

  template <typename D, typename M>
  __device__ static void
  error(const D *camera, const D *point, const M *obs, D *error,
        const std::tuple<Camera<T> *, Point<T> *> &vertices, const Data *data) {
    bal_reprojection_error<D, M, T>(camera, point, obs, error);
  }
};

template <typename T>
using ReprojectionError = FactorDescriptor<T, ReprojectionErrorTraits<T>>;

} // namespace glso

int main(void) {

  using namespace glso;

  using FP = double;
  // using FP = float;

  // std::string file_path = "../data/bal/problem-16-22106-pre.txt";
  std::string file_path = "../data/bal/problem-21-11315-pre.txt";
  // std::string file_path = "../data/bal/problem-257-65132-pre.txt";
  // std::string file_path = "../data/bal/problem-356-226730-pre.txt";
  // std::string file_path = "../data/bal/problem-1778-993923-pre.txt";

  initialize_cuda();

  // Create graph
  Graph<FP> graph;

  size_t num_points = 0;
  size_t num_cameras = 0;
  size_t num_observations = 0;

  // Open the file
  std::ifstream file(file_path);
  if (!file.is_open()) {
    std::cerr << "Error: Unable to open file " << file_path << std::endl;
    return -1;
  }

  // Read the number of cameras, points, and observations
  file >> num_cameras >> num_points >> num_observations;
  std::cout << "Number of cameras: " << num_cameras << std::endl;
  std::cout << "Number of points: " << num_points << std::endl;
  std::cout << "Number of observations: " << num_observations << std::endl;

  uninitialized_vector<Point<FP>> points(num_points);
  uninitialized_vector<Camera<FP>> cameras(num_cameras);

  // Create vertices
  auto point_desc = new PointDescriptor<FP>();
  point_desc->reserve(num_points);
  graph.add_vertex_descriptor(point_desc);

  auto camera_desc = new CameraDescriptor<FP>();
  camera_desc->reserve(num_cameras);
  graph.add_vertex_descriptor(camera_desc);

  // Create edges
  auto r_desc = graph.add_factor_descriptor<ReprojectionError<FP>>(camera_desc,
                                                                   point_desc);
  r_desc->reserve(num_observations);

  const auto loss = DefaultLoss<FP, 2>();
  Eigen::Matrix<FP, 2, 2> precision_matrix =
      Eigen::Matrix<FP, 2, 2>::Identity();

  auto start = std::chrono::steady_clock::now();

  // Read observations and create constraints
  for (size_t i = 0; i < num_observations; ++i) {
    size_t camera_idx, point_idx;
    FP x, y;

    // Read observation data
    file >> camera_idx >> point_idx >> x >> y;

    // Store the observation
    const Eigen::Matrix<FP, 2, 1> obs(x, y);

    // Add constraint to the graph
    r_desc->add_factor({camera_idx, point_idx}, obs, precision_matrix.data(), 0,
                       loss);
  }
  std::cout << "Adding constraints took "
            << std::chrono::duration<FP>(std::chrono::steady_clock::now() -
                                         start)
                   .count()
            << " seconds." << std::endl;

  start = std::chrono::steady_clock::now();
  // Create all camera vertices
  for (size_t i = 0; i < num_cameras; ++i) {
    std::array<FP, 9> camera_params;
    for (size_t j = 0; j < 9; ++j) {
      file >> camera_params[j];
    }
    cameras[i] = Camera<FP>(camera_params);
    camera_desc->add_vertex(i, &cameras[i]);
  }

  std::cout << "Adding cameras took "
            << std::chrono::duration<FP>(std::chrono::steady_clock::now() -
                                         start)
                   .count()
            << " seconds." << std::endl;

  start = std::chrono::steady_clock::now();
  // Create all point vertices
  for (size_t i = 0; i < num_points; ++i) {
    FP point_params[3];
    for (size_t j = 0; j < 3; ++j) {
      file >> point_params[j];
    }
    points[i] = Point<FP>(point_params[0], point_params[1], point_params[2]);
    point_desc->add_vertex(i, &points[i]);
  }
  std::cout << "Adding points took "
            << std::chrono::duration<FP>(std::chrono::steady_clock::now() -
                                         start)
                   .count()
            << " seconds." << std::endl;
  file.close();

  // Configure solver
  auto preconditioner = std::make_shared<glso::BlockJacobiPreconditioner<FP>>();
  PCGSolver<FP> solver(50, 1e-6, preconditioner);

  // Optimize
  constexpr size_t iterations = 50;
  std::cout << "Graph built with " << num_cameras << " cameras, " << num_points
            << " points, and " << r_desc->count() << " observations."
            << std::endl;
  std::cout << "Optimizing!" << std::endl;

  start = std::chrono::steady_clock::now();
  optimizer::levenberg_marquardt<FP>(&graph, &solver, iterations, 1e-6);
  auto end = std::chrono::steady_clock::now();

  std::chrono::duration<FP> elapsed = end - start;
  std::cout << "Optimization took " << elapsed.count() << " seconds."
            << std::endl;

  auto mse = graph.chi2() / num_observations;
  std::cout << "MSE: " << mse << std::endl;
  std::cout << "Half MSE: " << mse / 2 << std::endl;

  return 0;
}
