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

template <typename T>
class PointDescriptor : public VertexDescriptor<T, PointDescriptor> {};

template <typename T> struct VertexTraits<T, PointDescriptor> {
  static constexpr size_t dimension = 3;
  using State = Point<T>;
  using Vertex = Point<T>;

  hd_fn static std::array<T, dimension> parameters(const Vertex &vertex) {
    return vector_to_array<T, dimension>(vertex.p);
  }

  hd_fn static void update(Vertex &vertex, const T *delta) {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> d(delta);
    vertex.p += d;
  }

  hd_fn static State get_state(const Vertex &vertex) { return vertex; }

  hd_fn static void set_state(Vertex &vertex, const State &state) {
    vertex.p = state.p;
  }
};

template <typename T>
class CameraDescriptor : public VertexDescriptor<T, CameraDescriptor> {};

template <typename T> struct VertexTraits<T, CameraDescriptor> {
  static constexpr size_t dimension = 9;
  using State = Camera<T>;
  using Vertex = Camera<T>;

  hd_fn static std::array<T, dimension> parameters(const Vertex &vertex) {
    return vector_to_array<T, dimension>(vertex.params);
  }

  hd_fn static void update(Vertex &vertex, const T *delta) {
    Eigen::Map<const Eigen::Matrix<T, 9, 1>> d(delta);
    vertex.params += d;
  }

  hd_fn static State get_state(const Vertex &vertex) { return vertex; }

  hd_fn static void set_state(Vertex &vertex, const State &state) {
    vertex.params = state.params;
  }
};

template <typename T>
class ReprojectionError
    : public AutoDiffFactorDescriptor<T, ReprojectionError> {};

template <typename T> struct FactorTraits<T, ReprojectionError> {
  static constexpr size_t dimension = 2;
  using VertexDescriptors = std::tuple<CameraDescriptor<T>, PointDescriptor<T>>;
  using ObservationType = Eigen::Vector2d;
  using ConstraintDataType = unsigned char;

  using LossType = DefaultLoss<T, dimension>;

  template <typename D, typename M>
  __device__ static void
  error(const D *camera, const D *point, const M *obs, D *error,
        const std::tuple<Camera<T> *, Point<T> *> &vertices,
        const ConstraintDataType *data) {
    bal_reprojection_error<D, M, T>(camera, point, obs, error);
  }
};

} // namespace glso

int main(void) {

  using namespace glso;

  // std::string file_path = "../data/bal/problem-16-22106-pre.txt";
  std::string file_path = "../data/bal/problem-21-11315-pre.txt";
  // std::string file_path = "../data/bal/problem-257-65132-pre.txt";
  // std::string file_path = "../data/bal/problem-356-226730-pre.txt";
  // std::string file_path = "../data/bal/problem-1778-993923-pre.txt";

  initialize_cuda();

  // Create graph
  Graph<double> graph;

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

  uninitialized_vector<Point<double>> points(num_points);
  uninitialized_vector<Camera<double>> cameras(num_cameras);

  // Create vertices
  auto point_desc = new PointDescriptor<double>();
  point_desc->reserve(num_points);
  graph.add_vertex_descriptor(point_desc);

  auto camera_desc = new CameraDescriptor<double>();
  camera_desc->reserve(num_cameras);
  graph.add_vertex_descriptor(camera_desc);

  // Create edges
  auto r_desc = graph.add_factor_descriptor<ReprojectionError<double>>(
      camera_desc, point_desc);
  r_desc->reserve(num_observations);

  const auto loss = DefaultLoss<double, 2>();
  Eigen::Matrix2d precision_matrix = Eigen::Matrix2d::Identity();

  auto start = std::chrono::steady_clock::now();

  // Read observations and create constraints
  for (size_t i = 0; i < num_observations; ++i) {
    size_t camera_idx, point_idx;
    double x, y;

    // Read observation data
    file >> camera_idx >> point_idx >> x >> y;

    // Store the observation
    const Eigen::Vector2d obs(x, y);

    // Add constraint to the graph
    r_desc->add_factor({camera_idx, point_idx}, obs, precision_matrix.data(), 0,
                       loss);
  }
  std::cout << "Adding constraints took "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             start)
                   .count()
            << " seconds." << std::endl;

  start = std::chrono::steady_clock::now();
  // Create all camera vertices
  for (size_t i = 0; i < num_cameras; ++i) {
    std::array<double, 9> camera_params;
    for (size_t j = 0; j < 9; ++j) {
      file >> camera_params[j];
    }
    cameras[i] = Camera<double>(camera_params);
    camera_desc->add_vertex(i, &cameras[i]);
  }

  std::cout << "Adding cameras took "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             start)
                   .count()
            << " seconds." << std::endl;

  start = std::chrono::steady_clock::now();
  // Create all point vertices
  for (size_t i = 0; i < num_points; ++i) {
    double point_params[3];
    for (size_t j = 0; j < 3; ++j) {
      file >> point_params[j];
    }
    points[i] =
        Point<double>(point_params[0], point_params[1], point_params[2]);
    point_desc->add_vertex(i, &points[i]);
  }
  std::cout << "Adding points took "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             start)
                   .count()
            << " seconds." << std::endl;
  file.close();

  // Configure solver
  auto preconditioner =
      std::make_shared<glso::BlockJacobiPreconditioner<double>>();
  PCGSolver<double> solver(50, 1e-6, preconditioner);

  // Optimize
  constexpr size_t iterations = 50;
  std::cout << "Graph built with " << num_cameras << " cameras, " << num_points
            << " points, and " << r_desc->count() << " observations."
            << std::endl;
  std::cout << "Optimizing!" << std::endl;

  start = std::chrono::steady_clock::now();
  optimizer::levenberg_marquardt(&graph, &solver, iterations, 1e-6);
  auto end = std::chrono::steady_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Optimization took " << elapsed.count() << " seconds."
            << std::endl;

  auto mse = graph.chi2() / num_observations;
  std::cout << "MSE: " << mse << std::endl;
  std::cout << "Half MSE: " << mse / 2 << std::endl;

  return 0;
}
