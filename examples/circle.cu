#include <Eigen/Core>
#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <glso/core.hpp>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace glso {

// Point definition
template <typename T> using Point = Eigen::Matrix<T, 2, 1>;

// Traits for Point
template <typename T> struct PointTraits {
  static constexpr size_t dimension = 2;
  using Vertex = Point<T>;

  hd_fn static std::array<T, dimension> parameters(const Vertex &vertex) {
    return vector_to_array<T, dimension>(vertex);
  }

  hd_fn static void update(Vertex &vertex, const T *delta) {
    Eigen::Map<const Eigen::Matrix<T, dimension, 1>> d(delta);
    vertex += d;
  }
};

template <typename T>
using PointDescriptor = VertexDescriptor<T, PointTraits<T>>;

// Factor traits for the circle constraint
template <typename T> struct CircleFactorTraits {
  static constexpr size_t dimension = 1;
  using VertexDescriptors = std::tuple<PointDescriptor<T>>;
  using Observation = T;
  using Data = unsigned char;
  using Loss = DefaultLoss<T, dimension>;
  using Differentiation = DifferentiationMode::Auto;

  template <typename D, typename M>
  hd_fn static void error(const D *point, const M *obs, D *error,
                          const std::tuple<Point<T> *> &vertices,
                          const Data *data) {
    auto x = point[0];
    auto y = point[1];
    auto r = obs[0];
    error[0] = x * x + y * y - r * r;
  }
};

template <typename T>
using CircleFactor = FactorDescriptor<T, CircleFactorTraits<T>>;

} // namespace glso

int main(void) {

  using namespace glso;

  initialize_cuda();

  // Create graph
  Graph<double> graph;

  const size_t num_vertices = 5;

  // Create vertices
  auto point_desc = new PointDescriptor<double>();
  point_desc->reserve(num_vertices);
  graph.add_vertex_descriptor(point_desc);

  double center[2] = {0.0, 0.0};

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(0.0, 2 * M_PI);

  const double radius = 4.0;
  const double sigma = 0.3;

  std::normal_distribution<double> n1(0.0, sigma);
  std::normal_distribution<double> n2(0.0, sigma);

  managed_vector<Point<double>> pts(num_vertices); // addresses must not change
  constexpr auto id_offset = 10; // user provides arbitrary ids

  for (size_t vertex_id = 0; vertex_id < num_vertices; ++vertex_id) {
    double angle = dist(gen);
    double point[2] = {center[0] + radius * cos(angle),
                       center[1] + radius * sin(angle)};
    point[0] += n1(gen);
    point[1] += n2(gen);
    pts[vertex_id] = Point<double>(point[0], point[1]);
    std::cout << "Adding point " << vertex_id << "=(" << point[0] << ", "
              << point[1] << ") with radius="
              << sqrt(point[0] * point[0] + point[1] * point[1]) << std::endl;
    point_desc->add_vertex(vertex_id + id_offset, &pts[vertex_id]);
  }
  // Create edges
  auto factor_desc =
      graph.add_factor_descriptor<CircleFactor<double>>(point_desc);
  factor_desc->reserve(num_vertices);

  const auto loss = DefaultLoss<double, 1>();

  for (size_t vertex_id = 0; vertex_id < num_vertices; ++vertex_id) {
    factor_desc->add_factor({vertex_id + id_offset}, radius, nullptr, 0, loss);
  }

  // Set the last vertex as fixed
  point_desc->set_fixed(num_vertices - 1 + id_offset, true);

  // Configure solver
  auto preconditioner =
      std::make_shared<glso::IdentityPreconditioner<double>>();
  glso::PCGSolver<double> solver(50, 1e-6, preconditioner);

  // Optimize
  constexpr size_t iterations = 10;
  std::cout << "Graph built with " << num_vertices << " vertices and "
            << factor_desc->count() << " factors." << std::endl;
  std::cout << "Optimizing!" << std::endl;

  auto start = std::chrono::steady_clock::now();
  optimizer::levenberg_marquardt<double>(&graph, &solver, iterations, 1e-6);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Optimization took " << elapsed.count() << " seconds."
            << std::endl;

  // Read back optimized values
  for (size_t vertex_id = 0; vertex_id < num_vertices; ++vertex_id) {
    const auto point = point_desc->get_vertex(vertex_id + id_offset);
    const auto &p = *point;
    auto x = p(0);
    auto y = p(1);
    std::cout << "Optimized point " << vertex_id << "=(" << x << ", " << y
              << ") with radius=" << sqrt(x * x + y * y) << std::endl;
  }

  return 0;
}
