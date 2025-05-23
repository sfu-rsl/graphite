#include <array>
#include <chrono>
#include <glso/core.hpp>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace glso {

template <typename T> class Point {
public:
  T x;
  T y;

  Point() : x(0), y(0) {}

  Point(T x, T y) : x(x), y(y) {}

  using State = Point<T>;
  constexpr static size_t dimension = 2;

  SOLVER_FUNC std::array<T, dimension> parameters() const { return {x, y}; }

  SOLVER_FUNC void update(const T *delta) {
    x += delta[0];
    y += delta[1];
  }

  SOLVER_FUNC State get_state() const { return *this; }

  SOLVER_FUNC void set_state(const State &state) {
    x = state.x;
    y = state.y;
  }
};

template <typename T>
class PointSet : public VertexDescriptor<T, Point<T>, PointSet> {};

using ObsType = double;
using ConstraintData = unsigned char;
template <typename T>
class CircleFactor
    : public AutoDiffFactorDescriptor<T, 1, ObsType, ConstraintData, HuberLoss,
                                      CircleFactor, PointSet<T>> {
public:
  template <typename D, typename M>
  __device__ static void error(const D *point, const M *obs, D *error,
                               const std::tuple<Point<T> *> &vertices,
                               const ConstraintData *data) {

    D x = point[0];
    D y = point[1];
    D r = obs[0];

    error[0] = x * x + y * y - r * r;
  }
};
} // namespace glso

int main(void) {

  using namespace glso;

  initialize_cuda();

  // Create graph
  Graph<double> graph;

  const size_t num_vertices = 5;

  // Create vertices
  PointSet<double> *points = new PointSet<double>();
  points->reserve(num_vertices);
  graph.add_vertex_descriptor(points);

  double center[2] = {0.0, 0.0};

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(0.0, 2 * M_PI);

  const double radius = 4.0;
  const double sigma = 0.3;

  std::normal_distribution<double> n1(0.0, sigma);
  std::normal_distribution<double> n2(0.0, sigma);

  thrust::universal_vector<Point<double>> pts;
  constexpr auto id_offset = 10;
  pts.reserve(num_vertices); // addresses must not change

  for (size_t vertex_id = 0; vertex_id < num_vertices; ++vertex_id) {
    double angle = dist(gen);
    double point[2] = {center[0] + radius * cos(angle),
                       center[1] + radius * sin(angle)};

    point[0] += n1(gen);
    point[1] += n2(gen);
    pts.push_back(Point<double>(point[0], point[1]));
    std::cout << "Adding point " << vertex_id << "=(" << point[0] << ", "
              << point[1] << ") with radius="
              << sqrt(point[0] * point[0] + point[1] * point[1]) << std::endl;
    points->add_vertex(vertex_id + id_offset, &pts[vertex_id]);
  }

  // Create edges
  auto factor_desc = graph.add_factor_descriptor<CircleFactor<double>>(points);
  factor_desc->reserve(num_vertices);

  // const auto loss = DefaultLoss<double, 1>();
  const auto loss = HuberLoss<double, 1>(200);

  for (size_t vertex_id = 0; vertex_id < num_vertices; ++vertex_id) {
    factor_desc->add_factor({vertex_id + id_offset}, {radius}, nullptr, 0,
                            loss);
  }

  // Set the last vertex as fixed
  points->set_fixed(num_vertices - 1 + id_offset, true);

  // Optimize
  constexpr size_t iterations = 10;
  Optimizer opt;
  std::cout << "Graph built with " << num_vertices << " vertices and "
            << factor_desc->count() << " factors." << std::endl;
  std::cout << "Optimizing!" << std::endl;

  auto start = std::chrono::steady_clock::now();
  opt.optimize(&graph, iterations);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Optimization took " << elapsed.count() << " seconds."
            << std::endl;

  // Read back optimized values
  for (size_t vertex_id = 0; vertex_id < num_vertices; ++vertex_id) {
    const auto point = points->get_vertex(vertex_id + id_offset);
    const auto [x, y] = *point;

    std::cout << "Optimized point " << vertex_id << "=(" << x << ", " << y
              << ") with radius=" << sqrt(x * x + y * y) << std::endl;
  }

  return 0;
}
