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
#include <glso/types.hpp>

namespace glso {

template <typename T> using Point = Eigen::Matrix<T, 3, 1>;

template <typename T> using Camera = Eigen::Matrix<T, 9, 1>;

template <typename T> struct PointTraits {
  static constexpr size_t dimension = 3;
  using Vertex = Point<T>;

  hd_fn static std::array<T, dimension> parameters(const Vertex &vertex) {
    return vector_to_array<T, dimension>(vertex);
  }

  hd_fn static void update(Vertex &vertex, const T *delta) {
    Eigen::Map<const Eigen::Matrix<T, dimension, 1>> d(delta);
    vertex += d;
  }
};

template <typename T> struct CameraTraits {
  static constexpr size_t dimension = 9;
  using State = Camera<T>; // State can be optionally defined
  using Vertex = Camera<T>;

  hd_fn static std::array<T, dimension> parameters(const Vertex &vertex) {
    return vector_to_array<T, dimension>(vertex);
  }

  hd_fn static void update(Vertex &vertex, const T *delta) {
    Eigen::Map<const Eigen::Matrix<T, dimension, 1>> d(delta);
    vertex += d;
  }

  // Defining the state requires custom setters and getters
  hd_fn static State get_state(const Vertex &vertex) { return vertex; }

  hd_fn static void set_state(Vertex &vertex, const State &state) {
    vertex = state;
  }
};

template <typename T, typename S>
using PointDescriptor = VertexDescriptor<T, S, PointTraits<T>>;

template <typename T, typename S>
using CameraDescriptor = VertexDescriptor<T, S, CameraTraits<T>>;

template <typename T, typename S> struct ReprojectionErrorTraits {
  static constexpr size_t dimension = 2;
  using VertexDescriptors =
      std::tuple<CameraDescriptor<T, S>, PointDescriptor<T, S>>;
  using Observation = Eigen::Matrix<T, dimension, 1>;
  using Data = unsigned char;
  using Loss = DefaultLoss<T, dimension>;
  using Differentiation = DifferentiationMode::Auto;
  // using Differentiation = DifferentiationMode::Manual;

  template <typename D, typename M>
  hd_fn static void
  error(const D *camera, const D *point, const M *obs, D *error,
        const std::tuple<Camera<T> *, Point<T> *> &vertices, const Data *data) {
    bal_reprojection_error<D, M, T>(camera, point, obs, error);
    // bal_reprojection_error_simple<D, M, T>(camera, point, obs, error);
  }

  /*
  template <typename D, size_t I>
  hd_fn static void jacobian(const Camera<T>* camera, const Point<T> *point, const Observation *obs, D *jacobian,
                             const Data *data) {
    
          // Camera Jacobian
      auto &cam = *camera;
      auto &X = *point;

      Eigen::Map<Eigen::Matrix<T, 2, 9>> Jcam(jacobian);


      // error calc
      // Eigen::Map<const Eigen::Matrix<D, 3, 1>> X(point);
      // Eigen::Map<const Eigen::Matrix<D, 9, 1>> cam(camera);
      Eigen::Map<const Eigen::Matrix<T, 2, 1>> observation(obs->data());

      // Extract rotation vector and build rotation matrix using Eigen's AngleAxis
      Eigen::Matrix<D, 3, 1> rvec = cam.template head<3>();
      D theta = rvec.norm();
      Eigen::Matrix<D, 3, 3> R = Eigen::Matrix<D, 3, 3>::Identity();

      // if (theta > D(0)) {
      Eigen::AngleAxis<D> angle_axis(theta, rvec / theta);
      R = angle_axis.toRotationMatrix();
      // }

      // Apply rotation and translation
      Eigen::Matrix<D, 3, 1> t = cam.template segment<3>(3);
      Eigen::Matrix<D, 3, 1> P = R * X + t;

      // Perspective division
      Eigen::Matrix<D, 2, 1> p = -P.template head<2>() / P(2);

      // Radial distortion
      D f = cam(6);
      D k1 = cam(7);
      D k2 = cam(8);
      D r2 = p.squaredNorm();
      D radial_distortion = D(1.0) + k1 * r2 + k2 * r2 * r2;
      // Project to pixel coordinates and compute reprojection error
      Eigen::Matrix<D, 2, 1> reprojection_error =
          f * radial_distortion * p - observation.template cast<D>();
    
    
    
    if constexpr (I == 0) {
            // Camera Jacobian

      // Jacobian calculation
      Jcam.setZero();

      // wrt f
      const auto dres_df = (radial_distortion*p).transpose();
      Jcam.block<2, 1>(6, 0) = dres_df;

      // wrt radial distortion
      const auto dres_drad = f*p;

      // wrt k1
      const auto drad_dk1 = r2;
      const auto dres_dk1 = dres_drad * drad_dk1;
      Jcam.block<2, 1>(7, 0) = dres_dk1;

      // wrt k2
      const auto drad_dk2 = r2 * r2;
      const auto dres_dk2 = dres_drad * drad_dk2;
      Jcam.block<2, 1>(8, 0) = dres_dk2;

      // wrt translation (paramters 3-5)
      const auto drad_dr2 = k1 + 2*k2*r2;
      const auto dr2_dp = 2 * p;

      Eigen::Matrix<D, 2, 3> dp_dP = 
          Eigen::Matrix<D, 2, 3>::Zero();
      dp_dP(0, 0) = -1 / P(2);
      dp_dP(0, 2) = p(0) / (P(2) * P(2));
      dp_dP(1, 1) = -1 / P(2);
      dp_dP(1, 2) = p(1) / (P(2) * P(2));


      const auto dP_dt = Eigen::Matrix<D, 3, 3>::Identity();


    }
    else if constexpr (I == 1) {
      // Point Jacobian
      // auto &p = *point;
      // auto &cam = *camera;

      Eigen::Map<Eigen::Matrix<T, 2, 3>> Jpoint(jacobian);
      Jpoint.setZero();
      

      // auto x = P(0);
      // auto y = P(1);
      // auto z = P(2);

      // Jpoint <<
      //     -f * radial_distortion / z, 0, f * radial_distortion * x / (z * z),
      //     0, -f * radial_distortion / z, f * radial_distortion * y / (z * z);
    }
  }
  */

};

template <typename T, typename S>
using ReprojectionError = FactorDescriptor<T, S, ReprojectionErrorTraits<T, S>>;

} // namespace glso

int main(void) {

  using namespace glso;

  // using FP = double;
  // using SP = double;
  // using SP = FP;
  using FP = float;
  using SP = float;
  // using SP = __nv_bfloat16;
  // using SP = half;

  // std::string file_path = "../data/bal/problem-16-22106-pre.txt";
  // std::string file_path = "../data/bal/problem-21-11315-pre.txt";
  // std::string file_path = "../data/bal/problem-257-65132-pre.txt";
  // std::string file_path = "../data/bal/problem-356-226730-pre.txt";
  std::string file_path = "../data/bal/problem-1778-993923-pre.txt";
  // std::string file_path = "../data/bal/problem-4585-1324582-pre.txt";
  // std::string file_path = "../data/bal/problem-13682-4456117-pre.txt";

  initialize_cuda();

  // Create graph
  Graph<FP, SP> graph;

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
  auto point_desc = PointDescriptor<FP, SP>();
  point_desc.reserve(num_points);
  graph.add_vertex_descriptor(&point_desc);

  auto camera_desc = CameraDescriptor<FP, SP>();
  camera_desc.reserve(num_cameras);
  graph.add_vertex_descriptor(&camera_desc);

  // Create edges
  auto r_desc = ReprojectionError<FP, SP>(&camera_desc, &point_desc);
  r_desc.reserve(num_observations);
  graph.add_factor_descriptor(&r_desc);

  const auto loss = DefaultLoss<FP, 2>();
  Eigen::Matrix<SP, 2, 2> precision_matrix =
      Eigen::Matrix<SP, 2, 2>::Identity();

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
    r_desc.add_factor({camera_idx, point_idx}, obs, precision_matrix.data(), 0,
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
    FP camera_params[9];
    for (size_t j = 0; j < 9; ++j) {
      file >> camera_params[j];
    }
    cameras[i] = Eigen::Map<Camera<FP>>(camera_params);
    camera_desc.add_vertex(i, &cameras[i]);
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
    points[i] = Eigen::Map<Point<FP>>(point_params);
    point_desc.add_vertex(i, &points[i]);
  }
  std::cout << "Adding points took "
            << std::chrono::duration<FP>(std::chrono::steady_clock::now() -
                                         start)
                   .count()
            << " seconds." << std::endl;
  file.close();

  // Configure solver
  glso::BlockJacobiPreconditioner<FP, SP> preconditioner;
  // glso::IdentityPreconditioner<FP, SP> preconditioner;
  PCGSolver<FP, SP> solver(100, 1e-1, 1.0, &preconditioner);

  // Optimize
  constexpr size_t iterations = 50;
  std::cout << "Graph built with " << num_cameras << " cameras, " << num_points
            << " points, and " << r_desc.count() << " observations."
            << std::endl;
  std::cout << "Optimizing!" << std::endl;

  start = std::chrono::steady_clock::now();
  optimizer::levenberg_marquardt<FP, SP>(&graph, &solver, iterations, 1e-4);
  auto end = std::chrono::steady_clock::now();

  std::chrono::duration<FP> elapsed = end - start;
  std::cout << "Optimization took " << elapsed.count() << " seconds."
            << std::endl;

  auto mse = graph.chi2() / num_observations;
  std::cout << "MSE: " << mse << std::endl;
  std::cout << "Half MSE: " << mse / 2 << std::endl;

  return 0;
}
