/** @file bal.cu
  A bundle adjustment example using the BAL dataset.
*/
#include <Eigen/Core>
#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "argparse/argparse.hpp"
#include "bal.cuh"
#include <graphite/optimizer/levenberg_marquardt.hpp>
#include <graphite/preconditioner/block_jacobi.hpp>
#include <graphite/solver/cudss.hpp>
#include <graphite/solver/cudss_schur.hpp>
#include <graphite/solver/eigen.hpp>
#include <graphite/solver/eigen_schur.hpp>
#include <graphite/solver/pcg.hpp>
#include <graphite/solver/solver.hpp>
#include <graphite/stream.hpp>
#include <graphite/types.hpp>

template <typename T> const char *get_type_name() {
  if (std::is_same<T, double>::value) {
    return "double";
  } else if (std::is_same<T, float>::value) {
    return "float";
  } else if (std::is_same<T, __nv_bfloat16>::value) {
    return "bfloat16";
  } else {
    return "unknown";
  }
}

template <typename FP, typename SP>
void bundle_adjustment(argparse::ArgumentParser &program) {

  using namespace graphite;

  std::cout << "Running bundle adjustment with graph precision = "
            << get_type_name<FP>()
            << " and solver precision = " << get_type_name<SP>() << std::endl;
  std::string file_path = program.get<std::string>("file");

  // Initialize CUDA
  cudaSetDevice(0);

  // Create graph
  Graph<FP, SP> graph;

  size_t num_points = 0;
  size_t num_cameras = 0;
  size_t num_observations = 0;

  // Open the file
  std::ifstream file(file_path);
  if (!file.is_open()) {
    std::cerr << "Error: Unable to open file " << file_path << std::endl;
    throw std::runtime_error("File open error");
  }

  // Read the number of cameras, points, and observations
  file >> num_cameras >> num_points >> num_observations;
  std::cout << "Number of cameras: " << num_cameras << std::endl;
  std::cout << "Number of points: " << num_points << std::endl;
  std::cout << "Number of observations: " << num_observations << std::endl;

  managed_vector<Point<FP>> points(num_points);
  managed_vector<Camera<FP>> cameras(num_cameras);

  // Create vertices
  auto point_desc = PointDescriptor<FP, SP>();
  point_desc.reserve(num_points);
  graph.add_vertex_descriptor(&point_desc);

  auto camera_desc = CameraDescriptor<FP, SP>();
  camera_desc.reserve(num_cameras);
  graph.add_descriptor(&camera_desc);

  // Create edges
  auto r_desc = ReprojectionError<FP, SP>(&camera_desc, &point_desc);
  r_desc.reserve(num_observations);
  graph.add_descriptor(&r_desc);
  // r_desc.set_jacobian_storage(false);

  auto start = std::chrono::steady_clock::now();

  // Read observations and create constraints
  for (size_t i = 0; i < num_observations; ++i) {
    size_t camera_idx, point_idx;
    FP x, y;

    // Read observation data
    file >> camera_idx >> point_idx >> x >> y;

    // Store the observation
    const Eigen::Matrix<FP, 2, 1> obs(x, y);

    // Add constraint to the graph (values for precision matrix, constraint
    // data, and loss function) can be passed in as additional arguments
    r_desc.add_factor({camera_idx, point_idx + num_cameras}, obs);
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
    point_desc.add_vertex(i + num_cameras, &points[i]);
  }
  std::cout << "Adding points took "
            << std::chrono::duration<FP>(std::chrono::steady_clock::now() -
                                         start)
                   .count()
            << " seconds." << std::endl;
  file.close();

  // Configure solver
  graphite::BlockJacobiPreconditioner<FP, SP> preconditioner;

  const auto solver_type = program.get<std::string>("--solver");
  const bool use_schur_solver =
      (solver_type == "eigen-schur") || (solver_type == "cudss-schur");
  point_desc.set_eliminate(use_schur_solver);

  std::unique_ptr<Solver<FP, SP>> solver_ptr;

  if (solver_type == "pcg") {
    std::cout << "Using PCG solver." << std::endl;
    const auto pcg_iter = program.get<size_t>("--pcg_iterations");
    const auto pcg_tol = program.get<double>("--pcg_tolerance");
    const auto rej_ratio = program.get<double>("--rejection_ratio");
    solver_ptr = std::make_unique<PCGSolver<FP, SP>>(
        pcg_iter, pcg_tol, rej_ratio,
        &preconditioner); // good parameters: 10, 1e0, 5
  } else if (solver_type == "eigen") {
    if constexpr (std::is_same<SP, __nv_bfloat16>::value) {
      std::cerr << "Eigen solver does not support bfloat16 precision."
                << std::endl;
    } else {
      std::cout << "Using Eigen LDLT solver." << std::endl;
      solver_ptr = std::make_unique<EigenLDLTSolver<FP, SP>>();
    }
  } else if (solver_type == "eigen-schur") {
    if constexpr (std::is_same<SP, __nv_bfloat16>::value) {
      std::cerr << "Eigen Schur solver does not support bfloat16 precision."
                << std::endl;
    } else if constexpr (!std::is_same<FP, SP>::value) {
      std::cerr << "Eigen Schur solver requires graph and solver precision to "
                   "be the same."
                << std::endl;
    } else {
      std::cout << "Using Eigen Schur LDLT solver." << std::endl;
      solver_ptr = std::make_unique<EigenSchurLDLTSolver<FP, SP>>();
    }
  } else if (solver_type == "cudss") {
    if constexpr (std::is_same<SP, __nv_bfloat16>::value) {
      std::cerr << "cuDSS solver does not support bfloat16 precision."
                << std::endl;
    } else if constexpr (!std::is_same<FP, SP>::value) {
      std::cerr
          << "cuDSS solver requires graph and solver precision to be the same."
          << std::endl;
    } else {
      cudssSolverOptions cudss_options;
      const auto hybrid_memory_mb = program.get<int64_t>("--hybrid_memory");
      cudss_options.hybrid_memory = 1000000 * hybrid_memory_mb;
      std::cout << "Using cuDSS solver." << std::endl;
      solver_ptr = std::make_unique<cudssSolver<FP, SP>>(cudss_options);
    }
  } else if (solver_type == "cudss-schur") {
    if constexpr (std::is_same<SP, __nv_bfloat16>::value) {
      std::cerr << "cuDSS Schur solver does not support bfloat16 precision."
                << std::endl;
    } else if constexpr (!std::is_same<FP, SP>::value) {
      std::cerr << "cuDSS Schur solver requires graph and solver precision to "
                   "be the same."
                << std::endl;
    } else {
      cudssSolverOptions cudss_options;
      const auto hybrid_memory_mb = program.get<int64_t>("--hybrid_memory");
      cudss_options.hybrid_memory = 1000000 * hybrid_memory_mb;
      std::cout << "Using cuDSS Schur solver with hybrid memory limit = "
                << cudss_options.hybrid_memory << std::endl;
      std::cout << "Using cuDSS Schur solver." << std::endl;
      solver_ptr = std::make_unique<cudssSchurSolver<FP, SP>>(cudss_options);
    }
  } else {
    throw std::runtime_error("Unsupported solver option");
  }

  if (!solver_ptr) {
    throw std::runtime_error("Solver initialization failed");
  }

  // Optimize
  std::cout << "Graph built with " << num_cameras << " cameras, " << num_points
            << " points, and " << r_desc.internal_count() << " observations."
            << std::endl;
  std::cout << "Optimizing!" << std::endl;

  StreamPool streams(8); // Want at least max(degree(constraints)) streams
  constexpr uint8_t optimization_level = 0;

  graphite::optimizer::LevenbergMarquardtOptions<FP, SP> options;
  options.solver = solver_ptr.get();
  options.initial_damping = program.get<double>("--lambda");
  options.iterations = program.get<size_t>("--iterations");
  options.optimization_level = optimization_level;
  options.verbose = program.get<bool>("--verbose");
  options.streams = &streams;
  options.use_identity = program.get<bool>("--identity_damping");

  start = std::chrono::steady_clock::now();
  optimizer::levenberg_marquardt<FP, SP>(&graph, &options);
  auto end = std::chrono::steady_clock::now();

  std::chrono::duration<FP> elapsed = end - start;
  std::cout << "Optimization took " << elapsed.count() << " seconds."
            << std::endl;

  auto mse = graph.chi2() / num_observations;
  std::cout << "MSE: " << mse << std::endl;
  std::cout << "Half MSE: " << mse / 2 << std::endl;

  solver_ptr.reset(); // need this gone before everything else is cleaned up
}

int main(int argc, char *argv[]) {

  argparse::ArgumentParser program("bal");

  program.add_argument("file").help("Path to the BAL problem file");

  program.add_argument("--lambda")
      .help("Initial damping factor for Levenberg-Marquardt")
      .default_value(1.0e-4)
      .scan<'g', double>();

  program.add_argument("--iterations")
      .help("Number of LM iterations")
      .default_value(static_cast<size_t>(50))
      .scan<'u', size_t>();

  program.add_argument("--verbose").help("Enable verbose output").flag();

  program.add_argument("--pcg_iterations")
      .help("Number of PCG iterations per LM step")
      .default_value(static_cast<size_t>(10))
      .scan<'u', size_t>();

  program.add_argument("--pcg_tolerance")
      .help("Tolerance for PCG solver")
      .default_value(1.0)
      .scan<'g', double>();

  program.add_argument("--rejection_ratio")
      .help("Rejection ratio for PCG iteration")
      .default_value(5.0)
      .scan<'g', double>();

  program.add_argument("--precision")
      .help("Precision for graph and solver")
      .default_value(std::string("FP64-FP64"))
      .choices("FP64-FP64", "FP64-FP32", "FP64-BF16", "FP32-FP32", "FP32-BF16");

  program.add_argument("--solver")
      .help("Linear solver type")
      .default_value(std::string("pcg"))
      .choices("pcg", "cudss", "eigen", "eigen-schur", "cudss-schur");
  program.add_argument("--identity_damping")
      .help("Use identity damping instead of Hessian diagonal")
      .flag();
  program.add_argument("--hybrid_memory")
      .help("Sets the memory limit in MB for cuDSS hybrid memory mode; "
            "values > 0 disable hybrid execute mode")
      .default_value(static_cast<int64_t>(0))
      .scan<'i', int64_t>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  try {
    const auto precision = program.get<std::string>("--precision");
    if (precision == "FP64-FP64") {
      bundle_adjustment<double, double>(program);
    } else if (precision == "FP64-FP32") {
      bundle_adjustment<double, float>(program);
    } else if (precision == "FP64-BF16") {
      bundle_adjustment<double, __nv_bfloat16>(program);
    } else if (precision == "FP32-FP32") {
      bundle_adjustment<float, float>(program);
    } else if (precision == "FP32-BF16") {
      bundle_adjustment<float, __nv_bfloat16>(program);
    } else {
      throw std::runtime_error("Unsupported precision option");
    }

  } catch (const std::exception &e) {
    std::cerr << "Error during bundle adjustment: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
