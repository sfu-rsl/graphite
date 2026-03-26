#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <graphite/factor.hpp>
#include <graphite/graph.hpp>
#include <graphite/ops/schur.hpp>
#include <graphite/schur.hpp>
#include <graphite/solver/cudss.hpp>
#include <graphite/solver/cudss_schur.hpp>
#include <graphite/solver/eigen.hpp>
#include <graphite/solver/eigen_schur.hpp>
#include <graphite/stream.hpp>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <array>

#include "../examples/bal.cuh"
#include "schur_cpu_ref.hpp"

namespace graphite {

bool has_block(
    const std::unordered_map<graphite::BlockCoordinates, size_t> &blocks,
    size_t row, size_t col) {
  return blocks.find(graphite::BlockCoordinates{row, col}) != blocks.end();
}

template <typename T, typename S>
void build_bal_two_camera_three_point_problem(
    Graph<T, S> &graph, CameraDescriptor<T, S> &camera_desc,
    PointDescriptor<T, S> &point_desc, ReprojectionError<T, S> &reproj_desc,
    managed_vector<Camera<T>> &cameras, managed_vector<Point<T>> &points) {
  camera_desc.reserve(2);
  point_desc.reserve(3);
  reproj_desc.reserve(6);

  graph.add_descriptor(&camera_desc);
  graph.add_descriptor(&point_desc);
  graph.add_descriptor(&reproj_desc);

  cameras.resize(2);
  points.resize(3);

  // Use non-degenerate intrinsics/extrinsics so all Hessian diagonals are
  // informative across solver comparison tests.
  cameras[0] << static_cast<T>(0.12), static_cast<T>(-0.08),
      static_cast<T>(0.03), static_cast<T>(0.25), static_cast<T>(-0.10),
      static_cast<T>(0.20), static_cast<T>(800.0), static_cast<T>(0.01),
      static_cast<T>(-0.001);
  cameras[1] << static_cast<T>(-0.09), static_cast<T>(0.06),
      static_cast<T>(-0.04), static_cast<T>(-0.30), static_cast<T>(0.14),
      static_cast<T>(-0.22), static_cast<T>(820.0), static_cast<T>(-0.012),
      static_cast<T>(0.0009);
  points[0] << 0.1f, 0.0f, 2.0f;
  points[1] << -0.1f, 0.05f, 2.2f;
  points[2] << 0.0f, -0.05f, 1.8f;

  camera_desc.add_vertex(0, &cameras[0]);
  camera_desc.add_vertex(1, &cameras[1]);
  point_desc.add_vertex(2, &points[0]);
  point_desc.add_vertex(3, &points[1]);
  point_desc.add_vertex(4, &points[2]);

  point_desc.set_eliminate(true);

  const Eigen::Matrix<T, 2, 1> obs(0.0f, 0.0f);
  reproj_desc.add_factor({0, 2}, obs);
  reproj_desc.add_factor({1, 2}, obs);
  reproj_desc.add_factor({0, 3}, obs);
  reproj_desc.add_factor({1, 3}, obs);
  reproj_desc.add_factor({0, 4}, obs);
  reproj_desc.add_factor({1, 4}, obs);
}

template <typename S, typename I>
Eigen::SparseMatrix<S, Eigen::ColMajor>
to_eigen_sparse(const graphite::CSCMatrix<S, I> &d_matrix) {

  Eigen::SparseMatrix<S, Eigen::ColMajor> matrix;
  const auto dim = d_matrix.d_pointers.size() - 1;
  matrix.resize(dim, dim);
  matrix.resizeNonZeros(d_matrix.d_values.size());

  auto h_ptrs = matrix.outerIndexPtr();
  auto h_indices = matrix.innerIndexPtr();
  auto h_values = matrix.valuePtr();

  thrust::copy(d_matrix.d_pointers.begin(), d_matrix.d_pointers.end(), h_ptrs);
  thrust::copy(d_matrix.d_indices.begin(), d_matrix.d_indices.end(), h_indices);
  thrust::copy(d_matrix.d_values.begin(), d_matrix.d_values.end(), h_values);
  return matrix;
}

template <int dim, typename T>
Eigen::Matrix<T, dim, dim> make_spd_block(T scale) {
  Eigen::Matrix<T, dim, dim> A;
  for (int c = 0; c < dim; ++c) {
    for (int r = 0; r < dim; ++r) {
      A(r, c) = static_cast<T>((r + 1) * (c + 2)) + scale;
    }
  }

  return A.transpose() * A + static_cast<T>(0.5 + static_cast<T>(dim)) *
                                 Eigen::Matrix<T, dim, dim>::Identity();
}

/*
template <int dim, typename T> void run_small_inverse_case(T tolerance) {
  constexpr size_t num_blocks = 2;
  constexpr size_t block_size = static_cast<size_t>(dim * dim);

  std::array<Eigen::Matrix<T, dim, dim>, num_blocks> blocks;
  blocks[0] = make_spd_block<dim, T>(static_cast<T>(0.25));
  blocks[1] = make_spd_block<dim, T>(static_cast<T>(0.75));

  thrust::host_vector<T> h_src(num_blocks * block_size);
  for (size_t i = 0; i < num_blocks; ++i) {
    Eigen::Map<Eigen::Matrix<T, dim, dim>>(h_src.data() + i * block_size) =
        blocks[i];
  }

  thrust::device_vector<T> d_src = h_src;
  thrust::device_vector<T> d_dst(num_blocks * block_size, static_cast<T>(0));

  thrust::host_vector<T *> h_src_ptrs(num_blocks);
  thrust::host_vector<T *> h_dst_ptrs(num_blocks);
  T *src_base = d_src.data().get();
  T *dst_base = d_dst.data().get();
  for (size_t i = 0; i < num_blocks; ++i) {
    h_src_ptrs[i] = src_base + i * block_size;
    h_dst_ptrs[i] = dst_base + i * block_size;
  }

  thrust::device_vector<T *> d_src_ptrs = h_src_ptrs;
  thrust::device_vector<T *> d_dst_ptrs = h_dst_ptrs;

  const bool launched = ops::launch_small_block_inverse_batched<T>(
      static_cast<size_t>(dim), d_src_ptrs.data().get(),
      d_dst_ptrs.data().get(), num_blocks, 0);
  ASSERT_TRUE(launched);
  cudaDeviceSynchronize();

  thrust::host_vector<T> h_dst = d_dst;

  for (size_t i = 0; i < num_blocks; ++i) {
    const Eigen::Matrix<T, dim, dim> expected = blocks[i].inverse();
    Eigen::Map<const Eigen::Matrix<T, dim, dim>> got(h_dst.data() +
                                                     i * block_size);

    for (int c = 0; c < dim; ++c) {
      for (int r = 0; r < dim; ++r) {
        EXPECT_NEAR(got(r, c), expected(r, c), tolerance)
            << "Mismatch in dim=" << dim << " block=" << i << " r=" << r
            << " c=" << c;
      }
    }
  }
}

TEST(SchurTests, SmallEigenBlockInverseKernel) {
  using T = double;
  run_small_inverse_case<1, T>(1e-12);
  run_small_inverse_case<2, T>(1e-12);
  run_small_inverse_case<3, T>(1e-12);
  run_small_inverse_case<4, T>(1e-11);
}
*/

TEST(SchurTests, BALTwoCamerasThreePoints) {
  using T = double;
  using S = double;

  Graph<T, S> graph;
  CameraDescriptor<T, S> camera_desc;
  PointDescriptor<T, S> point_desc;
  ReprojectionError<T, S> reproj_desc(&camera_desc, &point_desc);
  managed_vector<Camera<T>> cameras;
  managed_vector<Point<T>> points;

  build_bal_two_camera_three_point_problem(graph, camera_desc, point_desc,
                                           reproj_desc, cameras, points);

  ASSERT_TRUE(graph.initialize_optimization(0));

  StreamPool streams(2);
  Hessian<T, S> H;
  SchurComplement<T, S> schur(H);

  using I = Eigen::SparseMatrix<S, Eigen::ColMajor>::StorageIndex;

  CSCMatrix<S, I> d_H;
  CSCMatrix<S, I> d_Schur;

  graph.build_structure();

  H.build_structure(&graph, streams);
  schur.build_structure(&graph, streams);

  graph.linearize(streams);
  EXPECT_NE(graph.chi2(), 0.0);

  H.update_values(&graph, streams);
  schur.update_values(&graph, streams);

  // Copy to csc
  H.build_csc_structure(&graph, d_H);
  schur.build_csc_structure(&graph, d_Schur);

  H.update_csc_values(&graph, d_H);
  schur.update_csc_values(&graph, d_Schur);

  // Copy to Eigen sparse CSC
  const auto hessian_csc = to_eigen_sparse(d_H);
  const auto schur_csc = to_eigen_sparse(d_Schur);

  // matrices must be non-zero
  // std::cout << "Hessian csc: \n" << hessian_csc << "\n\n";
  // std::cout << "Schur csc: \n" << schur_csc << "\n\n";
  // EXPECT_NE(hessian_csc.nonZeros(), 0.0);
  // EXPECT_NE(schur_csc.nonZeros(), 0.0);

  const auto hessian_sym = Eigen::SparseMatrix<S, Eigen::ColMajor>(
      hessian_csc.selfadjointView<Eigen::Upper>());

  // Compute sparse Schur complement on CPU with Eigen
  // S = Hpp - Hpl * Hll^(-1) * Hpl^T
  const size_t pose_start = 0;
  const size_t pose_dim = 2 * 9;
  const size_t landmark_start = 2 * 9;
  const size_t landmark_dim = 3 * 3;

  const auto schur_ref = test_helpers::build_schur_reference(
      hessian_sym, pose_start, pose_dim, landmark_start, landmark_dim);

  // Compare the GPU Schur complement with the CPU Eigen result
  EXPECT_TRUE(schur_csc.isApprox(schur_ref.schur_upper, 1e-12));

  // print matrices for debugging
  // std::cout << "GPU Schur complement (upper triangular):\n";
  // std::cout << schur_csc << "\n\n";
  // std::cout << "CPU Eigen Schur complement (upper triangular):\n";
  // std::cout << schur_eigen_ut << "\n";

  // Now compute b_Schur
  thrust::host_vector<T> b = graph.get_b();

  // Compute on GPU
  thrust::host_vector<T> b_Schur = schur.get_b_Schur();

  // Compute on CPU
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> b_map(b.data(), b.size());
  Eigen::Matrix<T, Eigen::Dynamic, 1> b_Schur_cpu =
      test_helpers::compute_b_schur_cpu(b_map, schur_ref.hpl, schur_ref.hll_inv,
                                        pose_dim, landmark_start, landmark_dim);

  // Check
  thrust::host_vector<T> b_Schur_host = b_Schur;
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> b_Schur_host_map(
      b_Schur_host.data(), b_Schur_host.size());

  for (Eigen::Index i = 0; i < b_Schur_host_map.size(); ++i) {
    EXPECT_NEAR(b_Schur_host_map[i], b_Schur_cpu[i], 1e-12)
        << "b_Schur mismatch at index " << i;
  }

  // print for debugging
  // std::cout << "GPU b_Schur:\n" << b_Schur_host_map << "\n\n";
  // std::cout << "CPU b_Schur:\n" << b_Schur_cpu << "\n";

  // Now check backsubstitution: v2 = Hll^(-1) * Hpl^T * v1
  Eigen::Matrix<T, Eigen::Dynamic, 1> dx_p(pose_dim);
  for (Eigen::Index i = 0; i < dx_p.size(); ++i) {
    dx_p[i] = static_cast<T>(0.01) * static_cast<T>(i + 1);
  }

  Eigen::Matrix<T, Eigen::Dynamic, 1> dx_l_cpu =
      test_helpers::compute_landmark_update_cpu(b_map, schur_ref.hplt,
                                                schur_ref.hll_inv, dx_p,
                                                landmark_start, landmark_dim);

  thrust::device_vector<T> d_dx_p(pose_dim);
  thrust::copy(dx_p.data(), dx_p.data() + pose_dim, d_dx_p.begin());

  thrust::device_vector<T> d_dx_l(landmark_dim);
  schur.compute_landmark_update(&graph, streams, d_dx_l.data().get(),
                                d_dx_p.data().get());

  thrust::host_vector<T> dx_l_host = d_dx_l;
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> dx_l_gpu_map(
      dx_l_host.data(), dx_l_host.size());

  for (Eigen::Index i = 0; i < dx_l_gpu_map.size(); ++i) {
    EXPECT_NEAR(dx_l_gpu_map[i], dx_l_cpu[i], 1e-12)
        << "dx_l mismatch at index " << i;
  }
}

TEST(SchurTests, EigenSchur) {
  using T = double;
  using S = double;

  Graph<T, S> graph;
  CameraDescriptor<T, S> camera_desc;
  PointDescriptor<T, S> point_desc;
  ReprojectionError<T, S> reproj_desc(&camera_desc, &point_desc);
  managed_vector<Camera<T>> cameras;
  managed_vector<Point<T>> points;

  build_bal_two_camera_three_point_problem(graph, camera_desc, point_desc,
                                           reproj_desc, cameras, points);

  ASSERT_TRUE(graph.initialize_optimization(0));
  ASSERT_TRUE(graph.build_structure());

  StreamPool streams(2);
  graph.linearize(streams);

  EigenLDLTSolver<T, S> full_solver;
  EigenSchurLDLTSolver<T, S> schur_solver;

  full_solver.update_structure(&graph, streams);
  schur_solver.update_structure(&graph, streams);

  full_solver.update_values(&graph, streams);
  schur_solver.update_values(&graph, streams);

  const T damping = static_cast<T>(1e-4);
  full_solver.set_damping_factor(&graph, damping, false, streams);
  schur_solver.set_damping_factor(&graph, damping, false, streams);

  thrust::device_vector<T> d_dx_full(graph.get_hessian_dimension());
  thrust::device_vector<T> d_dx_schur(graph.get_hessian_dimension());

  ASSERT_TRUE(full_solver.solve(&graph, d_dx_full.data().get(), streams));
  ASSERT_TRUE(schur_solver.solve(&graph, d_dx_schur.data().get(), streams));

  thrust::host_vector<T> h_dx_full = d_dx_full;
  thrust::host_vector<T> h_dx_schur = d_dx_schur;

  ASSERT_EQ(h_dx_full.size(), h_dx_schur.size());
  for (size_t i = 0; i < h_dx_full.size(); ++i) {
    EXPECT_NEAR(h_dx_schur[i], h_dx_full[i], 1e-8)
        << "delta_x mismatch at index " << i;
  }
}

TEST(SchurTests, CudssSchur) {
  using T = double;
  using S = double;

  Graph<T, S> graph;
  CameraDescriptor<T, S> camera_desc;
  PointDescriptor<T, S> point_desc;
  ReprojectionError<T, S> reproj_desc(&camera_desc, &point_desc);
  managed_vector<Camera<T>> cameras;
  managed_vector<Point<T>> points;

  build_bal_two_camera_three_point_problem(graph, camera_desc, point_desc,
                                           reproj_desc, cameras, points);

  ASSERT_TRUE(graph.initialize_optimization(0));
  ASSERT_TRUE(graph.build_structure());

  StreamPool streams(2);
  graph.linearize(streams);

  cudssSolver<T, S> full_solver;
  cudssSchurSolver<T, S> schur_solver;

  full_solver.update_structure(&graph, streams);
  schur_solver.update_structure(&graph, streams);

  full_solver.update_values(&graph, streams);
  schur_solver.update_values(&graph, streams);

  const T damping = static_cast<T>(1e-4);
  full_solver.set_damping_factor(&graph, damping, false, streams);
  schur_solver.set_damping_factor(&graph, damping, false, streams);

  thrust::device_vector<T> d_dx_full(graph.get_hessian_dimension());
  thrust::device_vector<T> d_dx_schur(graph.get_hessian_dimension());

  ASSERT_TRUE(full_solver.solve(&graph, d_dx_full.data().get(), streams));
  ASSERT_TRUE(schur_solver.solve(&graph, d_dx_schur.data().get(), streams));

  thrust::host_vector<T> h_dx_full = d_dx_full;
  thrust::host_vector<T> h_dx_schur = d_dx_schur;

  ASSERT_EQ(h_dx_full.size(), h_dx_schur.size());
  for (size_t i = 0; i < h_dx_full.size(); ++i) {
    EXPECT_NEAR(h_dx_schur[i], h_dx_full[i], 1e-8)
        << "delta_x mismatch at index " << i;
  }
}

} // namespace graphite
