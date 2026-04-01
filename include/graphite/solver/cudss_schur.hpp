/// @file cudss_schur.hpp
#pragma once

#include <cuda_runtime.h>
#include <cudss.h>
#include <graphite/hessian.hpp>
#include <graphite/schur.hpp>
#include <graphite/solver/cudss.hpp>
#include <graphite/solver/solver.hpp>

namespace graphite {

template <typename T, typename S, typename Index = int32_t>
class cudssSchurSolver : public Solver<T, S> {
private:
  static_assert(std::is_same<Index, int32_t>::value ||
                    std::is_same<Index, int64_t>::value,
                "cudssSchurSolver index type must be int32_t or int64_t");

  Hessian<T, S> H;
  SchurComplement<T, S> schur;
  CSCMatrix<S, Index> d_matrix;
  cudssMatrixType_t matrix_type;

  bool factorization_failed;

  cudaStream_t stream;
  cudssHandle_t handle;

  cudssConfig_t solver_config;
  cudssData_t solver_data;

  cudssMatrix_t m_x, m_b, m_A;

  thrust::device_vector<T> solver_x;
  size_t schur_dim;
  int64_t configured_hybrid_memory_limit;

  void fill_matrix_structure() {
    const auto dim = d_matrix.d_pointers.size() - 1;
    const auto nnz = d_matrix.d_values.size();
    const cudssMatrixViewType_t view_type = CUDSS_MVIEW_LOWER;
    const cudssIndexBase_t index_base = CUDSS_BASE_ZERO;

    if (m_A != NULL) {
      cudssMatrixDestroy(m_A);
      m_A = NULL;
    }

    cudssMatrixCreateCsr(&m_A, dim, dim, nnz, d_matrix.d_pointers.data().get(),
                         nullptr, d_matrix.d_indices.data().get(),
                         d_matrix.d_values.data().get(),
                         get_cuda_index_type<Index>(), get_cuda_data_type<S>(),
                         matrix_type, view_type, index_base);
  }

  void fill_matrix_values() {
    cudssMatrixSetValues(m_A, d_matrix.d_values.data().get());
  }

public:
  explicit cudssSchurSolver(
      const cudssSolverOptions &options = cudssSolverOptions())
      : schur(H), factorization_failed(false), schur_dim(0) {
    stream = NULL;
    m_x = NULL;
    m_b = NULL;
    m_A = NULL;
    matrix_type = options.matrix_type;
    configured_hybrid_memory_limit = options.hybrid_memory;

    cudaStreamCreate(&stream);
    cudssCreate(&handle);
    cudssSetStream(handle, stream);

    cudssConfigCreate(&solver_config);
    int enable_hybrid_exec_mode =
        (options.use_hybrid_execution && options.hybrid_memory <= 0) ? 1 : 0;
    cudssConfigSet(solver_config, CUDSS_CONFIG_HYBRID_EXECUTE_MODE,
                   &enable_hybrid_exec_mode, sizeof(enable_hybrid_exec_mode));
    int enable_hybrid_memory_mode = (options.hybrid_memory > 0) ? 1 : 0;
    cudssConfigSet(solver_config, CUDSS_CONFIG_HYBRID_MODE,
                   &enable_hybrid_memory_mode,
                   sizeof(enable_hybrid_memory_mode));
    if (options.hybrid_memory > 0) {
      int64_t mem_limit = options.hybrid_memory;
      cudssConfigSet(solver_config, CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT,
                     &mem_limit, sizeof(mem_limit));
    }
    cudssAlgType_t reordering_alg = CUDSS_ALG_DEFAULT;
    cudssConfigSet(solver_config, CUDSS_CONFIG_REORDERING_ALG, &reordering_alg,
                   sizeof(reordering_alg));
    cudssDataCreate(handle, &solver_data);
  }

  ~cudssSchurSolver() {
    if (m_A != NULL) {
      cudssMatrixDestroy(m_A);
      m_A = NULL;
    }
    if (m_b != NULL) {
      cudssMatrixDestroy(m_b);
      m_b = NULL;
    }
    if (m_x != NULL) {
      cudssMatrixDestroy(m_x);
      m_x = NULL;
    }

    cudssDataDestroy(handle, solver_data);
    cudssConfigDestroy(solver_config);
    cudssDestroy(handle);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }

  virtual void update_structure(Graph<T, S> *graph,
                                StreamPool &streams) override {
    H.build_structure(graph, streams);
    schur.build_structure(graph, streams);
    schur.build_csc_structure(graph, d_matrix);
    fill_matrix_structure();

    const auto &offsets = graph->get_offset_vector();
    schur_dim = offsets[schur.lowest_eliminated_block_col];

    if (m_b != NULL) {
      cudssMatrixDestroy(m_b);
      m_b = NULL;
    }
    int ldb = static_cast<int>(schur_dim);
    cudssMatrixCreateDn(&m_b, schur_dim, 1, ldb,
                        schur.get_b_Schur().data().get(),
                        get_cuda_data_type<T>(), CUDSS_LAYOUT_COL_MAJOR);

    if (m_x != NULL) {
      cudssMatrixDestroy(m_x);
      m_x = NULL;
    }
    solver_x.resize(schur_dim);
    int ldx = static_cast<int>(schur_dim);
    cudssMatrixCreateDn(&m_x, schur_dim, 1, ldx, solver_x.data().get(),
                        get_cuda_data_type<T>(), CUDSS_LAYOUT_COL_MAJOR);

    factorization_failed = false;
    auto status = cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solver_config,
                               solver_data, m_A, m_x, m_b);
    if (status != CUDSS_STATUS_SUCCESS) {
      factorization_failed = true;
      std::cerr << "cudss Schur analysis failed with error code: " << status
                << std::endl;
    } else if (configured_hybrid_memory_limit > 0) {
      int64_t min_hybrid_memory = 0;
      size_t size_written = 0;
      status = cudssDataGet(
          handle, solver_data, CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN,
          &min_hybrid_memory, sizeof(min_hybrid_memory), &size_written);
      if (status == CUDSS_STATUS_SUCCESS &&
          configured_hybrid_memory_limit < min_hybrid_memory) {
        configured_hybrid_memory_limit = min_hybrid_memory;
        std::cerr << "Requested cuDSS Schur hybrid memory limit is too low; "
                     "raising to minimum required "
                  << configured_hybrid_memory_limit << " bytes." << std::endl;
        status = cudssConfigSet(solver_config,
                                CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT,
                                &configured_hybrid_memory_limit,
                                sizeof(configured_hybrid_memory_limit));
        if (status != CUDSS_STATUS_SUCCESS) {
          factorization_failed = true;
          std::cerr
              << "Failed to update cuDSS Schur hybrid memory limit with error "
                 "code: "
              << status << std::endl;
        }
      }
    }
    cudaStreamSynchronize(stream);
  }

  virtual void update_values(Graph<T, S> *graph, StreamPool &streams) override {
    H.update_values(graph, streams);
  }

  virtual void set_damping_factor(Graph<T, S> *graph, T damping_factor,
                                  const bool use_identity,
                                  StreamPool &streams) override {
    H.apply_damping(graph, damping_factor, use_identity, streams);
  }

  virtual bool solve(Graph<T, S> *graph, T *x, StreamPool &streams) override {
    if (factorization_failed) {
      return false;
    }

    // Update values before solving
    schur.update_values(graph, streams);
    schur.update_csc_values(graph, d_matrix);
    fill_matrix_values();

    const auto dim = graph->get_hessian_dimension();

    thrust::fill(thrust::device, x, x + dim, static_cast<T>(0.0));
    thrust::fill(thrust::device, solver_x.begin(), solver_x.end(),
                 static_cast<T>(0.0));

    cudssStatus_t status;

    cudssMatrixSetValues(m_b, schur.get_b_Schur().data().get());
    cudssMatrixSetValues(m_x, solver_x.data().get());

    status = cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solver_config,
                          solver_data, m_A, m_x, m_b);
    if (status != CUDSS_STATUS_SUCCESS) {
      std::cerr << "cudss Schur factorization failed with error code: "
                << status << std::endl;
      return false;
    }

    status = cudssExecute(handle, CUDSS_PHASE_SOLVE, solver_config, solver_data,
                          m_A, m_x, m_b);
    if (status != CUDSS_STATUS_SUCCESS) {
      std::cerr << "cudss Schur solve failed with error code: " << status
                << std::endl;
      return false;
    }
    cudaStreamSynchronize(stream);

    thrust::copy(thrust::device, solver_x.data(), solver_x.data() + schur_dim,
                 x);

    schur.compute_landmark_update(graph, streams, x + schur_dim, x);

    return true;
  }
};

} // namespace graphite
