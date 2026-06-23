/// @file cudss.hpp
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cudss.h>
#include <graphite/hessian.hpp>
#include <graphite/solver/solver.hpp>
#include <type_traits>

namespace graphite {

class cudssSolverOptions {
public:
  /// Default is true, which enables cudss hybrid execution modewhere
  bool use_hybrid_execution;

  /// Setting this to a value > 0 DISABLES hybrid execution mode and instead
  /// enables hybrid memory mode. Use this when your problem size is too large
  /// to fit in GPU memory.
  int64_t hybrid_memory;

  /// The matrix type to use for cudss. Default is CUDSS_MTYPE_SYMMETRIC, which
  /// is appropriate for Hessian matrices. You may also try CUDSS_MTYPE_SPD.
  cudssMatrixType_t matrix_type;

  cudssSolverOptions() {
    use_hybrid_execution = true;
    hybrid_memory = -1;
    matrix_type = CUDSS_MTYPE_SYMMETRIC;
  }
};

template <typename T> cudssDataType_t get_cuda_data_type();

template <> inline cudssDataType_t get_cuda_data_type<double>() {
  return CUDSS_R_64F;
}

template <> inline cudssDataType_t get_cuda_data_type<float>() {
  return CUDSS_R_32F;
}

template <typename Index> inline cudssDataType_t get_cuda_index_type() {
  static_assert(std::is_same<Index, int32_t>::value ||
                    std::is_same<Index, int64_t>::value,
                "cudssSolver index type must be int32_t or int64_t");
  if (std::is_same<Index, int32_t>::value) {
    return CUDSS_R_32I;
  }
  return CUDSS_R_64I;
}

template <typename T, typename S, typename Index = int32_t>
class cudssSolver : public Solver<T, S> {
private:
  static_assert(std::is_same<Index, int32_t>::value ||
                    std::is_same<Index, int64_t>::value,
                "cudssSolver index type must be int32_t or int64_t");

  void fill_matrix_structure() {
    const auto dim = d_matrix.d_pointers.size() - 1;
    const auto nnz = d_matrix.d_values.size();
    const cudssMatrixViewType_t view_type = CUDSS_MVIEW_LOWER;
    const cudssIndexBase_t index_base = CUDSS_BASE_ZERO;

    if (m_A != NULL) {
      cudssMatrixDestroy(m_A);
      m_A = NULL;
    }

    cudssMatrixCreateCsr(
        &m_A, dim, dim, nnz, d_matrix.d_pointers.data().get(), nullptr,
        d_matrix.d_indices.data().get(), d_matrix.d_values.data().get(),
        get_cuda_index_type<Index>(), get_cuda_index_type<Index>(),
        get_cuda_data_type<S>(), matrix_type, view_type, index_base);
  }

  void fill_matrix_values() {
    cudssMatrixSetValues(m_A, d_matrix.d_values.data().get());
  }

  Hessian<T, S> H;
  CSCMatrix<S, Index> d_matrix;
  cudssMatrixType_t matrix_type;

  bool factorization_failed;

  cudaStream_t stream;
  cudssHandle_t handle;

  cudssConfig_t solver_config;
  cudssData_t solver_data;

  cudssMatrix_t m_x, m_b, m_A;

  thrust::device_vector<T> solver_x;
  int64_t configured_hybrid_memory_limit;

public:
  explicit cudssSolver(
      const cudssSolverOptions &options = cudssSolverOptions()) {
    stream = NULL;
    m_x = NULL;
    m_b = NULL;
    m_A = NULL;
    factorization_failed = false;
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
    cudssConfigSet(solver_config, CUDSS_CONFIG_HYBRID_MEMORY_MODE,
                   &enable_hybrid_memory_mode,
                   sizeof(enable_hybrid_memory_mode));
    if (options.hybrid_memory > 0) {
      int64_t mem_limit = options.hybrid_memory;
      cudssConfigSet(solver_config, CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT,
                     &mem_limit, sizeof(mem_limit));
    }
    auto reordering_alg = CUDSS_REORDERING_ALG_DEFAULT;
    cudssConfigSet(solver_config, CUDSS_CONFIG_REORDERING_ALG, &reordering_alg,
                   sizeof(reordering_alg));
    cudssDataCreate(handle, &solver_data);
  }

  ~cudssSolver() {
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
    H.build_csc_structure(graph, d_matrix);
    fill_matrix_structure();

    // Create matrices for b and x
    if (m_b != NULL) {
      cudssMatrixDestroy(m_b);
      m_b = NULL;
    }
    const auto dim = graph->get_hessian_dimension();
    auto &b = graph->get_b();
    int ldb = dim;
    int ldx = dim;
    cudssMatrixCreateDn(&m_b, dim, 1, ldb, b.data().get(),
                        get_cuda_data_type<T>(), CUDSS_LAYOUT_COL_MAJOR);

    if (m_x != NULL) {
      cudssMatrixDestroy(m_x);
      m_x = NULL;
    }
    solver_x.resize(b.size());
    cudssMatrixCreateDn(&m_x, dim, 1, ldx, solver_x.data().get(),
                        get_cuda_data_type<T>(), CUDSS_LAYOUT_COL_MAJOR);

    // Factorize
    factorization_failed = false;
    auto status = cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solver_config,
                               solver_data, m_A, m_x, m_b);
    if (status != CUDSS_STATUS_SUCCESS) {
      factorization_failed = true;
      std::cerr << "cudss Analysis failed with error code: " << status
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
        std::cerr << "Requested cuDSS hybrid memory limit is too low; raising "
                     "to minimum required "
                  << configured_hybrid_memory_limit << " bytes." << std::endl;
        status = cudssConfigSet(solver_config,
                                CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT,
                                &configured_hybrid_memory_limit,
                                sizeof(configured_hybrid_memory_limit));
        if (status != CUDSS_STATUS_SUCCESS) {
          factorization_failed = true;
          std::cerr << "Failed to update cuDSS hybrid memory limit with error "
                       "code: "
                    << status << std::endl;
        }
      }
    }
    cudaStreamSynchronize(stream);
  }

  virtual void update_values(Graph<T, S> *graph, StreamPool &streams) override {
    H.update_values(graph, streams);
    H.update_csc_values(graph, d_matrix);
    fill_matrix_values(); // for CPU matrix
  }

  virtual void set_damping_factor(Graph<T, S> *graph, T damping_factor,
                                  const bool use_identity,
                                  StreamPool &streams) override {
    H.apply_damping(graph, damping_factor, use_identity, streams);
    H.update_csc_values(graph, d_matrix);
    fill_matrix_values(); // TODO: Use a more lightweight method to just update
                          // diagonal
  }

  virtual bool solve(Graph<T, S> *graph, T *x, StreamPool &streams) override {

    if (factorization_failed) {
      return false;
    }

    auto dim = graph->get_hessian_dimension();

    thrust::fill(thrust::device, x, x + dim, static_cast<T>(0.0));
    thrust::copy(thrust::device, x, x + dim, solver_x.data());

    cudssStatus_t status;

    // set values for b and x
    cudssMatrixSetValues(m_b, graph->get_b().data().get());
    cudssMatrixSetValues(m_x, solver_x.data().get());

    status = cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solver_config,
                          solver_data, m_A, m_x, m_b);
    if (status != CUDSS_STATUS_SUCCESS) {
      std::cerr << "cudss Factorization failed with error code: " << status
                << std::endl;
      return false;
    }

    status = cudssExecute(handle, CUDSS_PHASE_SOLVE, solver_config, solver_data,
                          m_A, m_x, m_b);
    if (status != CUDSS_STATUS_SUCCESS) {
      std::cerr << "cudss Solve failed with error code: " << status
                << std::endl;
      return false;
    }
    cudaStreamSynchronize(stream);

    thrust::copy(thrust::device, solver_x.data(), solver_x.data() + dim, x);

    return true;
  }
};

} // namespace graphite