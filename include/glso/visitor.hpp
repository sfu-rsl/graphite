#pragma once
#include <glso/common.hpp>
#include <glso/differentiation.hpp>
#include <glso/types.hpp>
#include <glso/vertex.hpp>
#include <glso/stream.hpp>

namespace glso {

__device__ size_t get_thread_id() {
  return static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
         static_cast<size_t>(threadIdx.x);
}

template <typename VPtr, class VD, typename T, size_t D>
__device__ void device_copy(const VPtr v, T *dst) {
  // const std::array<T, D> src = v->parameters();
  const std::array<T, D> src = VD::Traits::parameters(*v);
#pragma unroll
  for (size_t i = 0; i < D; i++) {
    dst[i] = src[i];
  }
}

template <typename VPtr, class VD, typename T, typename P, size_t D>
__device__ void real_to_dual(const VPtr v, Dual<T, P> *dst) {
  // const std::array<T, D> src = v->parameters();
  const std::array<T, D> src = VD::Traits::parameters(*v);
#pragma unroll
  for (size_t i = 0; i < D; i++) {
    dst[i] = Dual<T, P>(src[i]);
  }
}

__device__ bool is_fixed(const uint32_t *fixed, const size_t vertex_id) {
  const uint32_t mask = 1 << (vertex_id % 32);
  return (fixed[vertex_id / 32] & mask);
}

template <typename T, typename P, size_t E>
__device__ T compute_chi2(const T *residuals, const P *pmat,
                          const size_t factor_id) {
  T r2[E] = {0};

#pragma unroll
  for (int i = 0; i < E; i++) {
#pragma unroll
    for (int j = 0; j < E; j++) {
      r2[i] += static_cast<T>(pmat[factor_id * E * E + i * E + j]) *
               residuals[factor_id * E + j];
    }
  }

  T value = 0;
#pragma unroll
  for (int i = 0; i < E; i++) {
    value += r2[i] * residuals[factor_id * E + i];
  }

  return value;
}

template <typename T, typename G, size_t I, size_t N, typename M, size_t E,
          typename F, typename VT, std::size_t... Is>
__device__ void compute_Jblock(
    G *jacobian, const size_t factor_id, const size_t vertex_id, const M *obs,
    const typename F::ConstraintDataType *constraint_data, const size_t *ids,
    const size_t *hessian_ids, VT args, std::index_sequence<Is...>) {

  constexpr auto vertex_sizes = F::get_vertex_sizes();

  const M *local_obs = obs + factor_id;
  const typename F::ConstraintDataType *local_data =
      constraint_data + factor_id;

  auto vargs = cuda::std::make_tuple(
      (*(std::get<Is>(args) + ids[factor_id * N + Is]))...);

  F::Traits::template jacobian<T, I>(cuda::std::get<Is>(vargs)..., local_obs,
                                     jacobian, local_data);
}

template <typename T, typename S, typename Descriptor, typename V>
__global__ void
apply_update_kernel(V **vertices, const T *delta_x, const T *jacobian_scales,
                    const size_t *hessian_ids, const uint32_t *fixed,
                    const size_t num_threads) {
  const size_t vertex_id = get_thread_id();

  if (vertex_id >= num_threads || is_fixed(fixed, vertex_id)) {
    return;
  }

  const T *delta = delta_x + hessian_ids[vertex_id];
  const T *scales = jacobian_scales + hessian_ids[vertex_id];

  std::array<T, Descriptor::dim> scaled_delta;
#pragma unroll
  for (size_t i = 0; i < Descriptor::dim; i++) {
    scaled_delta[i] = delta[i] * scales[i];
  }

  // vertices[vertex_id]->update(scaled_delta.data());
  Descriptor::Traits::update(*vertices[vertex_id], scaled_delta.data());
}

template <typename S, int D>
__global__ void augment_hessian_diagonal_kernel(S *diagonal_blocks, const S mu,
                                                const uint32_t *fixed,
                                                const size_t num_threads) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr auto block_size = D * D;

  const auto vertex_id = idx;
  if (is_fixed(fixed, vertex_id)) {
    return;
  }

  S *block = diagonal_blocks + vertex_id * block_size;
  for (size_t i = 0; i < D; i++) {
    // block[i * D + i] +=
    //     mu * static_cast<S>(
    //              std::clamp(static_cast<double>(block[i * D +
    //              i]), 1.0e-6, 1.0e32));

    const double diag = static_cast<double>(block[i * D + i]);
    const double new_diag =
        diag + static_cast<double>(mu) * std::clamp(diag, 1.0e-6, 1.0e32);
    block[i * D + i] = static_cast<S>(new_diag);
  }
}

template <typename T, typename S, int D>
__global__ void apply_block_jacobi_kernel(T *z, const T *r, S *block_diagonal,
                                          const size_t *hessian_ids,
                                          const uint32_t *fixed,
                                          const size_t num_threads) {
  const size_t idx = get_thread_id();
  const auto local_vertex_id = idx / D;

  if (idx >= num_threads || is_fixed(fixed, local_vertex_id)) {
    return;
  }

  constexpr auto block_size = D * D;

  S *block = block_diagonal + local_vertex_id * block_size;
  const auto hessian_offset = hessian_ids[local_vertex_id];
  const auto offset = idx % D;
  const auto row = offset;

  T value = 0;
#pragma unroll
  for (size_t i = 0; i < D; i++) {
    value += (T)block[row + i * D] * r[hessian_offset + i];
  }
  z[hessian_offset + row] = value;
}

template <typename T, typename S, size_t I, size_t N, typename M, size_t E,
          typename F, typename VT, std::size_t... Is>
__global__ void compute_error_kernel_autodiff(
    const M *obs, T *error,
    const typename F::ConstraintDataType *constraint_data, 
    const size_t* active_ids, const size_t *ids,
    const size_t *hessian_ids, const size_t num_threads, VT args, S *jacs,
    const uint32_t *fixed, std::index_sequence<Is...>) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr auto vertex_sizes = F::get_vertex_sizes();
  const auto factor_id = idx / vertex_sizes[I];
  const auto vertex_id = ids[factor_id * N + I];

  // printf("CEAD: Thread %d, Vertex %d, Factor %d\n", idx, vertex_id,
  // factor_id);
  using G = std::conditional_t<is_low_precision<S>::value, T, S>;
  const M *local_obs = obs + factor_id;
  Dual<T, G> local_error[E];
  const typename F::ConstraintDataType *local_data =
      constraint_data + factor_id;

  auto v = cuda::std::make_tuple(std::array<Dual<T, G>, vertex_sizes[Is]>{}...);

  auto vargs =
      std::make_tuple((*(std::get<Is>(args) + ids[factor_id * N + Is]))...);

  auto copy_vertices = [&v, &vertex_sizes, &vargs](auto &&...ptrs) {
    ((real_to_dual<
         decltype(std::get<Is>(vargs)),
         std::tuple_element<Is, typename F::Traits::VertexDescriptors>::type, T,
         G, vertex_sizes[Is]>(std::get<Is>(vargs),
                              cuda::std::get<Is>(v).data())),
     ...);
  };

  std::apply(copy_vertices, vargs);

  cuda::std::get<I>(v)[idx % vertex_sizes[I]].dual = static_cast<G>(1);

  F::Traits::error(cuda::std::get<Is>(v).data()..., local_obs, local_error,
                   vargs, local_data);

  constexpr auto j_size = vertex_sizes[I] * E;
  // constexpr auto col_offset = I*E;
  const auto col_offset = (idx % vertex_sizes[I]) * E;
  // Store column-major Jacobian blocks.
  // Write one scalar column (length E) of the Jacobian matrix.
  // TODO: make sure this only writes to each location once
  // The Jacobian is stored as E x vertex_size in col major

  // Only run once per factor - this check won't work for multiple kernel
  // launches
  // TODO: make sure this only writes to each location once for the error
  if (idx % vertex_sizes[I] == 0) {
#pragma unroll
    for (size_t i = 0; i < E; ++i) {
      error[factor_id * E + i] = local_error[i].real;
    }
  }

  // This should write one Jacobian column per dimension per vertex for each
  // factor We only need a Jacobian if the vertex is not fixed
  if (is_fixed(fixed, vertex_id)) {
    return;
  }

  if constexpr (std::is_same<S, __half>::value) {
// Need to clamp range
#pragma unroll
    for (size_t i = 0; i < E; ++i) {
      jacs[j_size * factor_id + col_offset + i] =
          static_cast<S>(std::clamp(local_error[i].dual, -65504.0f, 65504.0f));
    }
  } else {
#pragma unroll
    for (size_t i = 0; i < E; ++i) {
      jacs[j_size * factor_id + col_offset + i] = local_error[i].dual;
    }
  }
}
// TODO: Make this more efficient and see if code can be shared with the
// autodiff kernel
template <typename T, size_t N, typename M, size_t E, typename F, typename VT,
          std::size_t... Is>
__global__ void
compute_error_kernel(const M *obs, T *error,
                     const typename F::ConstraintDataType *constraint_data,
                     const size_t* active_ids,
                     const size_t *ids, const size_t num_threads, VT args,
                     std::index_sequence<Is...>) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr auto vertex_sizes = F::get_vertex_sizes();
  const auto factor_id = active_ids[idx];

  const M *local_obs = obs + factor_id;
  T *local_error = error + factor_id * E;
  const typename F::ConstraintDataType *local_data =
      constraint_data + factor_id;

  auto v = cuda::std::make_tuple(std::array<T, vertex_sizes[Is]>{}...);

  auto vargs =
      std::make_tuple((*(std::get<Is>(args) + ids[factor_id * N + Is]))...);

  auto copy_vertices = [&v, &vertex_sizes, &vargs](auto &&...ptrs) {
    ((device_copy<
         decltype(std::get<Is>(vargs)),
         std::tuple_element<Is, typename F::Traits::VertexDescriptors>::type, T,
         vertex_sizes[Is]>(std::get<Is>(vargs), cuda::std::get<Is>(v).data())),
     ...);
  };

  std::apply(copy_vertices, vargs);

  F::Traits::error(cuda::std::get<Is>(v).data()..., local_obs, local_error,
                   vargs, local_data);

#pragma unroll
  for (size_t i = 0; i < E; ++i) {
    error[factor_id * E + i] = local_error[i];
  }
}

template <typename T, typename S, size_t I, size_t N, typename M, size_t E,
          typename F, typename VT, std::size_t... Is>
__global__ void
compute_jacobian_kernel(const M *obs, T *error, S *jacs,
                        const typename F::ConstraintDataType *constraint_data,
                        const size_t* active_ids, const size_t *ids, const size_t num_threads, const VT args,
                        const uint32_t *fixed, std::index_sequence<Is...>) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr auto vertex_sizes = F::get_vertex_sizes();
  const auto factor_id = active_ids[idx];
  const size_t vertex_id = ids[factor_id * N + I];
  if (is_fixed(fixed, vertex_id)) {
    return;
  }

  const M *local_obs = obs + factor_id;
  // T *local_error = error + factor_id * E;
  const typename F::ConstraintDataType *local_data =
      constraint_data + factor_id;

  auto vargs = cuda::std::make_tuple(
      (*(std::get<Is>(args) + ids[factor_id * N + Is]))...);

  constexpr size_t jacobian_size = E * vertex_sizes[I];

  S *j = jacs + factor_id * jacobian_size;
  if constexpr (is_low_precision<S>::value) {
    T jacobian[jacobian_size];
    F::Traits::template jacobian<T, I>(cuda::std::get<Is>(vargs)..., local_obs,
                                       jacobian, local_data);

#pragma unroll
    for (size_t i = 0; i < jacobian_size; ++i) {
      j[i] = jacobian[i];
    }
  } else {
    F::Traits::template jacobian<T, I>(cuda::std::get<Is>(vargs)..., local_obs,
                                       j, local_data);
  }
}

// The output will be part of b with length of the vertex (where b = -J^T * r)
// Note the negative sign - different papers use different conventions
// TODO: Replace with generic J^T x r kernel?
// Note: The error vector is local to the factor
// Include precision matrix
template <typename T, typename S, size_t I, size_t N, size_t E, typename F,
          std::size_t... Is>
__global__ void
compute_b_kernel(T *b, const T *error, 
  const size_t* active_ids, const size_t *ids,
                 const size_t *hessian_ids, const size_t num_threads, 
                 const S *jacs,
                 const uint32_t *fixed, const S *pmat, const S *loss_derivative,
                 std::index_sequence<Is...>) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr auto vertex_sizes = F::get_vertex_sizes();
  constexpr auto jacobian_size = vertex_sizes[I] * E;

  // Stored as E x d col major, but we need to transpose it to d x E, where d is
  // the vertex size
  const size_t factor_id = active_ids[idx / vertex_sizes[I]];
  const size_t local_id =
      ids[factor_id * N +
          I]; // N is the number of vertices involved in the factor
  if (is_fixed(fixed, local_id)) {
    return;
  }
  const auto jacobian_offset = factor_id * jacobian_size;
  const auto error_offset = factor_id * E;
  // constexpr auto col_offset = I*E; // for untransposed J
  const auto col_offset = (idx % vertex_sizes[I]) * E; // for untransposed J

  // Use loss kernel
  const T dL = loss_derivative[factor_id];

  T value = 0.0;
  constexpr auto precision_matrix_size = E * E;
  const auto precision_offset = factor_id * precision_matrix_size;
  T x2[E] = {0.0};

#pragma unroll
  for (int i = 0; i < E; i++) { // pmat row
#pragma unroll
    for (int j = 0; j < E; j++) { // pmat col
      // x2[i] += pmat[precision_offset + i + j*E] * error[error_offset + j]; //
      // col major
      x2[i] += dL * (T)pmat[precision_offset + i * E + j] *
               static_cast<T>(
                   error[error_offset + j]); // row major (use for faster access
                                             // on symmetrical matrix)
    }
  }

#pragma unroll
  for (int i = 0; i < E; i++) {
    value -= (T)jacs[jacobian_offset + col_offset + i] * x2[i];
  }

  const auto hessian_offset =
      hessian_ids[local_id]; // each vertex has a hessian_ids array

  // printf("Hessian offset: %u\n", hessian_offset);
  // printf("Adding b[%d] += %f\n", hessian_offset + (idx % vertex_sizes[I]),
  // value); printf("Thread %d, Hessian offset: %u\n", idx, hessian_offset);

  atomicAdd(&b[hessian_offset + (idx % vertex_sizes[I])], value);
}

template <typename T, typename S, size_t I, size_t N, typename M, size_t E,
          typename F, typename VT, std::size_t... Is>
__global__ void
compute_b_dynamic_kernel(T *b, const T *error, const size_t *ids,
                         const size_t *hessian_ids, const size_t num_threads,
                         const VT args, const M *obs, const T *jacobian_scales,
                         const typename F::ConstraintDataType *constraint_data,
                         const uint32_t *fixed, const S *pmat,
                         const S *loss_derivative, std::index_sequence<Is...>) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr auto vertex_sizes = F::get_vertex_sizes();
  constexpr auto jacobian_size = vertex_sizes[I] * E;

  // Stored as E x d col major, but we need to transpose it to d x E, where d is
  // the vertex size
  const size_t factor_id = idx / vertex_sizes[I];
  const size_t local_id =
      ids[factor_id * N +
          I]; // N is the number of vertices involved in the factor
  if (is_fixed(fixed, local_id)) {
    return;
  }
  const auto jacobian_offset = factor_id * jacobian_size;
  const auto error_offset = factor_id * E;
  // constexpr auto col_offset = I*E; // for untransposed J
  const auto col_offset = (idx % vertex_sizes[I]) * E; // for untransposed J

  // Use loss kernel
  const T dL = loss_derivative[factor_id];

  T value = 0.0;
  constexpr auto precision_matrix_size = E * E;
  const auto precision_offset = factor_id * precision_matrix_size;
  T x2[E] = {0.0};

#pragma unroll
  for (int i = 0; i < E; i++) { // pmat row
#pragma unroll
    for (int j = 0; j < E; j++) { // pmat col
      // x2[i] += pmat[precision_offset + i + j*E] * error[error_offset + j]; //
      // col major
      x2[i] += dL * (T)pmat[precision_offset + i * E + j] *
               static_cast<T>(
                   error[error_offset + j]); // row major (use for faster access
                                             // on symmetrical matrix)
    }
  }

  using G = std::conditional_t<is_low_precision<S>::value, T, S>;
  G jacobian[jacobian_size];

  compute_Jblock<T, G, I, N, M, E, F, VT>(jacobian, factor_id, local_id, obs,
                                          constraint_data, ids, hessian_ids,
                                          args, std::make_index_sequence<N>{});
  const auto hessian_offset = hessian_ids[local_id];
  const auto scale = jacobian_scales[hessian_offset + (idx % vertex_sizes[I])];
#pragma unroll
  for (int i = 0; i < E; i++) {
    value -= jacobian[col_offset + i] * x2[i];
  }
  value *= scale;

  atomicAdd(&b[hessian_offset + (idx % vertex_sizes[I])], value);
}

// Compute J * x where the length of vector x matches the Hessian dimension
// Each Jacobian block needs to be accessed just once
// So we need E threads for each block (error dimension)
// In total we should hae E*num_factors threads?
template <typename T, typename S, size_t I, size_t N, size_t E, size_t D,
          typename F, std::size_t... Is>
__global__ void compute_Jv_kernel(T *y, const T *x, 
                                  const size_t* active_ids, const size_t *ids,
                                  const size_t *hessian_ids,
                                  const size_t num_threads, const S *jacs,
                                  const uint32_t *fixed,
                                  std::index_sequence<Is...>) {

  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr auto jacobian_size = D * E;

  // Each J block is stored as E x d col major, where d is the vertex size
  const size_t factor_id = active_ids[idx / E];
  const size_t local_id =
      ids[factor_id * N +
          I]; // N is the number of vertices involved in the factor
  if (is_fixed(fixed, local_id)) {
    return;
  }
  const auto jacobian_offset = factor_id * jacobian_size;

  T value = 0;

  const auto hessian_offset =
      hessian_ids[local_id]; // each vertex has a hessian_ids array
  const auto row_offset = (idx % E);
  // Adding i*E skips to the next column
  // size_t residual_offset = 0; // need to pass this in
  // it's the offset into the r vector
  // #pragma unroll
  // for (int i = 0; i < d; i++) {
  //     value += jacs[jacobian_offset + row_offset + i*E] * x[hessian_offset +
  //     i];
  // }

  const S *jrow = jacs + jacobian_offset + row_offset;
  const T *x_start = x + hessian_offset;

#pragma unroll
  for (int i = 0; i < D; i++) {
    value += (T)(jrow[i * E] * (S)x_start[i]);
  }

  atomicAdd(&y[idx], value);
  // y[idx] += value; // avoid unless sure that atomicAdd is not needed
}

/*
template <typename T, typename S, size_t I, size_t N, size_t E, size_t D,
          typename F, std::size_t... Is>
__global__ void
compute_Jv_dynamic_kernel(T *y, T *x, size_t *ids, const size_t *hessian_ids,
                  const size_t num_threads, const S *jacs,
                  const uint32_t *fixed, std::index_sequence<Is...>) {

  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr auto jacobian_size = D * E;

  // Each J block is stored as E x d col major, where d is the vertex size
  const size_t factor_id = idx / E;
  const size_t local_id =
      ids[factor_id * N +
          I]; // N is the number of vertices involved in the factor
  if (is_fixed(fixed, local_id)) {
    return;
  }
  const auto jacobian_offset = factor_id * jacobian_size;

  T value = 0;

  const auto hessian_offset =
      hessian_ids[local_id]; // each vertex has a hessian_ids array
  const auto row_offset = (idx % E);

  const S *jrow = jacs + jacobian_offset + row_offset;
  const T *x_start = x + hessian_offset;

#pragma unroll
  for (int i = 0; i < D; i++) {
    value += (T)(jrow[i * E] * (S)x_start[i]);
  }

  atomicAdd(&y[idx], value);
  // y[idx] += value; // avoid unless sure that atomicAdd is not needed
}
*/

template <typename T, typename G, size_t I, size_t N, typename M, size_t E,
          typename F, typename VT, std::size_t... Is>
__device__ void
compute_Jcol_ad(Dual<T, G> *error, const size_t col, const size_t factor_id,
                const size_t vertex_id, const M *obs,
                const typename F::ConstraintDataType *constraint_data,
                size_t *ids, const size_t *hessian_ids, VT args,
                std::index_sequence<Is...>) {

  constexpr auto vertex_sizes = F::get_vertex_sizes();

  const M *local_obs = obs + factor_id;
  // Dual<T, G> local_error[E];
  const typename F::ConstraintDataType *local_data =
      constraint_data + factor_id;

  auto v = cuda::std::make_tuple(std::array<Dual<T, G>, vertex_sizes[Is]>{}...);

  auto vargs =
      std::make_tuple((*(std::get<Is>(args) + ids[factor_id * N + Is]))...);

  auto copy_vertices = [&v, &vertex_sizes, &vargs](auto &&...ptrs) {
    ((real_to_dual<
         decltype(std::get<Is>(vargs)),
         std::tuple_element<Is, typename F::Traits::VertexDescriptors>::type, T,
         G, vertex_sizes[Is]>(std::get<Is>(vargs),
                              cuda::std::get<Is>(v).data())),
     ...);
  };

  std::apply(copy_vertices, vargs);

  cuda::std::get<I>(v)[col].dual = static_cast<G>(1);

  F::Traits::error(cuda::std::get<Is>(v).data()..., local_obs, error, vargs,
                   local_data);
}

template <typename T, typename S, size_t I, size_t N, typename M, size_t E,
          typename F, typename VT, size_t rows_per_tb, std::size_t... Is>
__global__ void compute_Jv_dynamic_autodiff(
    T *y, T *x, const M *obs, const T *jacobian_scales,
    const typename F::ConstraintDataType *constraint_data, size_t *ids,
    const size_t *hessian_ids, const size_t num_factors, VT args,
    const uint32_t *fixed, std::index_sequence<Is...>) {
  const size_t idx = get_thread_id();
  constexpr size_t D = F::get_vertex_sizes()[I];
  // const size_t rows_per_tb =
  //     blockDim.x / D;
  constexpr size_t rows_per_jacobian = E;

  // const size_t thread_block_idx = blockIdx.x;

  // Assume that number of threads in a block is a multiple of D (row length)
  const size_t row_idx = idx / D;
  const size_t factor_id = row_idx / rows_per_jacobian;
  const size_t column = idx % D;
  const size_t row_in_jacobian = row_idx % rows_per_jacobian;
  const size_t row_in_tb = row_idx % rows_per_tb;

  __shared__ T product_rows[rows_per_tb][D];

  // T product = 0.0;
  if (threadIdx.x < rows_per_tb * D) {
    product_rows[row_in_tb][column] = 0.0;
  }

  if (factor_id < num_factors) {

    const auto vertex_id = ids[factor_id * N + I];

    const auto hess_col = hessian_ids[vertex_id];
    const T *x_start = x + hess_col;

    const T scale = jacobian_scales[hess_col + column];
    // T product = 0.0;
    // Each thread block stores a complete Jacobian row in shared memory
    if (!is_fixed(fixed, vertex_id)) {
      using G = std::conditional_t<is_low_precision<S>::value, T, S>;
      Dual<T, G> error[E];
      compute_Jcol_ad<T, G, I, N, M, E, F, VT>(
          error, column, factor_id, vertex_id, obs, constraint_data, ids,
          hessian_ids, args, std::make_index_sequence<N>{});
      const T scaled_j = static_cast<T>(error[row_in_jacobian].dual) * scale;
      ;
      const T product = static_cast<T>((S)scaled_j * (S)x_start[column]);
      product_rows[row_in_tb][column] = product;
      // product_rows[row_idx % rows_per_tb][column] =
      // error[row_in_jacobian].dual;
    }
    // product_rows[row_in_tb][column] = product;
  }

  // The Jacobian is stored as E x vertex_size in col major
  // constexpr auto jacobian_size = E * D;

  __syncthreads();

  if (factor_id < num_factors) {

    const auto vertex_id = ids[factor_id * N + I];
    if (!is_fixed(fixed, vertex_id)) {
      // const auto hess_col = hessian_ids[vertex_id];
      // const T *x_start = x + hess_col;

      // const T scale = jacobian_scales[hess_col + column];
      // const T* scales = jacobian_scales + hess_col;
      T sum = 0.0;
// TODO: do a warp reduction
#pragma unroll
      for (size_t i = 0; i < D; i++) {
        sum += product_rows[row_in_tb][i];
      }

      if (column == 0) {
        // printf("Thread %llu, Factor %llu, Row %llu, Column %llu, Sum: %f\n",
        // idx, factor_id, row_in_jacobian, column, sum);
        atomicAdd(&y[factor_id * E + row_in_jacobian], sum);
      }
    }
  }
}

template <typename T, typename S, size_t I, size_t N, typename M, size_t E,
          typename F, typename VT, size_t rows_per_tb, std::size_t... Is>
__global__ void
compute_Jv_dynamic_manual(T *y, T *x, const M *obs, const T *jacobian_scales,
                          const typename F::ConstraintDataType *constraint_data,
                          size_t *ids, const size_t *hessian_ids,
                          const size_t num_factors, VT args,
                          const uint32_t *fixed, std::index_sequence<Is...>) {
  const size_t idx = get_thread_id();
  constexpr size_t D = F::get_vertex_sizes()[I];
  // const size_t rows_per_tb =
  //     blockDim.x / D;
  constexpr size_t rows_per_jacobian = E;

  // const size_t thread_block_idx = blockIdx.x;

  // Assume that number of threads in a block is a multiple of D (row length)
  const size_t row_idx = idx / D;
  const size_t factor_id = row_idx / rows_per_jacobian;
  const size_t column = idx % D;
  const size_t row_in_jacobian = row_idx % rows_per_jacobian;
  const size_t row_in_tb = row_idx % rows_per_tb;

  // __shared__ T product_rows[rows_per_tb][D];

  // T product = 0.0;
  // if (threadIdx.x < rows_per_tb * D) {
  //   product_rows[row_in_tb][column] = 0.0;
  // }

  if (factor_id < num_factors) {

    const auto vertex_id = ids[factor_id * N + I];

    const auto hess_col = hessian_ids[vertex_id];
    const T *x_start = x + hess_col;

    // const T scale = jacobian_scales[hess_col + column];
    // T product = 0.0;
    // Each thread block stores a complete Jacobian row in shared memory
    if (!is_fixed(fixed, vertex_id)) {
      using G = std::conditional_t<is_low_precision<S>::value, T, S>;
      // Dual<T, G> error[E];
      constexpr auto jacobian_size = E * D;
      G jacobian[jacobian_size];
      compute_Jblock<T, G, I, N, M, E, F, VT>(
          jacobian, factor_id, vertex_id, obs, constraint_data, ids,
          hessian_ids, args, std::make_index_sequence<N>{});
      // const T scaled_j = static_cast<T>(error[row_in_jacobian].dual) * scale;
      // const T product = static_cast<T>((S)scaled_j * (S)x_start[column]);
      // product_rows[row_in_tb][column] = product;
      // product_rows[row_idx % rows_per_tb][column] =
      // error[row_in_jacobian].dual;

      // T sum = 0.0;
      // for (size_t i = 0; i < D; i++) {
      //   const T scale = jacobian_scales[hess_col + i]
      //   sum += (static_cast<T>(jacobian[i * E + column])*scale) *
      //   (S)x_start[i];
      // }
      const T scale = jacobian_scales[hess_col + column];
      const T scaled_j =
          static_cast<T>(jacobian[row_in_jacobian + column * E]) * scale;
      const T sum = static_cast<T>((S)scaled_j * (S)x_start[column]);
      atomicAdd(&y[factor_id * E + row_in_jacobian], sum);
    }
    // product_rows[row_in_tb][column] = product;
  }

  // The Jacobian is stored as E x vertex_size in col major
  // constexpr auto jacobian_size = E * D;
  /*
  __syncthreads();

  if (factor_id < num_factors) {

    const auto vertex_id = ids[factor_id * N + I];
    if (!is_fixed(fixed, vertex_id)) {
      // const auto hess_col = hessian_ids[vertex_id];
      // const T *x_start = x + hess_col;

      // const T scale = jacobian_scales[hess_col + column];
      // const T* scales = jacobian_scales + hess_col;
      T sum = 0.0;
// TODO: do a warp reduction
#pragma unroll
      for (size_t i = 0; i < D; i++) {
        sum += product_rows[row_in_tb][i];
      }

      if (column == 0) {
        atomicAdd(&y[factor_id * E + row_in_jacobian], sum);
      }
    }
  }
  */
}

template <typename T, typename S, size_t I, size_t N, typename M, size_t E,
          typename F, typename VT, std::size_t... Is>
__global__ void compute_Jv_dynamic_manual2(
    T *y, T *x, const M *obs, const T *jacobian_scales,
    const typename F::ConstraintDataType *constraint_data, size_t *ids,
    const size_t *hessian_ids, const size_t num_factors, VT args,
    const uint32_t *fixed, std::index_sequence<Is...>) {
  const size_t idx = get_thread_id();
  constexpr size_t D = F::get_vertex_sizes()[I];

  const size_t factor_id = idx / E;
  const size_t row_in_jacobian = idx % E;
  if (factor_id < num_factors) {

    const auto vertex_id = ids[factor_id * N + I];

    const auto hess_col = hessian_ids[vertex_id];
    const T *x_start = x + hess_col;

    // Each thread block stores a complete Jacobian row in shared memory
    if (!is_fixed(fixed, vertex_id)) {
      using G = std::conditional_t<is_low_precision<S>::value, T, S>;
      // Dual<T, G> error[E];
      constexpr auto jacobian_size = E * D;
      G jacobian[jacobian_size];
      compute_Jblock<T, G, I, N, M, E, F, VT>(
          jacobian, factor_id, vertex_id, obs, constraint_data, ids,
          hessian_ids, args, std::make_index_sequence<N>{});

      T sum = 0.0;
#pragma unroll
      for (size_t i = 0; i < D; i++) {
        const T scaled_j = static_cast<T>(jacobian[row_in_jacobian + i * E]) *
                           jacobian_scales[hess_col + i];
        sum += static_cast<T>((S)scaled_j * (S)x_start[i]);
      }
      atomicAdd(&y[idx], sum);
    }
  }
}

// Compute J^T * x where x is the size of the residual vector
// Each Jacobian block needs to be accessed just once
// For each block, we need d threads where d is the vertex size
// We need to load the x vector location for the corresponding block row of J
// So this assumes that the x vector has the same layout as the residual vector
// for this factor (rather than a global residual vector) The aggregate output
// will be H x len(x) where H is hessian dimension
// Compute J^T * P * x where P is the precision matrix
template <typename T, typename S, size_t I, size_t N, size_t E, size_t D,
          typename F, std::size_t... Is>
__global__ void compute_JtPv_kernel(T *y, const T *x, const size_t* active_ids,
  const size_t *ids,
                                    const size_t *hessian_ids,
                                    const size_t num_threads, const S *jacs,
                                    const uint32_t *fixed, const S *pmat,
                                    const S *chi2_derivative,
                                    const std::index_sequence<Is...>) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr auto jacobian_size = D * E;

  // Stored as E x d col major, but we need to transpose it to d x E, where d is
  // the vertex size
  const size_t factor_id = active_ids[idx / D];
  const size_t local_id =
      ids[factor_id * N +
          I]; // N is the number of vertices involved in the factor
  if (is_fixed(fixed, local_id)) {
    return;
  }
  const auto jacobian_offset = factor_id * jacobian_size;
  const auto error_offset = factor_id * E;
  const auto col_offset = (idx % D) * E; // for untransposed J

  constexpr auto precision_matrix_size = E * E;
  const auto precision_offset = factor_id * precision_matrix_size;

  // Use loss kernel
  // const auto dL = chi2_derivative[factor_id];

  // T x2[E] = {0};
  // T value = 0;

  const S *jcol = jacs + jacobian_offset + col_offset;

  const S *precision_matrix = pmat + precision_offset;
  const T *x_start = x + error_offset;

  // precision matrices are column major (but should be symmetrical?)
  /*
    T value = 0;
  #pragma unroll
    for (int i = 0; i < E; i++) { // pmat row
    const auto p_row = precision_matrix + i*E;
    S x2 = 0;
    if constexpr (E == 2 && is_low_precision<S>::value) {
      using hp2 = typename vec2_type<T>::type;
      using vec2 = typename vec2_type<S>::type;
      const vec2 xs2 = convert_to_low_precision<hp2,
  vec2>(reinterpret_cast<const hp2 *>(x_start)[0]); const vec2 p2 =
  reinterpret_cast<const vec2 *>(p_row)[0]; const auto sum = __hmul2(p2, xs2);
      x2 += sum.x + sum.y;
    }
    else {
      #pragma unroll
        for (int j = 0; j < E; j++) { // pmat col
          // x2[i] += pmat[precision_offset + i + j*E] * x[error_offset + j]; //
  col
          // major x2[i] += pmat[precision_offset + i*E + j] * x[error_offset +
  j];
          // // row major x2[i] += precision_matrix[i*E + j] * x_start[j]; //
  row
          // major
          // x2 += precision_matrix[i * E + j] * (S)x_start[j];
            x2 += p_row[j] * (S)x_start[j];

          // value += (T)((S)jcol[i] * (S)precision_matrix[i * E + j] *
  (S)x_start[j]);
        }
    }
      value += (T)(jcol[i] * x2);
      // x2[i] *= dL;
      // value += dL*jacs[jacobian_offset + col_offset + i] * x2[i];
      // value += jcol[i] * x2[i];
      // value += jcol[i] * x[i];
    }
    value *= (T)chi2_derivative[factor_id];
    */

  T value = 0;
#pragma unroll
  for (int i = 0; i < E; i++) { // pmat row
    const auto p_row = precision_matrix + i * E;
    S x2 = 0;
#pragma unroll
    for (int j = 0; j < E; j++) { // pmat col
      x2 += p_row[j] * (S)x_start[j];
    }
    value += (T)(jcol[i] * x2);
  }

  value *= (T)chi2_derivative[factor_id];

  const auto hessian_offset =
      hessian_ids[local_id]; // each vertex has a hessian_ids array

  atomicAdd(&y[hessian_offset + (idx % D)], value);
}

template <typename T, typename S, size_t I, size_t N, typename M, size_t E,
          size_t D, typename F, typename VT, std::size_t... Is>
__global__ void compute_JtPv_dynamic_kernel(
    T *y, const T *x, const size_t *ids, const size_t *hessian_ids,
    const size_t num_threads, const VT args, const M *obs,
    const T *jacobian_scales,
    const typename F::ConstraintDataType *constraint_data,
    const uint32_t *fixed, const S *pmat, const S *chi2_derivative,
    const std::index_sequence<Is...>) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr auto jacobian_size = D * E;

  // Stored as E x d col major, but we need to transpose it to d x E, where d is
  // the vertex size
  const size_t factor_id = idx / D;
  const size_t local_id =
      ids[factor_id * N +
          I]; // N is the number of vertices involved in the factor
  if (is_fixed(fixed, local_id)) {
    return;
  }
  const auto jacobian_offset = factor_id * jacobian_size;
  const auto error_offset = factor_id * E;
  const auto col_offset = (idx % D) * E; // for untransposed J

  constexpr auto precision_matrix_size = E * E;
  const auto precision_offset = factor_id * precision_matrix_size;

  // Use loss kernel
  // const auto dL = chi2_derivative[factor_id];

  // T x2[E] = {0};
  // T value = 0;

  using G = std::conditional_t<is_low_precision<S>::value, T, S>;
  G jacobian[jacobian_size];

  compute_Jblock<T, G, I, N, M, E, F, VT>(jacobian, factor_id, local_id, obs,
                                          constraint_data, ids, hessian_ids,
                                          args, std::make_index_sequence<N>{});

  const auto hessian_offset = hessian_ids[local_id];
  const auto scale = jacobian_scales[hessian_offset + (idx % D)];

  // const S *jcol = jacs + jacobian_offset + col_offset;
  const G *jcol = jacobian + col_offset;

  const S *precision_matrix = pmat + precision_offset;
  const T *x_start = x + error_offset;

  T value = 0;
#pragma unroll
  for (int i = 0; i < E; i++) { // pmat row
    const auto p_row = precision_matrix + i * E;
    S x2 = 0;
#pragma unroll
    for (int j = 0; j < E; j++) { // pmat col
      x2 += p_row[j] * (S)x_start[j];
    }
    value += (T)((S)jcol[i] * x2);
  }

  value *= (T)chi2_derivative[factor_id] * scale;

  atomicAdd(&y[hessian_offset + (idx % D)], value);
}

template <typename T, typename S, size_t E, typename L>
__global__ void
compute_chi2_kernel(T *chi2, S *chi2_derivative, const T *residuals,
                    const size_t num_threads, const S *pmat, const L *loss) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }
  T raw_chi2 = compute_chi2<T, S, E>(residuals, pmat, idx);
  chi2[idx] = loss[idx].loss(raw_chi2);
  chi2_derivative[idx] = loss[idx].loss_derivative(raw_chi2);
}

template <typename highp, typename InvP, typename T, size_t I, size_t N,
          size_t E, size_t D>
__global__ void
compute_hessian_diagonal_kernel(InvP *diagonal_blocks, const T *jacs,
                                const size_t *active_ids, 
                                const size_t *ids, const uint32_t *fixed,
                                const T *pmat, const T *chi2_derivative,
                                const size_t num_threads) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr size_t jacobian_size = D * E;
  constexpr size_t block_size = D * D;

  // Stored as E x d col major, but we need to transpose it to d x E, where d is
  // the vertex size
  const size_t factor_id = active_ids[idx / block_size];
  const size_t local_id =
      ids[factor_id * N +
          I]; // N is the number of vertices involved in the factor
  if (is_fixed(fixed, local_id)) {
    return;
  }
  const auto jacobian_offset = factor_id * jacobian_size;

  constexpr auto precision_matrix_size = E * E;
  const auto precision_offset = factor_id * precision_matrix_size;

  // Identify H block row and column (column major)
  // const size_t row = idx % D;
  // const size_t col = idx / D;

  const size_t offset = idx % block_size;
  const size_t row = offset % D;
  const size_t col = offset / D;

  // left[i]*pmat[i*E+j]*right[i] = h value
  // where i goes from 0 to E
  const T *Jt = jacs + jacobian_offset + row * E;
  const T *J = jacs + jacobian_offset + col * E;

  const T *precision_matrix = pmat + precision_offset;

  highp value = 0;
#pragma unroll
  for (int i = 0; i < E; i++) { // pmat row
    highp pj = 0;
#pragma unroll
    for (int j = 0; j < E; j++) { // pmat col
      pj += (highp)precision_matrix[i * E + j] * (highp)J[j];
    }
    value += (highp)Jt[i] * pj;
    // value += Jt[i]*J[i];
  }

  // if (row == col) {
  //     value = 1;
  //     // printf("Thread %d, row: %d, col: %d, value: %f\n", idx, row, col,
  //     value);
  //     // printf("D=%d\n", D);
  // }
  // else {
  //     value = 0;
  // }
  value *= (highp)chi2_derivative[factor_id];

  // T* block = diagonal_blocks + local_id*block_size + (idx % block_size);
  // printf("Thread %d, Hessian offset: %u\n", idx, local_id);
  // if (value != 0) {
  //  printf("Thread %d, row: %d, col: %d, value: %f\n", idx, row, col, value);
  // }
  // if (D == 9 && local_id*block_size + row + col*D > 1701) {
  //     printf("Thread %d, vertex id: %u, row: %d, col: %d, value: %f, offset:
  //     %u \n", idx, local_id, row, col, value, local_id*block_size + row +
  //     col*D);

  // }
  // value = 5.0;
  // T* block = diagonal_blocks + local_id*block_size + row + col*D;
  InvP *block = diagonal_blocks + (local_id * block_size + row + col * D);
  // if (row == col) {
  //     // local_id = 5;
  //     // row = 6; col = 7;
  //     // value = 8.5;
  //     printf("Thread %d, vertex id: %llu, row: %llu, col: %llu, value: %f,
  //     offset: %u \n", idx, local_id, row, col, value, local_id*block_size +
  //     row + col*D);
  // }
  // if (row == 8 && col == 8) {
  //     printf("Number of threads: %d\n", num_threads);
  //     printf("Thread %d, vertex id: %u, row: %zu, col: %zu, value: %f\n",
  //     idx, local_id, row, col, value);
  // }
  // T* block = diagonal_blocks + row*D + col;
  InvP lp_value = static_cast<InvP>(value);
  atomicAdd(block, lp_value);
  // block[0] = value;
  // *block = value;
}

template <typename highp, typename InvP, typename T, size_t I, size_t N,
          size_t E, size_t D, typename VT, typename F>
__global__ void compute_hessian_diagonal_dynamic_kernel(
    InvP *diagonal_blocks, const size_t *ids, const size_t *hessian_ids,
    const VT args, const typename F::ObservationType *obs,
    const highp *jacobian_scales,
    const typename F::ConstraintDataType *constraint_data,
    const uint32_t *fixed, const T *pmat, const T *chi2_derivative,
    const size_t num_threads) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr size_t jacobian_size = D * E;
  constexpr size_t block_size = D * D;

  // Stored as E x d col major, but we need to transpose it to d x E, where d is
  // the vertex size
  const size_t factor_id = idx / block_size;
  const size_t local_id =
      ids[factor_id * N +
          I]; // N is the number of vertices involved in the factor
  if (is_fixed(fixed, local_id)) {
    return;
  }
  const auto jacobian_offset = factor_id * jacobian_size;

  constexpr auto precision_matrix_size = E * E;
  const auto precision_offset = factor_id * precision_matrix_size;

  // Identify H block row and column (column major)
  // const size_t row = idx % D;
  // const size_t col = idx / D;

  const size_t offset = idx % block_size;
  const size_t row = offset % D;
  const size_t col = offset / D;

  using G = std::conditional_t<is_low_precision<T>::value, highp, T>;
  G jacobian[jacobian_size];

  compute_Jblock<highp, G, I, N, typename F::ObservationType, E, F, VT>(
      jacobian, factor_id, local_id, obs, constraint_data, ids, hessian_ids,
      args, std::make_index_sequence<N>{});

  const auto hessian_offset = hessian_ids[local_id];

  const highp *Jt = jacobian + row * E;
  const highp *J = jacobian + col * E;

  const T *precision_matrix = pmat + precision_offset;

  const highp *jscale = jacobian_scales + hessian_offset;

  highp value = 0;
#pragma unroll
  for (int i = 0; i < E; i++) { // pmat row
    highp pj = 0;
#pragma unroll
    for (int j = 0; j < E; j++) { // pmat col
      pj += (highp)precision_matrix[i * E + j] * (highp)J[j] * jscale[col];
    }
    value += (highp)Jt[i] * jscale[row] * pj;
  }

  value *= (highp)chi2_derivative[factor_id];

  InvP *block = diagonal_blocks + (local_id * block_size + row + col * D);

  InvP lp_value = static_cast<InvP>(value);
  atomicAdd(block, lp_value);
}

template <typename highp, typename T, size_t I, size_t N, size_t E, size_t D>
__global__ void
scale_jacobians_kernel(T *jacs, const highp *jacobian_scales,
                        const size_t *active_ids, 
                       const size_t *ids,
                       const size_t *hessian_ids, const uint32_t *fixed,
                       const size_t num_threads) {

  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr size_t jacobian_size = D * E;

  // This time, each factor will have D threads
  const size_t factor_id = active_ids[idx / D];
  const size_t col = idx % D;

  // Stored as E x d col major, but we need to transpose it to d x E, where d is
  // the vertex size
  const size_t local_id =
      ids[factor_id * N +
          I]; // N is the number of vertices involved in the factor
  if (is_fixed(fixed, local_id)) {
    return;
  }

  const auto jacobian_offset = factor_id * jacobian_size;
  const size_t hessian_offset = hessian_ids[local_id];
  const highp scale = jacobian_scales[hessian_offset + col];

  T *Jcol = jacs + jacobian_offset + col * E;
#pragma unroll
  for (size_t i = 0; i < E; i++) {

    const highp scaled_j = static_cast<highp>(Jcol[i]) * scale;

    Jcol[i] = static_cast<T>(scaled_j);
  }
}

template <typename highp, typename T, size_t I, size_t N, size_t E, size_t D>
__global__ void compute_hessian_scalar_diagonal_kernel(
    highp *diagonal, const T *jacs, 
    const size_t *active_ids,
    const size_t *ids,
    const size_t *hessian_ids, const uint32_t *fixed, const T *pmat,
    const T *chi2_derivative, const size_t num_threads) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr size_t jacobian_size = D * E;

  // Stored as E x d col major, but we need to transpose it to d x E, where d is
  // the vertex size
  const size_t factor_id = active_ids[idx / D];
  const size_t local_id =
      ids[factor_id * N +
          I]; // N is the number of vertices involved in the factor
  if (is_fixed(fixed, local_id)) {
    return;
  }
  const auto jacobian_offset = factor_id * jacobian_size;

  constexpr auto precision_matrix_size = E * E;
  const auto precision_offset = factor_id * precision_matrix_size;

  // Identify H block row and column (column major)
  const size_t row = idx % D;
  const size_t col = row;

  // left[i]*pmat[i*E+j]*right[i] = h value
  // where i goes from 0 to E
  const T *Jt = jacs + jacobian_offset + row * E;
  const T *J = jacs + jacobian_offset + col * E;

  const T *precision_matrix = pmat + precision_offset;

  highp value = 0;
#pragma unroll
  for (int i = 0; i < E; i++) { // pmat row
    highp pj = 0;
#pragma unroll
    for (int j = 0; j < E; j++) { // pmat col
      pj += (highp)precision_matrix[i * E + j] * (highp)J[j];
    }
    value += (highp)Jt[i] * pj;
  }

  value *= (highp)chi2_derivative[factor_id];

  // T* block = diagonal_blocks+(local_id*block_size + row + col*D);
  const size_t hessian_offset = hessian_ids[local_id];
  highp *location = diagonal + hessian_offset + row;
  // T lp_value = static_cast<T>(value);
  atomicAdd(location, value);
}

template <typename highp, typename T, size_t I, size_t N, size_t E, size_t D,
          typename VT, typename F, bool use_scales>
__global__ void compute_hessian_scalar_diagonal_dynamic_kernel(
    highp *diagonal, const T *jacs, const size_t *ids,
    const size_t *hessian_ids, const VT args,
    const typename F::ObservationType *obs, const highp *jacobian_scales,
    const typename F::ConstraintDataType *constraint_data,
    const uint32_t *fixed, const T *pmat, const T *chi2_derivative,
    const size_t num_threads) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr size_t jacobian_size = D * E;

  // Stored as E x d col major, but we need to transpose it to d x E, where d is
  // the vertex size
  const size_t factor_id = idx / D;
  const size_t local_id =
      ids[factor_id * N +
          I]; // N is the number of vertices involved in the factor
  if (is_fixed(fixed, local_id)) {
    return;
  }
  const auto jacobian_offset = factor_id * jacobian_size;

  constexpr auto precision_matrix_size = E * E;
  const auto precision_offset = factor_id * precision_matrix_size;

  // Identify H block row and column (column major)
  const size_t row = idx % D;
  const size_t col = row;

  using G = std::conditional_t<is_low_precision<T>::value, highp, T>;
  G jacobian[jacobian_size];

  compute_Jblock<highp, G, I, N, typename F::ObservationType, E, F, VT>(
      jacobian, factor_id, local_id, obs, constraint_data, ids, hessian_ids,
      args, std::make_index_sequence<N>{});
  const auto hessian_offset = hessian_ids[local_id];
  const highp *Jt = jacobian + row * E;
  const highp *J = jacobian + col * E;

  const T *precision_matrix = pmat + precision_offset;
  highp value = 0;
  if constexpr (use_scales) {
    const highp *jscale = jacobian_scales + hessian_offset;

#pragma unroll
    for (int i = 0; i < E; i++) { // pmat row
      highp pj = 0;
#pragma unroll
      for (int j = 0; j < E; j++) { // pmat col
        pj += (highp)precision_matrix[i * E + j] * (highp)J[j] * jscale[col];
      }
      value += (highp)Jt[i] * jscale[row] * pj;
    }
  } else {

#pragma unroll
    for (int i = 0; i < E; i++) { // pmat row
      highp pj = 0;
#pragma unroll
      for (int j = 0; j < E; j++) { // pmat col
        pj += (highp)precision_matrix[i * E + j] * (highp)J[j];
      }
      value += (highp)Jt[i] * pj;
    }
  }

  value *= (highp)chi2_derivative[factor_id];

  // T* block = diagonal_blocks+(local_id*block_size + row + col*D);
  highp *location = diagonal + hessian_offset + row;
  // T lp_value = static_cast<T>(value);
  atomicAdd(location, value);
}

template <typename T, typename S> class GraphVisitor {
public:
  using InvP = std::conditional_t<is_low_precision<S>::value, T, S>;

private:
  template <typename F, typename VT, std::size_t... Is>
  void launch_kernel_autodiff(
      F *f, std::array<const size_t *, F::get_num_vertices()> &hessian_ids,
      VT &verts, std::array<S *, F::get_num_vertices()> &jacs,
      const size_t num_factors, StreamPool& streams, std::index_sequence<Is...>) {
    (([&] {
       constexpr auto num_vertices = F::get_num_vertices();
       const auto num_threads = num_factors * F::get_vertex_sizes()[Is];
       // std::cout << "Launching autodiff kernel" << std::endl;
       // std::cout << "Num threads: " << num_threads << std::endl;
       size_t threads_per_block = 256;
       size_t num_blocks =
           (num_threads + threads_per_block - 1) / threads_per_block;

       // std::cout << "Checking obs ptr: " << f->device_obs.data().get() <<
       // std::endl; std::cout << "Checking residual ptr: " <<
       // f->residuals.data().get() << std::endl; std::cout << "Checking ids
       // ptr: " << f->device_ids.data().get() << std::endl;

       compute_error_kernel_autodiff<T, S, Is, num_vertices,
                                     typename F::ObservationType, F::error_dim,
                                     F, typename F::VertexPointerPointerTuple>
           <<<num_blocks, threads_per_block, 0, streams.select(Is)>>>(
               f->device_obs.data().get(), f->residuals.data().get(),
               f->data.data().get(), f->active_indices.data().get(), 
               f->device_ids.data().get(),
               hessian_ids[Is], num_threads, verts, jacs[Is],
               f->vertex_descriptors[Is]->get_fixed_mask(),
               std::make_index_sequence<num_vertices>{});
     }()),
     ...);
  }

  template <typename F, typename VT, std::size_t... Is>
  void launch_kernel_jacobians(
      F *f, std::array<const size_t *, F::get_num_vertices()> &hessian_ids,
      VT &verts, std::array<S *, F::get_num_vertices()> &jacs,
      const size_t num_factors, StreamPool& streams, std::index_sequence<Is...>) {
    (([&] {
       constexpr auto num_vertices = F::get_num_vertices();
       const auto num_threads = num_factors;
       size_t threads_per_block = 256;
       size_t num_blocks =
           (num_threads + threads_per_block - 1) / threads_per_block;

       compute_jacobian_kernel<T, S, Is, num_vertices,
                               typename F::ObservationType, F::error_dim, F,
                               typename F::VertexPointerPointerTuple>
           <<<num_blocks, threads_per_block, 0, streams.select(Is)>>>(
               f->device_obs.data().get(), f->residuals.data().get(), jacs[Is],
               f->data.data().get(), f->active_indices.data().get(), f->device_ids.data().get(), num_threads,
               verts, f->vertex_descriptors[Is]->get_fixed_mask(),
               std::make_index_sequence<num_vertices>{});
     }()),
     ...);
  }

  template <typename F, std::size_t... Is>
  void launch_kernel_compute_b(
      F *f, T *b,
      std::array<const size_t *, F::get_num_vertices()> &hessian_ids,
      std::array<S *, F::get_num_vertices()> &jacs, const T *jacobian_scales,
      const size_t num_factors, std::index_sequence<Is...>) {
    (([&] {
       constexpr auto num_vertices = F::get_num_vertices();
       const auto num_threads = num_factors * F::get_vertex_sizes()[Is];
       // std::cout << "Launching compute b kernel" << std::endl;
       // std::cout << "Num threads: " << num_threads << std::endl;
       size_t threads_per_block = 256;
       size_t num_blocks =
           (num_threads + threads_per_block - 1) / threads_per_block;

       // std::cout << "Checking obs ptr: " << f->device_obs.data().get() <<
       // std::endl; std::cout << "Checking residual ptr: " <<
       // f->residuals.data().get() << std::endl; std::cout << "Checking ids
       // ptr: " << f->device_ids.data().get() << std::endl;

       if (f->store_jacobians() || !is_analytical<F>()) {

         compute_b_kernel<T, S, Is, num_vertices, F::error_dim, F>
             <<<num_blocks, threads_per_block>>>(
                 b, f->residuals.data().get(), 
                 f->active_indices.data().get(),
                 f->device_ids.data().get(),
                 hessian_ids[Is], num_threads, jacs[Is],
                 f->vertex_descriptors[Is]->get_fixed_mask(),
                 f->precision_matrices.data().get(),
                 f->chi2_derivative.data().get(),
                 std::make_index_sequence<num_vertices>{});

       } else {
         // std::cout << "Launching compute b dynamic kernel" << std::endl;
         if constexpr (is_analytical<F>()) {
           compute_b_dynamic_kernel<T, S, Is, num_vertices,
                                    typename F::ObservationType, F::error_dim,
                                    F, typename F::VertexPointerPointerTuple>
               <<<num_blocks, threads_per_block>>>(
                   b, f->residuals.data().get(), f->device_ids.data().get(),
                   hessian_ids[Is], num_threads, f->get_vertices(),
                   f->device_obs.data().get(), jacobian_scales,
                   f->data.data().get(),
                   f->vertex_descriptors[Is]->get_fixed_mask(),
                   f->precision_matrices.data().get(),
                   f->chi2_derivative.data().get(),
                   std::make_index_sequence<num_vertices>{});
         }
       }
     }()),
     ...);
  }

  template <typename F, std::size_t... Is>
  void launch_kernel_compute_JtPv(
      F *f, T *out, T *in,
      std::array<const size_t *, F::get_num_vertices()> &hessian_ids,
      std::array<S *, F::get_num_vertices()> &jacs, const T *jacobian_scales,
      const size_t num_factors, cudaStream_t* streams, const size_t num_streams, std::index_sequence<Is...>) {
    (([&] {
       constexpr auto num_vertices = F::get_num_vertices();
       const auto num_threads = num_factors * F::get_vertex_sizes()[Is];
       // std::cout << "Launching compute Jtv kernel" << std::endl;
       // std::cout << "Num threads: " << num_threads << std::endl;
       size_t threads_per_block = 256;
       size_t num_blocks =
           (num_threads + threads_per_block - 1) / threads_per_block;

       // std::cout << "Checking obs ptr: " << f->device_obs.data().get() <<
       // std::endl; std::cout << "Checking residual ptr: " <<
       // f->residuals.data().get() << std::endl; std::cout << "Checking ids
       // ptr: " << f->device_ids.data().get() << std::endl;
       if (f->store_jacobians() || !is_analytical<F>()) {
         compute_JtPv_kernel<T, S, Is, num_vertices, F::error_dim,
                             f->get_vertex_sizes()[Is], F>
             <<<num_blocks, threads_per_block, 0, streams[Is % num_streams]>>>(
                 out, in, f->active_indices.data().get(), f->device_ids.data().get(), hessian_ids[Is],
                 num_threads, jacs[Is],
                 f->vertex_descriptors[Is]->get_fixed_mask(),
                 f->precision_matrices.data().get(),
                 f->chi2_derivative.data().get(),
                 std::make_index_sequence<num_vertices>{});
       } else {
         if constexpr (is_analytical<F>()) {
           compute_JtPv_dynamic_kernel<T, S, Is, num_vertices,
                                       typename F::ObservationType,
                                       F::error_dim, f->get_vertex_sizes()[Is],
                                       F, typename F::VertexPointerPointerTuple>
               <<<num_blocks, threads_per_block, 0, streams[Is % num_streams]>>>(
                   out, in, f->device_ids.data().get(), hessian_ids[Is],
                   num_threads, f->get_vertices(), f->device_obs.data().get(),
                   jacobian_scales, f->data.data().get(),
                   f->vertex_descriptors[Is]->get_fixed_mask(),
                   f->precision_matrices.data().get(),
                   f->chi2_derivative.data().get(),
                   std::make_index_sequence<num_vertices>{});
         }
       }
     }()),
     ...);
  }

  template <typename F, std::size_t... Is>
  void launch_kernel_compute_Jv(
      F *f, T *out, T *in,
      std::array<const size_t *, F::get_num_vertices()> &hessian_ids,
      std::array<S *, F::get_num_vertices()> &jacs, const T *jacobian_scales,
      const size_t num_factors, 
      cudaStream_t* streams, size_t num_streams,
      std::index_sequence<Is...>) {
    (([&] {
       constexpr auto num_vertices = F::get_num_vertices();
       constexpr auto vertex_sizes = F::get_vertex_sizes();
       if (f->store_jacobians() || !is_analytical<F>()) {
         const auto num_threads = num_factors * F::error_dim;

         size_t threads_per_block = 256;
         size_t num_blocks =
             (num_threads + threads_per_block - 1) / threads_per_block;
         compute_Jv_kernel<T, S, Is, num_vertices, F::error_dim,
                           f->get_vertex_sizes()[Is], F>
             <<<num_blocks, threads_per_block, 0, streams[Is % num_streams]>>>(
                 out, in, f->active_indices.data().get(), f->device_ids.data().get(), hessian_ids[Is],
                 num_threads, jacs[Is],
                 f->vertex_descriptors[Is]->get_fixed_mask(),
                 std::make_index_sequence<num_vertices>{});
       }
       /*
       else if (f->use_autodiff()) {
         // TODO: Remove
         constexpr size_t warp_size =
             256; // should be 32 but we don't use warp ops
         constexpr size_t rows_per_block = warp_size / vertex_sizes[Is];
         constexpr size_t threads_per_block = rows_per_block * vertex_sizes[Is];
         constexpr size_t E = F::error_dim;

         const size_t num_jacobian_rows = num_factors * E;
         const size_t num_threads = num_jacobian_rows * vertex_sizes[Is];

         const size_t num_blocks =
             (num_threads + threads_per_block - 1) / threads_per_block;
         compute_Jv_dynamic_autodiff<
             T, S, Is, num_vertices, typename F::ObservationType, E, F,
             typename F::VertexPointerPointerTuple, rows_per_block>
             <<<num_blocks, threads_per_block>>>(
                 out, in, f->device_obs.data().get(), jacobian_scales,
                 f->data.data().get(), f->device_ids.data().get(),
                 hessian_ids[Is], num_factors, f->get_vertices(),
                 f->vertex_descriptors[Is]->get_fixed_mask(),
                 std::make_index_sequence<num_vertices>{});
       } */
       else {
         // using manual jacobians
         /*
              constexpr size_t warp_size = 256;
          constexpr size_t rows_per_block = warp_size / vertex_sizes[Is];
          constexpr size_t threads_per_block = rows_per_block *
          vertex_sizes[Is]; constexpr size_t E = F::error_dim;

          const size_t num_jacobian_rows = num_factors * E;
          const size_t num_threads = num_jacobian_rows * vertex_sizes[Is];

          const size_t num_blocks =
              (num_threads + threads_per_block - 1) / threads_per_block;
          compute_Jv_dynamic_manual<
              T, S, Is, num_vertices, typename F::ObservationType, E, F,
              typename F::VertexPointerPointerTuple, rows_per_block>
              <<<num_blocks, threads_per_block>>>(
                  out, in, f->device_obs.data().get(), jacobian_scales,
                  f->data.data().get(), f->device_ids.data().get(),
                  hessian_ids[Is], num_factors, f->get_vertices(),
                  f->vertex_descriptors[Is]->get_fixed_mask(),
                  std::make_index_sequence<num_vertices>{});
                  */

         // Faster
         constexpr auto num_vertices = F::get_num_vertices();
         constexpr auto vertex_sizes = F::get_vertex_sizes();
         const auto num_threads = num_factors * F::error_dim;

         size_t threads_per_block = 256;
         size_t num_blocks =
             (num_threads + threads_per_block - 1) / threads_per_block;
         constexpr size_t E = F::error_dim;

         if constexpr (is_analytical<F>()) {

           compute_Jv_dynamic_manual2<T, S, Is, num_vertices,
                                      typename F::ObservationType, E, F,
                                      typename F::VertexPointerPointerTuple>
               <<<num_blocks, threads_per_block, 0, streams[Is % num_streams]>>>(
                   out, in, f->device_obs.data().get(), jacobian_scales,
                   f->data.data().get(), f->device_ids.data().get(),
                   hessian_ids[Is], num_factors, f->get_vertices(),
                   f->vertex_descriptors[Is]->get_fixed_mask(),
                   std::make_index_sequence<num_vertices>{});
         }
       }
     }()),
     ...);
  }

  template <typename F, std::size_t... Is>
  void launch_kernel_block_diagonal(
      F *f, std::array<InvP *, F::get_num_vertices()> &diagonal_blocks,
      std::array<const size_t *, F::get_num_vertices()> &hessian_ids,
      std::array<S *, F::get_num_vertices()> &jacs, const T *jacobian_scales,
      const size_t num_factors, std::index_sequence<Is...>) {
    (([&] {
       constexpr size_t num_vertices = F::get_num_vertices();
       constexpr size_t dimension = F::get_vertex_sizes()[Is];
       const size_t num_threads = num_factors * dimension * dimension;
       // std::cout << "Launching block diagonal kernel" << std::endl;
       // std::cout << "Num threads: " << num_threads << std::endl;
       // std::cout << "dimension: " << dimension << std::endl;
       // std::cout << "num_factors: " << num_factors << std::endl;
       size_t threads_per_block = 256;
       size_t num_blocks =
           (num_threads + threads_per_block - 1) / threads_per_block;

       // std::cout << "Checking obs ptr: " << f->device_obs.data().get() <<
       // std::endl; std::cout << "Checking residual ptr: " <<
       // f->residuals.data().get() << std::endl; std::cout << "Checking ids
       // ptr: " << f->device_ids.data().get() << std::endl;

       if (f->store_jacobians() || !is_analytical<F>()) {
         compute_hessian_diagonal_kernel<T, InvP, S, Is, num_vertices,
                                         F::error_dim, dimension>
             <<<num_blocks, threads_per_block>>>(
                 diagonal_blocks[Is], jacs[Is], 
                 f->active_indices.data().get(),
                 f->device_ids.data().get(),
                 f->vertex_descriptors[Is]->get_fixed_mask(),
                 f->precision_matrices.data().get(),
                 f->chi2_derivative.data().get(), num_threads);
       } else {
         if constexpr (is_analytical<F>()) {
           compute_hessian_diagonal_dynamic_kernel<
               T, InvP, S, Is, num_vertices, F::error_dim, dimension,
               typename F::VertexPointerPointerTuple, F>
               <<<num_blocks, threads_per_block>>>(
                   diagonal_blocks[Is], f->device_ids.data().get(),
                   hessian_ids[Is], f->get_vertices(),
                   f->device_obs.data().get(), jacobian_scales,
                   f->data.data().get(),
                   f->vertex_descriptors[Is]->get_fixed_mask(),
                   f->precision_matrices.data().get(),
                   f->chi2_derivative.data().get(), num_threads);
         }
       }
       //  cudaError_t err = cudaGetLastError();
       //  if (err != cudaSuccess) {
       //    std::cerr << "CUDA error: " << cudaGetErrorString(err) <<
       //    std::endl;
       //  }

       //  err = cudaDeviceSynchronize();
       //  if (err != cudaSuccess) {
       //    std::cerr << "CUDA error after kernel execution: "
       //              << cudaGetErrorString(err) << std::endl;
       //  }
     }()),
     ...);
  }

  template <typename F, std::size_t... Is>
  void launch_kernel_scalar_diagonal(
      F *f, T *diagonal,
      std::array<const size_t *, F::get_num_vertices()> &hessian_ids,
      std::array<S *, F::get_num_vertices()> &jacs, const T *jacobian_scales,
      const size_t num_factors, std::index_sequence<Is...>) {
    (([&] {
       constexpr size_t num_vertices = F::get_num_vertices();
       constexpr size_t dimension = F::get_vertex_sizes()[Is];
       const size_t num_threads = num_factors * dimension;

       size_t threads_per_block = 256;
       size_t num_blocks =
           (num_threads + threads_per_block - 1) / threads_per_block;

       if (f->store_jacobians() || !is_analytical<F>()) {
         compute_hessian_scalar_diagonal_kernel<T, S, Is, num_vertices,
                                                F::error_dim, dimension>
             <<<num_blocks, threads_per_block>>>(
                 diagonal, jacs[Is], 
                 f->active_indices.data().get(),
                 f->device_ids.data().get(),
                 hessian_ids[Is], f->vertex_descriptors[Is]->get_fixed_mask(),
                 f->precision_matrices.data().get(),
                 f->chi2_derivative.data().get(), num_threads);
       } else {
         if constexpr (is_analytical<F>()) {
           if (jacobian_scales == nullptr) {
             compute_hessian_scalar_diagonal_dynamic_kernel<
                 T, S, Is, num_vertices, F::error_dim, dimension,
                 typename F::VertexPointerPointerTuple, F, false>
                 <<<num_blocks, threads_per_block>>>(
                     diagonal, jacs[Is], f->device_ids.data().get(),
                     hessian_ids[Is], f->get_vertices(),
                     f->device_obs.data().get(), nullptr, f->data.data().get(),
                     f->vertex_descriptors[Is]->get_fixed_mask(),
                     f->precision_matrices.data().get(),
                     f->chi2_derivative.data().get(), num_threads);
           } else {
             compute_hessian_scalar_diagonal_dynamic_kernel<
                 T, S, Is, num_vertices, F::error_dim, dimension,
                 typename F::VertexPointerPointerTuple, F, true>
                 <<<num_blocks, threads_per_block>>>(
                     diagonal, jacs[Is], f->device_ids.data().get(),
                     hessian_ids[Is], f->get_vertices(),
                     f->device_obs.data().get(), jacobian_scales,
                     f->data.data().get(),
                     f->vertex_descriptors[Is]->get_fixed_mask(),
                     f->precision_matrices.data().get(),
                     f->chi2_derivative.data().get(), num_threads);
           }
         }
       }
     }()),
     ...);
  }

  template <typename F, std::size_t... Is>
  void launch_kernel_scale_jacobians(
      F *f, T *jacobian_scales,
      std::array<const size_t *, F::get_num_vertices()> &hessian_ids,
      std::array<S *, F::get_num_vertices()> &jacs, const size_t num_factors,
      std::index_sequence<Is...>) {
    (([&] {
       constexpr size_t num_vertices = F::get_num_vertices();
       constexpr size_t dimension = F::get_vertex_sizes()[Is];
       const size_t num_threads = num_factors * dimension;

       size_t threads_per_block = 256;
       size_t num_blocks =
           (num_threads + threads_per_block - 1) / threads_per_block;

       scale_jacobians_kernel<T, S, Is, num_vertices, F::error_dim, dimension>
           <<<num_blocks, threads_per_block>>>(
               jacs[Is], jacobian_scales, 
               f->active_indices.data().get(),
               f->device_ids.data().get(),
               hessian_ids[Is], f->vertex_descriptors[Is]->get_fixed_mask(),
               num_threads);
     }()),
     ...);
  }

public:
  GraphVisitor() = default;

  template <typename F> void compute_error_autodiff(F *f, StreamPool& streams) {
    // Assume autodiff

    // Then for each vertex, we need to compute the error
    constexpr auto num_vertices = F::get_num_vertices();
    constexpr auto vertex_sizes = F::get_vertex_sizes();

    // At this point all necessary data should be on the GPU
    // std::array<T*, num_vertices> verts;
    auto verts = f->get_vertices();
    std::array<S *, num_vertices> jacs;
    std::array<const size_t *, num_vertices> hessian_ids;
    for (int i = 0; i < num_vertices; i++) {
      // verts[i] = f->vertex_descriptors[i]->vertices();
      jacs[i] = f->jacobians[i].data.data().get();
      hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();

      // Important: Must clear Jacobian storage
      thrust::fill(f->jacobians[i].data.begin(), f->jacobians[i].data.end(), 0);
    }

    const auto num_factors = f->active_count();

    launch_kernel_autodiff(f, hessian_ids, verts, jacs, num_factors, streams,
                           std::make_index_sequence<num_vertices>{});
    cudaDeviceSynchronize();
  }

  template <typename F> void compute_jacobians(F *f, StreamPool& streams) {
    if (!(f->store_jacobians() || !is_analytical<F>())) {
      return;
    }
    // Then for each vertex, we need to compute the error
    constexpr auto num_vertices = F::get_num_vertices();
    constexpr auto vertex_sizes = F::get_vertex_sizes();

    // At this point all necessary data should be on the GPU
    // std::array<T*, num_vertices> verts;
    auto verts = f->get_vertices();
    std::array<S *, num_vertices> jacs;
    std::array<const size_t *, num_vertices> hessian_ids;
    for (int i = 0; i < num_vertices; i++) {
      // verts[i] = f->vertex_descriptors[i]->vertices();
      jacs[i] = f->jacobians[i].data.data().get();
      hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();

      // Important: Must clear Jacobian storage
      thrust::fill(f->jacobians[i].data.begin(), f->jacobians[i].data.end(), 0);
    }

    const auto num_factors = f->active_count();

    launch_kernel_jacobians(f, hessian_ids, verts, jacs, num_factors, streams,
                            std::make_index_sequence<num_vertices>{});
    cudaDeviceSynchronize();
  }

  template <typename F> void compute_error(F *f) {
    // Then for each vertex, we need to compute the error
    constexpr auto num_vertices = F::get_num_vertices();
    constexpr auto vertex_sizes = F::get_vertex_sizes();

    // At this point all necessary data should be on the GPU
    auto verts = f->get_vertices();
    // std::array<T*, num_vertices> verts;
    std::array<const size_t *, num_vertices> hessian_ids;
    for (int i = 0; i < num_vertices; i++) {
      // verts[i] = f->vertex_descriptors[i]->x();
      hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();
    }

    constexpr auto error_dim = F::error_dim;
    const auto num_factors = f->active_count();

    const auto num_threads = num_factors;
    size_t threads_per_block = 256;
    size_t num_blocks =
        (num_threads + threads_per_block - 1) / threads_per_block;

    compute_error_kernel<T, num_vertices, typename F::ObservationType,
                         F::error_dim, F, typename F::VertexPointerPointerTuple>
        <<<num_blocks, threads_per_block>>>(
            f->device_obs.data().get(), f->residuals.data().get(),
            f->data.data().get(), f->active_indices.data().get(), 
            f->device_ids.data().get(), num_threads,
            verts, std::make_index_sequence<num_vertices>{});

    cudaDeviceSynchronize();
  }

  template <typename F> void compute_chi2(F *f) {
    // Then for each vertex, we need to compute the error
    constexpr auto num_vertices = F::get_num_vertices();
    constexpr auto vertex_sizes = F::get_vertex_sizes();

    // At this point all necessary data should be on the GPU
    auto verts = f->get_vertices();
    // std::array<T*, num_vertices> verts;
    // std::array<const size_t*, num_vertices> hessian_ids;
    // for (int i = 0; i < num_vertices; i++) {
    // verts[i] = f->vertex_descriptors[i]->x();
    // hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();
    // }

    constexpr auto error_dim = F::error_dim;
    const auto num_factors = f->active_count();

    const auto num_threads = num_factors;
    size_t threads_per_block = 256;
    size_t num_blocks =
        (num_threads + threads_per_block - 1) / threads_per_block;

    compute_chi2_kernel<T, S, F::error_dim><<<num_blocks, threads_per_block>>>(
        f->chi2_vec.data().get(), f->chi2_derivative.data().get(),
        f->residuals.data().get(), num_threads,
        f->precision_matrices.data().get(), f->loss.data().get());

    cudaDeviceSynchronize();
  }

  template <typename F> void compute_b(F *f, T *b, const T *jacobian_scales) {
    constexpr auto num_vertices = F::get_num_vertices();
    constexpr auto vertex_sizes = F::get_vertex_sizes();

    // std::array<T*, num_vertices> verts;
    auto verts = f->get_vertices();
    std::array<S *, num_vertices> jacs;
    std::array<const size_t *, num_vertices> hessian_ids;
    for (int i = 0; i < num_vertices; i++) {
      jacs[i] = f->jacobians[i].data.data().get();
      hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();
    }

    constexpr auto error_dim = F::error_dim;
    const auto num_factors = f->active_count();
    // std::cout << "Computing b for " << num_factors << " factors" << std::endl;
    launch_kernel_compute_b(f, b, hessian_ids, jacs, jacobian_scales,
                            num_factors,
                            std::make_index_sequence<num_vertices>{});
    cudaDeviceSynchronize();
  }

  template <typename F>
  void compute_Jv(F *f, T *out, T *in, const T *jacobian_scales, cudaStream_t* streams, size_t num_streams) {
    constexpr auto num_vertices = F::get_num_vertices();
    constexpr auto vertex_sizes = F::get_vertex_sizes();

    // std::array<T*, num_vertices> verts;
    auto verts = f->get_vertices();
    std::array<S *, num_vertices> jacs;
    std::array<const size_t *, num_vertices> hessian_ids;
    for (int i = 0; i < num_vertices; i++) {
      // verts[i] = f->vertex_descriptors[i]->x();
      jacs[i] = f->jacobians[i].data.data().get();
      hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();
    }

    const auto num_factors = f->active_count();

    launch_kernel_compute_Jv(f, out, in, hessian_ids, jacs, jacobian_scales,
                             num_factors, streams, num_streams,
                             std::make_index_sequence<num_vertices>{});
    cudaDeviceSynchronize();
  }

  template <typename F>
  void compute_Jtv(F *f, T *out, T *in, const T *jacobian_scales, cudaStream_t* streams, size_t num_streams) {
    constexpr auto num_vertices = f->get_num_vertices();
    constexpr auto vertex_sizes = F::get_vertex_sizes();

    // std::array<T*, num_vertices> verts;
    auto verts = f->get_vertices();
    std::array<S *, num_vertices> jacs;
    std::array<const size_t *, num_vertices> hessian_ids;
    for (int i = 0; i < num_vertices; i++) {
      // verts[i] = f->vertex_descriptors[i]->x();
      jacs[i] = f->jacobians[i].data.data().get();
      hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();
    }

    const auto num_factors = f->active_count();

    launch_kernel_compute_JtPv(f, out, in, hessian_ids, jacs, jacobian_scales,
                               num_factors, streams, num_streams,
                               std::make_index_sequence<num_vertices>{});
    cudaDeviceSynchronize();
  }

  template <typename F>
  void compute_block_diagonal(
      F *f, std::array<InvP *, F::get_num_vertices()> &diagonal_blocks,
      const T *jacobian_scales) {

    // Then for each vertex, we need to compute the error
    constexpr auto num_vertices = F::get_num_vertices();
    constexpr auto vertex_sizes = F::get_vertex_sizes();

    // At this point all necessary data should be on the GPU
    // std::array<T*, num_vertices> verts;
    auto verts = f->get_vertices();
    std::array<S *, num_vertices> jacs;
    std::array<const size_t *, num_vertices> hessian_ids;
    for (int i = 0; i < num_vertices; i++) {
      jacs[i] = f->jacobians[i].data.data().get();
      hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();
    }

    const auto num_factors = f->active_count();

    launch_kernel_block_diagonal(f, diagonal_blocks, hessian_ids, jacs,
                                 jacobian_scales, num_factors,
                                 std::make_index_sequence<num_vertices>{});
    cudaDeviceSynchronize();
  }

  template <typename F>
  void compute_scalar_diagonal(F *f, T *diagonal, const T *jacobian_scales) {

    // Then for each vertex, we need to compute the error
    constexpr auto num_vertices = F::get_num_vertices();
    constexpr auto vertex_sizes = F::get_vertex_sizes();

    // At this point all necessary data should be on the GPU
    // std::array<T*, num_vertices> verts;
    auto verts = f->get_vertices();
    std::array<S *, num_vertices> jacs;
    std::array<const size_t *, num_vertices> hessian_ids;
    for (int i = 0; i < num_vertices; i++) {
      jacs[i] = f->jacobians[i].data.data().get();
      hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();
    }

    const auto num_factors = f->active_count();

    launch_kernel_scalar_diagonal(f, diagonal, hessian_ids, jacs,
                                  jacobian_scales, num_factors,
                                  std::make_index_sequence<num_vertices>{});
    cudaDeviceSynchronize();
  }

  template <typename F> void scale_jacobians(F *f, T *jacobian_scales) {

    if (!(f->store_jacobians() || !is_analytical<F>())) {
      return;
    }
    // Then for each vertex, we need to compute the error
    constexpr auto num_vertices = F::get_num_vertices();
    constexpr auto vertex_sizes = F::get_vertex_sizes();

    // At this point all necessary data should be on the GPU
    // std::array<T*, num_vertices> verts;
    auto verts = f->get_vertices();
    std::array<S *, num_vertices> jacs;
    std::array<const size_t *, num_vertices> hessian_ids;
    for (int i = 0; i < num_vertices; i++) {
      jacs[i] = f->jacobians[i].data.data().get();
      hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();
    }

    const auto num_factors = f->active_count();

    launch_kernel_scale_jacobians(f, jacobian_scales, hessian_ids, jacs,
                                  num_factors,
                                  std::make_index_sequence<num_vertices>{});
    cudaDeviceSynchronize();
  }

  template <typename V>
  void apply_step(V *v, const T *delta_x, T *jacobian_scales, cudaStream_t stream) {
    const size_t num_parameters = v->count() * v->dimension();
    const size_t num_threads = v->count();
    const auto threads_per_block = 256;
    const auto num_blocks =
        (num_threads + threads_per_block - 1) / threads_per_block;

    apply_update_kernel<T, S, V, typename V::VertexType>
        <<<num_blocks, threads_per_block, 0, stream>>>(
            v->vertices(), delta_x, jacobian_scales, v->get_hessian_ids(),
            thrust::raw_pointer_cast(v->fixed_mask.data()), num_threads);
    // cudaDeviceSynchronize();

    // return num_parameters;
  }

  template <typename V>
  void augment_block_diagonal(V *v, InvP *block_diagonal, T mu) {
    const size_t num_threads = v->count();
    const auto threads_per_block = 256;
    const auto num_blocks =
        (num_threads + threads_per_block - 1) / threads_per_block;

    augment_hessian_diagonal_kernel<InvP, V::dim>
        <<<num_blocks, threads_per_block>>>(
            block_diagonal, (InvP)mu,
            thrust::raw_pointer_cast(v->fixed_mask.data()), num_threads);
    cudaDeviceSynchronize();
  }

  template <typename V>
  void apply_block_jacobi(V *v, T *z, const T *r, InvP *block_diagonal, cudaStream_t stream) {
    const size_t num_parameters = v->count() * v->dimension();
    const size_t num_threads = num_parameters;
    const auto threads_per_block = 256;
    const auto num_blocks =
        (num_threads + threads_per_block - 1) / threads_per_block;

    apply_block_jacobi_kernel<T, InvP, V::dim>
        <<<num_blocks, threads_per_block, 0, stream>>>(
            z, r, block_diagonal, v->get_hessian_ids(),
            thrust::raw_pointer_cast(v->fixed_mask.data()), num_threads);
    // cudaDeviceSynchronize();
  }
};
} // namespace glso